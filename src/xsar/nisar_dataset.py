# -*- coding: utf-8 -*-
"""
NISAR dataset class for xsar.

Loads NISAR L1 RSLC data through NisarReader (via NisarMeta), applies
calibration and denoising, and provides a standard xarray.Dataset
following the xsar conventions (variable names as for Sentinel-1 SLC).

NISAR RSLC is stripmap: one frame is one continuous image (no bursts), i.e.
the equivalent of a large Sentinel-1 WV imagette. All high-resolution
variables are lazy (dask), interpolated block by block from the annotation
grids, so that a full frame (~45000 x 6400 pixels) is never loaded at once.

Calibration (same convention as Sentinel-1, checked on P05023 products):
    sigma0_raw = |DN|^2 / sigma0_lut^2
    nesz       = noise_lut / sigma0_lut^2       (noise_lut in DN^2)
    sigma0     = sigma0_raw - nesz              (not clipped)
and the same for gamma0 / negz with gamma0_lut.
"""

import logging
import warnings

import dask
import dask.array
import numpy as np
import xarray
import xarray as xr
import yaml
from scipy.interpolate import RectBivariateSpline, interp1d

from .base_dataset import BaseDataset
from .nisar_meta import NisarMeta
from .utils import BlockingActorProxy, map_blocks_coords, merge_yaml

logger = logging.getLogger("xsar.nisar_dataset")
logger.addHandler(logging.NullHandler())

C_LIGHT = 299792458.0

# Mapping from dataset variable names to geolocation grid variable names
_MAPPING_DATASET_GEOLOC = {
    "latitude": "latitude",
    "longitude": "longitude",
    "incidence": "incidenceAngle",
    "elevation": "elevationAngle",
    "altitude": "height",
}


def _spline(grid):
    """Bilinear interpolator of a 2D (line, sample) grid on fractional pixel coords."""
    return RectBivariateSpline(grid.line.values, grid.sample.values, np.asarray(grid.values), kx=1, ky=1)


class NisarDataset(BaseDataset):
    """
    Handle a NISAR L1 RSLC product dataset.

    The main attribute is ``self.dataset`` (``xarray.Dataset`` with all
    variables parsed from the product).

    Parameters
    ----------
    dataset_id : str or NisarMeta
        Path to the NISAR RSLC HDF5 file (or ``NISAR_DS:<path>:<frequency>``),
        or a NisarMeta object.
    resolution : None, optional
        Only full resolution is supported (complex SLC).
    luts : bool, optional
        Kept for API compatibility: LUT variables are always included.
    chunks : dict, optional
        Dask chunks for ``{'line': ..., 'sample': ...}``. Multiples of the HDF5
        chunks (512 x 512) are recommended.
    dtypes : dict or None, optional
        Override data types for variables.
    skip_variables : list of str, optional
        Variables not to compute (e.g. ['land_mask', 'ground_heading']).
    """

    def __init__(
        self,
        dataset_id,
        resolution=None,
        luts=False,
        chunks=None,
        dtypes=None,
        skip_variables=None,
    ):
        if resolution is not None:
            raise NotImplementedError("NISAR RSLC: only full resolution is supported")
        if chunks is None:
            chunks = {"line": 4096, "sample": 4096}
        if dtypes is not None:
            self._dtypes = {**self._dtypes, **dtypes}
        skip_variables = set(skip_variables or [])

        self.sar_meta = None
        if not isinstance(dataset_id, NisarMeta):
            self.sar_meta = BlockingActorProxy(NisarMeta, dataset_id)
        else:
            self.sar_meta = BlockingActorProxy(NisarMeta.from_dict, dataset_id.dict)
        del dataset_id

        # Load digital number
        self.resolution, DN_tmp = self.sar_meta.reader.load_digital_number(
            resolution=resolution, chunks=chunks
        )

        geoloc = self.sar_meta.geoloc.copy()
        geoloc.attrs = {"history": "NISAR metadata/geolocationGrid (height 0)"}
        ds_cal = self.sar_meta.get_calibration_luts
        ds_noise = self.sar_meta.get_noise_luts

        self.datatree = xr.DataTree.from_dict(
            {
                "measurement": DN_tmp,
                "geolocation_annotation": geoloc,
                "orbit": self.sar_meta.orbit,
                "image": self.sar_meta.image,
                "calibration": ds_cal,
                "noise": ds_noise,
                "doppler": self.sar_meta.dt["doppler"].to_dataset(),
            }
        )

        self._dataset = self.datatree["measurement"].to_dataset()

        # Add metadata attributes
        for att in ["name", "short_name", "product", "safe", "swath", "multidataset"]:
            if att not in self.datatree.attrs:
                val = self.sar_meta.__getattr__(att)
                self.datatree.attrs[att] = val
                self._dataset.attrs[att] = val

        # Pixel spacing (full resolution, slant range as for Sentinel-1 SLC)
        self._dataset["sampleSpacing"] = xarray.DataArray(
            self.sar_meta.image["slantRangePixelSpacing"],
            attrs={"units": "m", "referential": "slant"},
        )
        self._dataset["lineSpacing"] = xarray.DataArray(
            self.sar_meta.image["azimuthPixelSpacing"], attrs={"units": "m"}
        )

        # Template DataArray for coordinate-based operations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._da_tmpl = xr.DataArray(
                dask.array.empty_like(
                    self._dataset.digital_number.isel(pol=0).drop_vars("pol").data,
                    dtype=np.int8,
                    name="empty_var_tmpl-%s" % dask.base.tokenize(self.sar_meta.name),
                ),
                dims=("line", "sample"),
                coords={
                    "line": self._dataset.digital_number.line,
                    "sample": self._dataset.digital_number.sample,
                },
            )

        self._dataset.attrs.update(self.sar_meta.to_dict("all"))
        self.datatree["measurement"] = self.datatree["measurement"].assign(self._dataset)
        self.datatree.attrs.update(self.sar_meta.to_dict("all"))

        self._add_high_resolution_variables(ds_cal, ds_noise, geoloc, skip_variables)
        self._apply_nisar_calibration()

        self.datatree["measurement"] = self.datatree["measurement"].assign(self._dataset)
        self.datatree["measurement"].attrs = self.datatree.attrs

        self.sliced = False
        self.resampled = False
        self._bbox_coords_ori = self._bbox_coords

    # ---------------------------------------------------------------- helpers
    def _interp_grid(self, grid, name, dtype="f4", attrs=None, key=None):
        """Lazy full-resolution (line, sample) DataArray interpolated from a 2D grid."""
        spline = _spline(grid)

        def _f(lines, samples):
            return spline(lines, samples)

        res = map_blocks_coords(self._da_tmpl.astype(dtype), _f,
                                name=f"{key or name}-{dask.base.tokenize(self.sar_meta.name)}")
        res.attrs = dict(attrs or {})
        return res.rename(name)

    def load_ground_heading(self):
        return super().load_ground_heading(approx=False)

    def _time_vars(self):
        """azimuth_time (from line) and slant_range_time (from sample), lazy 2D as for S1."""
        image = self.sar_meta.image
        t0 = image["firstAzimuthTime"].values
        dt_ns = float(image["azimuthTimeInterval"]) * 1e9
        r0 = float(image["nearSlantRange"])
        dr = float(image["slantRangePixelSpacing"])

        def _azi(lines, samples):
            t = t0 + (lines * dt_ns).astype("timedelta64[ns]")
            return np.broadcast_to(t[:, None], (lines.size, samples.size))

        def _srt(lines, samples):
            srt = 2 * (r0 + samples * dr) / C_LIGHT
            return np.broadcast_to(srt[None, :], (lines.size, samples.size))

        azi = map_blocks_coords(self._da_tmpl.astype("datetime64[ns]"), _azi, name="azimuth_time")
        azi.attrs = {"comment": "zero Doppler azimuth time"}
        srt = map_blocks_coords(self._da_tmpl.astype("f8"), _srt, name="slant_range_time")
        srt.attrs = {"units": "s", "comment": "two-way slant range time"}
        return xr.Dataset({"azimuth_time": azi, "slant_range_time": srt})

    def get_sensor_velocity(self):
        """Sensor velocity norm (m/s) interpolated at each line azimuth time."""
        orbit = self.sar_meta.orbit
        vel = np.sqrt(sum(orbit[f"velocity_{c}"].values ** 2 for c in "xyz"))
        f = interp1d(orbit.time.values.astype(float), vel)
        image = self.sar_meta.image
        lines = self._dataset.line.values
        t = image["firstAzimuthTime"].values + (lines * float(image["azimuthTimeInterval"]) * 1e9).astype(
            "timedelta64[ns]"
        )
        res = xr.DataArray(f(t.astype(float)), dims=["line"], coords={"line": lines},
                           attrs={"units": "m/s", "comment": "sensor velocity norm (ECEF)"})
        return xr.Dataset({"velocity": res})

    # ------------------------------------------------------------- variables
    def _add_high_resolution_variables(self, ds_cal, ds_noise, geoloc, skip_variables):
        """
        Add lazy high-resolution variables: LUTs, geolocation variables,
        times, sensor velocity, valid mask, ground heading and land mask.
        """
        ds_merge_list = [self._dataset]

        # LUTs de calibration et de bruit
        hist_cal = yaml.safe_dump({"sigma0_lut": ds_cal.attrs.get("history", "")})
        luts = {
            "sigma0_lut": self._interp_grid(ds_cal.sigma0_lut, "sigma0_lut", self._dtypes["sigma0_lut"],
                                            {"history": hist_cal}),
            "gamma0_lut": self._interp_grid(ds_cal.gamma0_lut, "gamma0_lut", self._dtypes["gamma0_lut"],
                                            {"history": hist_cal.replace("sigma0_lut", "gamma0_lut")}),
        }
        noise = xr.concat(
            [self._interp_grid(ds_noise.noise_lut.sel(pol=pol), "noise_lut", self._dtypes["noise_lut"],
                               key=f"noise_lut_{pol}")
             for pol in self._dataset.pol.values],
            dim="pol",
        ).assign_coords(pol=self._dataset.pol)
        noise.attrs = {"units": "DN^2", "history": yaml.safe_dump({"noise_lut": ds_noise.attrs.get("history", "")})}
        luts["noise_lut"] = noise
        ds_merge_list.append(xr.Dataset(luts))

        # Géolocalisation
        geo = {}
        for var, gvar in _MAPPING_DATASET_GEOLOC.items():
            if var in skip_variables:
                continue
            grid = geoloc[gvar]
            wraps = var == "longitude" and self.sar_meta.cross_antimeridian
            if wraps:
                grid = grid % 360
            result = self._interp_grid(grid, var, self._dtypes.get(var, "f4"),
                                       geoloc[gvar].attrs)
            if wraps:
                result = (result + 180) % 360 - 180
                result.attrs = geoloc[gvar].attrs
            geo[var] = result
        if geo:
            ds_merge_list.append(xr.Dataset(geo))
        ds_merge_list.append(self._time_vars())

        if "velocity" not in skip_variables:
            ds_merge_list.append(self.get_sensor_velocity())

        if "valid_mask" not in skip_variables:
            chunks = dict(zip(("line", "sample"), self._da_tmpl.data.chunksize))
            vm = self.sar_meta.reader.load_valid_mask(chunks=chunks)
            vm.attrs = {"comment": "validSamplesSubSwath and inputDataExceptionMask == 0"}
            ds_merge_list.append(vm.to_dataset())

        if "ground_heading" not in skip_variables:
            ds_merge_list.append(self.load_ground_heading())

        if "land_mask" not in skip_variables:
            self._rasterized_masks = self.load_rasterized_masks()
            ds_merge_list.append(self._rasterized_masks)

        attrs = self._dataset.attrs
        self._dataset = xr.merge(ds_merge_list, compat="override")
        self._dataset.attrs = attrs

        self.recompute_attrs()
        self.datatree["measurement"] = self.datatree["measurement"].assign(self._dataset)

    def _apply_nisar_calibration(self):
        """sigma0/gamma0 raw, NESZ/NEGZ and denoised sigma0/gamma0 (S1 convention)."""
        dn2 = np.abs(self._dataset.digital_number) ** 2
        for var, noise_var, lut_name in (("sigma0", "nesz", "sigma0_lut"), ("gamma0", "negz", "gamma0_lut")):
            lut2 = self._dataset[lut_name] ** 2
            raw = dn2 / lut2
            raw.attrs = {
                "units": "linear",
                "history": merge_yaml([self._dataset[lut_name].attrs["history"]], section=f"{var}_raw"),
                "comment": f"{var}_raw = |DN|^2 / {lut_name}^2",
            }
            nz = self._dataset["noise_lut"] / lut2
            nz.attrs = {"units": "linear", "comment": f"{var} noise = noise_lut / {lut_name}^2"}
            den = raw - nz
            den.attrs = {"units": "linear", "comment": "not clipped, some values can be <0",
                         "history": raw.attrs["history"]}
            self._dataset[f"{var}_raw"] = raw
            self._dataset[noise_var] = nz
            self._dataset[var] = den

    # --------------------------------------------------------------- dataset
    @property
    def dataset(self):
        """
        ``xarray.Dataset`` representation of this NisarDataset object.
        """
        res = self.datatree["measurement"].to_dataset()
        res.attrs = self.datatree.attrs
        return res

    @dataset.setter
    def dataset(self, ds):
        if self.sar_meta.name == ds.attrs.get("name"):
            if not self.sliced:
                self.sliced = any(
                    [
                        list(ds[d].values) != list(self._dataset[d].values)
                        for d in ["line", "sample"]
                    ]
                )
            self._dataset = ds
            self.datatree["measurement"] = xr.DataTree(self._dataset)
            self.recompute_attrs()
            self.datatree.attrs.update(self._dataset.attrs)
            self.datatree["measurement"] = xr.DataTree(self._dataset)
        else:
            raise ValueError("dataset must be same kind as original one.")

    @dataset.deleter
    def dataset(self):
        logger.debug("deleter dataset")

    def __repr__(self):
        intro = "sliced" if self.sliced else "full coverage"
        return "<NisarDataset %s object>" % intro
