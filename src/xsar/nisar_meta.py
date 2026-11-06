# -*- coding: utf-8 -*-
"""
NISAR metadata class for xsar.

Wraps the NisarReader (from xarray-nisar package) and implements
the BaseMeta interface required by xsar. Only L1 RSLC products for now.
"""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.control import GroundControlPoint
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon

from .base_meta import BaseMeta
from .utils import haversine, timing

logger = logging.getLogger("xsar.nisar_meta")
logger.addHandler(logging.NullHandler())


class NisarMeta(BaseMeta):
    """
    Handle NISAR dataset metadata.

    A ``NisarMeta`` object can be used with ``xsar.open_dataset``.
    It wraps a ``NisarReader`` from the ``xarray-nisar`` package and
    adapts its interface to the xsar ``BaseMeta`` contract.

    Parameters
    ----------
    name : str
        Path to the NISAR RSLC HDF5 file, or ``NISAR_DS:<path>:<frequency>``
        to choose the sub-band ('A' or 'B'; default 'A' if present, else 'B').
    """

    @timing
    def __init__(self, name):
        super().__init__()

        from xarray_nisar import NisarReader

        frequency = None
        name = os.fspath(name)
        if name.startswith("NISAR_DS:"):
            path, frequency = name[len("NISAR_DS:"):].rsplit(":", 1)
            frequency = frequency or None
        else:
            path = name

        self.reader = NisarReader(path, frequency=frequency)
        # Preserve the selected frequency when reconstructing/pickling metadata.
        self.name = self.reader.datasets_names[0]
        self.short_name = self.reader.short_name
        self.path = str(self.reader.path)
        self.safe = os.path.basename(self.path)
        self.product = self.reader.product
        self.frequency = self.reader.frequency
        self.dt = self.reader.datatree
        self.manifest_attrs = self.reader.manifest_attrs
        self.manifest = None
        self.multidataset = False
        self.subdatasets = gpd.GeoDataFrame(geometry=[], index=[])
        self.dsid = ""
        self.platform = "NISAR"
        self._time_range = None
        self._geoloc = None
        self._dict_coords2ll_cache = None

        for name_feat, feature in self.__class__._mask_features_raw.items():
            self.set_mask_feature(name_feat, feature)

    def _get_time_range(self):
        start, stop = self.reader.time_range
        return pd.Interval(left=pd.Timestamp(start), right=pd.Timestamp(stop), closed="both")

    @property
    def _image_corners(self):
        # (line, sample) des coins de l'image, dans l'ordre attendu par coverage
        n_lines = int(self.image["numberOfLines"])
        n_samples = int(self.image["numberOfSamples"])
        return [(0, 0), (0, n_samples - 1), (n_lines - 1, n_samples - 1), (n_lines - 1, 0)]

    @property
    def footprint(self):
        """Footprint as a shapely Polygon from the image corners."""
        return self.geoloc.attrs["footprint"]

    @property
    def geometry(self):
        """Alias for footprint."""
        return self.footprint

    @property
    def geoloc(self):
        """
        Geolocation grid (height 0 above ellipsoid) with longitude, latitude,
        incidenceAngle, elevationAngle, height, groundHeading on fractional
        full-resolution (line, sample) coordinates.
        """
        if self._geoloc is not None:
            return self._geoloc

        self._geoloc = self.reader.geoloc.copy()
        self._geoloc.attrs = {}
        self._geoloc["longitude"] = self._continuous_longitude(self._geoloc.longitude)

        lines, samples = zip(*self._image_corners)
        lon, lat = self.coords2ll(np.array(lines, float), np.array(samples, float))
        if self.cross_antimeridian:
            lon = lon % 360
        corners = list(zip(lon, lat))
        self._geoloc.attrs["footprint"] = Polygon(corners)

        acq_sample_meters, _ = haversine(*corners[0], *corners[1])
        acq_line_meters, _ = haversine(*corners[1], *corners[2])
        self._geoloc.attrs["coverage"] = "%dkm * %dkm (line * sample )" % (
            acq_line_meters / 1000,
            acq_sample_meters / 1000,
        )

        # approx transform (inexact), à partir de GCPs du sous-échantillon de la grille
        # (col=line, row=sample, comme RadarSat2Meta : approx_transform * (line, sample) -> (lon, lat))
        g = self._geoloc.isel(
            line=slice(None, None, max(1, self._geoloc.sizes["line"] // 20)),
            sample=slice(None, None, max(1, self._geoloc.sizes["sample"] // 20)),
        )
        gcps = [
            GroundControlPoint(row=s, col=l, x=x, y=y, z=0.0)
            for l, row_x, row_y in zip(g.line.values, g.longitude.values, g.latitude.values)
            for s, x, y in zip(g.sample.values, row_x, row_y)
        ]
        self._geoloc.attrs["approx_transform"] = rasterio.transform.from_gcps(gcps)
        return self._geoloc

    @property
    def coverage(self):
        """Coverage string like '240km * 240km (line * sample )'."""
        return self.geoloc.attrs.get("coverage", "unknown")

    @property
    def _dict_coords2ll(self):
        """
        dict with keys ['longitude', 'latitude'] with interpolation function
        (RectBivariateSpline) as values, built on the reader geolocation grid.

        Notes
        -----
        if self.cross_antimeridian is True, 'longitude' will be in range [0, 360]
        """
        if self._dict_coords2ll_cache is None:
            geoloc = self.reader.geoloc
            lon = geoloc["longitude"].values
            if (lon.max() - lon.min()) > 180:
                lon = lon % 360
            idx_line, idx_sample = geoloc.line.values, geoloc.sample.values
            self._dict_coords2ll_cache = {
                "longitude": RectBivariateSpline(idx_line, idx_sample, lon, kx=1, ky=1),
                "latitude": RectBivariateSpline(idx_line, idx_sample, geoloc["latitude"].values, kx=1, ky=1),
            }
        return self._dict_coords2ll_cache

    def ll2coords(self, *args):
        """
        Get (lines, samples) from (lon, lat), exact inverse of coords2ll.

        Overrides BaseMeta.ll2coords (one-step affine correction, ~10 lines
        error at the frame edges): Newton iterations on the bilinear
        geolocation splines, starting from the approx transform.

        Parameters
        ----------
        *args : lon, lat or shapely geometry
            lon and lat might be iterables or scalars

        Returns
        -------
        tuple of np.array or tuple of float
            (lines, samples), or a shapely object if input was shapely
        """
        import shapely.geometry.base

        if isinstance(args[0], shapely.geometry.base.BaseGeometry):
            return self._ll2coords_shapely(args[0])

        lon, lat = args
        scalar = not hasattr(lon, "__iter__")
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        if self.cross_antimeridian:
            lon = lon % 360

        line, sample = ~self.approx_transform * (lon, lat)
        f = self._dict_coords2ll
        for _ in range(10):
            dlon = lon - f["longitude"].ev(line, sample)
            dlat = lat - f["latitude"].ev(line, sample)
            # jacobienne d(lon, lat) / d(line, sample) par différences centrées
            # (FITPACK ne dérive pas une spline de degré 1)
            h = 0.5
            a, c = ((f[k].ev(line + h, sample) - f[k].ev(line - h, sample)) / (2 * h)
                    for k in ("longitude", "latitude"))
            b, d = ((f[k].ev(line, sample + h) - f[k].ev(line, sample - h)) / (2 * h)
                    for k in ("longitude", "latitude"))
            det = a * d - b * c
            dl = (d * dlon - b * dlat) / det
            ds = (a * dlat - c * dlon) / det
            line, sample = line + dl, sample + ds
            if np.nanmax(np.abs(np.concatenate([dl, ds]))) < 1e-3:
                break

        if scalar:
            return float(line[0]), float(sample[0])
        return line, sample

    @property
    def approx_transform(self):
        """
        Affine transform from geolocation grid.

        This is an approximate transform. Use coords2ll / ll2coords for
        accurate coordinate conversions.
        """
        return self.geoloc.attrs["approx_transform"]

    @property
    def swath(self):
        """Swath type (always 'SM': NISAR is stripmap, one subswath)."""
        return "SM"

    @property
    def orbit_pass(self):
        """Orbit pass direction (Ascending/Descending)."""
        return self.manifest_attrs["orbitPassDirection"]

    @property
    def platform_heading(self):
        """Platform ground heading relative to north (degrees), at scene center."""
        return self.orbit.attrs["platform_heading"]

    @property
    def orbit(self):
        """Orbit data as xarray.Dataset."""
        return self.dt["orbit"].to_dataset()

    @property
    def image(self):
        """Image information dataset."""
        return self.dt["image"].to_dataset()

    @property
    def pixel_line_m(self):
        """Pixel line spacing in meters."""
        return self.image["azimuthPixelSpacing"]

    @property
    def pixel_sample_m(self):
        """Pixel sample spacing in meters (ground range, scene center)."""
        return self.image["groundRangePixelSpacing"]

    @property
    def pols(self):
        """Polarizations as a space-separated string."""
        return " ".join(self.manifest_attrs["polarizations"])

    @property
    def denoised(self):
        """Dict with pol as key and bool as value."""
        return self.reader.denoised

    @property
    def get_calibration_luts(self):
        """Calibration LUTs dataset (sigma0_lut, beta0_lut, gamma0_lut)."""
        return self.dt["calibration_luts"].to_dataset()

    @property
    def get_noise_luts(self):
        """Noise LUTs dataset (noise_lut, per pol, in DN^2)."""
        return self.dt["noise_luts"].to_dataset()

    def to_dict(self, keys="minimal"):
        info_keys = {
            "minimal": ["platform", "swath", "product", "pols"],
        }
        info_keys["all"] = info_keys["minimal"] + [
            "name",
            "start_date",
            "stop_date",
            "footprint",
            "coverage",
            "orbit_pass",
            "platform_heading",
            "frequency",
        ]

        if isinstance(keys, str):
            keys = info_keys.get(keys, info_keys["minimal"])

        res_dict = {}
        for k in keys:
            if hasattr(self, k):
                res_dict[k] = getattr(self, k)
            elif k in self.manifest_attrs:
                res_dict[k] = self.manifest_attrs[k]
        return res_dict

    def __repr__(self):
        return "<NisarMeta %s freq %s object>" % (self.product, self.frequency)

    def __reduce__(self):
        return self.__class__, (self.name,), self.dict
