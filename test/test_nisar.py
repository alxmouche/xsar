"""NISAR dispatch, serialization, calibration and Dask regression checks."""

import importlib
import pickle

import dask
import dask.array as da
import h5py
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("xarray_nisar")

import xsar
from xsar.utils import map_blocks_coords
from nisar_product import make_product


@pytest.fixture
def product(tmp_path):
    return make_product(tmp_path / "NISAR_L1_TEST.h5")


def test_meta_roundtrip_preserves_frequency(product):
    meta = xsar.NisarMeta(f"NISAR_DS:{product}:B")
    restored = pickle.loads(pickle.dumps(meta))
    assert restored.frequency == "B"
    assert xsar.NisarMeta.from_dict(meta.dict).frequency == "B"
    line, sample = np.array([1., 3., 6.]), np.array([2., 4., 7.])
    lon, lat = meta.coords2ll(line, sample)
    actual_line, actual_sample = meta.ll2coords(lon, lat)
    np.testing.assert_allclose(actual_line, line, atol=1e-3)
    np.testing.assert_allclose(actual_sample, sample, atol=1e-3)


def test_dataset_calibration(product):
    obj = xsar.NisarDataset(product, chunks={"line": 3, "sample": 4},
                            skip_variables=["land_mask"])
    ds = obj.dataset
    assert isinstance(ds.sigma0.data, da.Array)
    result = ds[["sigma0_raw", "sigma0", "nesz", "gamma0_raw", "gamma0", "negz"]].compute()
    for name, value in {
        "sigma0_raw": 25 / 4, "sigma0": 24.5 / 4, "nesz": .5 / 4,
        "gamma0_raw": 25 / 9, "gamma0": 24.5 / 9, "negz": .5 / 9,
    }.items():
        np.testing.assert_allclose(result[name], value, rtol=1e-6)
    assert ds.valid_mask.isel(line=-1).compute().sum() == 0
    assert set(["measurement", "calibration", "noise"]) <= set(obj.datatree.children)
    sliced = ds.isel(line=slice(1, 5), sample=slice(2, 7))
    obj.dataset = sliced
    assert obj.sliced
    assert obj.dataset.sizes["line"] == 4


@pytest.mark.parametrize("op", ["open_dataset", "open_datatree"])
@pytest.mark.parametrize("kind", ["path", "string", "id", "meta"])
def test_dispatch(product, op, kind):
    if kind == "path":
        source = product
    elif kind == "string":
        source = str(product)
    elif kind == "id":
        source = f"NISAR_DS:{product}:B"
    else:
        source = xsar.NisarMeta(product)
    result = getattr(xsar, op)(source, skip_variables=["land_mask"])
    ds = result if op == "open_dataset" else result["measurement"].to_dataset()
    assert ds.sizes["pol"] == 2
    assert ds.attrs["frequency"] == ("B" if kind == "id" else "A")


def test_coordinate_graphs_are_independent():
    template = xr.DataArray(da.zeros((4, 5), chunks=(2, 3)), dims=("line", "sample"),
                            coords={"line": np.arange(4), "sample": np.arange(5)})

    def build(offset, values=template):
        return map_blocks_coords(values, lambda line, sample: line[:, None] + sample + offset,
                                 name="interpolation")

    first, second, shifted = dask.compute(build(1), build(2), build(1, template.assign_coords(line=np.arange(4) + 10)))
    np.testing.assert_array_equal(second - first, np.ones((4, 5)))
    np.testing.assert_array_equal(shifted.values - first.values, np.full((4, 5), 10))


def test_nisar_does_not_take_sentinel_path(monkeypatch):
    module = importlib.import_module("xsar.xsar")
    sentinel = object()

    class FakeNisar:
        def __init__(self, *args, **kwargs):
            self.dataset = sentinel

    monkeypatch.setattr(module, "NisarDataset", FakeNisar)
    # A parent folder containing S1 must not route a NISAR identifier to Sentinel1.
    assert module.open_dataset("NISAR_DS:/data/S1/archive/arbitrary.h5:B") is sentinel


def test_antimeridian_longitude_interpolation(product):
    with h5py.File(product, "r+") as f:
        geo = f["science/LSAR/RSLC/metadata/geolocationGrid"]
        samples = (geo["slantRange"][()] - 800000) / 10
        longitude = (179.9 + samples * .05 + 180) % 360 - 180
        geo["coordinateX"][...] = np.broadcast_to(longitude, geo["coordinateX"].shape)
    obj = xsar.NisarDataset(product, skip_variables=["land_mask", "ground_heading"])
    longitude = obj.dataset.longitude.isel(line=2).compute().values
    expected = (179.9 + np.arange(10) * .05 + 180) % 360 - 180
    np.testing.assert_allclose(longitude, expected, atol=5e-5)
    assert obj.sar_meta.cross_antimeridian is True
    assert obj.sar_meta.footprint.bounds[2] - obj.sar_meta.footprint.bounds[0] < 1
    line, sample = obj.sar_meta.ll2coords(-179.9, 10.02)
    np.testing.assert_allclose([line, sample], [2., 4.], atol=1e-3)


def test_nisar_mask_across_antimeridian(product):
    from shapely.geometry import box
    from xsar.base_meta import _ShapefileFeature

    with h5py.File(product, "r+") as f:
        geo = f["science/LSAR/RSLC/metadata/geolocationGrid"]
        samples = (geo["slantRange"][()] - 800000) / 10
        longitude = (179.9 + samples * .05 + 180) % 360 - 180
        geo["coordinateX"][...] = np.broadcast_to(longitude, geo["coordinateX"].shape)
    meta = xsar.NisarMeta(product)
    meta.set_mask_feature("test_island", _ShapefileFeature([
        box(179.95, 10.01, 180., 10.04), box(-180., 10.01, -179.95, 10.04)
    ]))
    mask = meta.get_mask("test_island")
    assert mask.is_valid
    assert mask.area == pytest.approx(.1 * .03)
    assert mask.bounds == pytest.approx((179.95, 10.01, 180.05, 10.04))
