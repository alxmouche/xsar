# NISAR RSLC support

Install the companion `xarray-nisar` reader from its private repository at the
chosen release/commit, then install this xsar branch in the same environment.
The `NISAR` extra declares the companion dependency but does not configure
private Git hosting or a package index.

```python
import xsar

meta = xsar.NisarMeta("NISAR_DS:/data/NISAR_L1_PR_RSLC_example.h5:B")
obj = xsar.NisarDataset(meta, chunks={"line": 4096, "sample": 4096})
ds = obj.dataset

# Or let xsar select frequency A when present, otherwise B:
ds = xsar.open_dataset("/data/NISAR_L1_PR_RSLC_example.h5")
tree = xsar.open_datatree("/data/NISAR_L1_PR_RSLC_example.h5")
```

Only full resolution RSLC reads are supported. The reader selects one frequency
at a time. HDF5 digital numbers, interpolated geolocation, LUTs and calibration
outputs remain lazy. `valid_mask` indicates invalid samples; calibration output
is not automatically masked. `sigma0` and `gamma0` subtract noise without clipping
negative values; `sigma0_raw` and `gamma0_raw` retain the calibrated signal.

The supplied calibration convention is `sigma0_raw = abs(DN)**2 / sigma0_lut**2`
and `nesz = noise_lut / sigma0_lut**2`. Synthetic tests check the implementation
of these formulas; real-product validation remains required for the supported
processing baselines. Annotation geolocation currently requires EPSG:4326.

Use `skip_variables=["land_mask"]` to avoid land-mask loading where the configured
land-mask source is unavailable. Other missions do not require `xarray-nisar`:
the optional reader import occurs when constructing `NisarMeta`.

Tests with small generated HDF5 products are in `test/test_nisar.py`. The real
product tests live in the companion reader and use `NISAR_TEST_RSLC`.

This branch was prepared from local commit `d1b1571` plus the preserved dirty
BIOMASS change. Current upstream `umr-lops/xsar/main` has not yet been fetched or
merged because network access was unavailable during preparation. See
`NISAR_PROVENANCE.json`; do not treat this branch as synchronized with upstream.
