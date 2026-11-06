"""Synthetic RSLC product for NISAR integration tests."""

import h5py
import numpy as np


def make_product(path, frequencies=("A", "B"), dn=3 + 4j):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = np.array([-1.0, 0.0, 3.0, 8.0])
    samples = np.array([-1.0, 0.0, 5.0, 10.0])
    shape = (len(lines), len(samples))
    epoch = "seconds since 2026-10-09T00:00:00"

    def grid(group):
        group["zeroDopplerTime"] = 10 + lines * 0.5
        group["zeroDopplerTime"].attrs["units"] = epoch
        group["slantRange"] = 800000 + samples * 10

    with h5py.File(path, "w") as f:
        root = f.create_group("science/LSAR")
        ident = root.create_group("identification")
        attrs = {
            "productType": "RSLC", "orbitPassDirection": "Descending",
            "lookDirection": "Left", "zeroDopplerStartTime": "2026-10-09T00:00:10",
            "zeroDopplerEndTime": "2026-10-09T00:00:14",
            "boundingPolygon": "POLYGON ((19 9, 21 9, 21 11, 19 11, 19 9))",
        }
        for key, value in attrs.items():
            ident[key] = np.bytes_(value)
        ident["listOfFrequencies"] = np.array(frequencies, dtype="S1")
        rslc = root.create_group("RSLC")
        swaths = rslc.create_group("swaths")
        swaths["zeroDopplerTime"] = 10 + np.arange(8) * 0.5
        swaths["zeroDopplerTime"].attrs["units"] = epoch
        swaths["zeroDopplerTimeSpacing"] = 0.5
        meta = rslc.create_group("metadata")
        geo = meta.create_group("geolocationGrid")
        grid(geo)
        geo["heightAboveEllipsoid"] = [0.0]
        geo["epsg"] = 4326
        values = {
            "coordinateX": np.broadcast_to(20 + samples[None, :] * 0.01, shape),
            "coordinateY": np.broadcast_to(10 + lines[:, None] * 0.01, shape),
            "incidenceAngle": np.full(shape, 40.0),
            "elevationAngle": np.full(shape, 35.0),
            "groundTrackVelocity": np.full(shape, 7000.0),
            "alongTrackUnitVectorX": np.full(shape, -0.17),
            "alongTrackUnitVectorY": np.full(shape, -0.98),
            "losUnitVectorX": np.full(shape, 0.98),
            "losUnitVectorY": np.full(shape, -0.17),
        }
        for name, value in values.items():
            geo[name] = value[None, ...]
        orbit = meta.create_group("orbit")
        orbit["time"] = [0.0, 20.0]
        orbit["time"].attrs["units"] = epoch
        orbit["position"] = [[7000000., 0., 0.], [7000000., 140000., 0.]]
        orbit["velocity"] = [[0., 7000., 0.], [0., 7000., 0.]]
        orbit["orbitType"] = np.bytes_("POE")
        orbit["interpMethod"] = np.bytes_("Hermite")
        cal = meta.create_group("calibrationInformation")
        geometry = cal.create_group("geometry")
        grid(geometry)
        for name, value in [("sigma0", 2.), ("beta0", 1.), ("gamma0", 3.)]:
            geometry[name] = np.full(shape, value)
        for frequency in frequencies:
            sw = swaths.create_group(f"frequency{frequency}")
            sw["listOfPolarizations"] = np.array([b"VV", b"VH"])
            for pol in ["VV", "VH"]:
                sw[pol] = np.full((8, 10), dn, dtype=np.complex64)
            sw["slantRange"] = 800000 + np.arange(10) * 10
            for key, value in {
                "slantRangeSpacing": 10., "sceneCenterAlongTrackSpacing": 5.,
                "sceneCenterGroundRangeSpacing": 15., "processedCenterFrequency": 1.2e9,
                "processedRangeBandwidth": 5e6, "processedAzimuthBandwidth": 1000.,
                "nominalAcquisitionPRF": 2000., "numberOfSubSwaths": 1,
            }.items():
                sw[key] = value
            bounds = np.tile([0, 9], (8, 1))
            bounds[-1] = 0
            sw["validSamplesSubSwath1"] = bounds
            mask = np.zeros((8, 10), dtype=np.uint8)
            mask[1, 2] = 1
            sw["inputDataExceptionMask"] = mask
            fc = cal.create_group(f"frequency{frequency}")
            for pol in ["VV", "VH"]:
                fc.create_group(pol)["scaleFactor"] = 1.
            noise = fc.create_group("noiseEquivalentBackscatter")
            grid(noise)
            for pol in ["VV", "VH"]:
                noise[pol] = np.full(shape, 0.5)
            doppler = meta.create_group(f"processingInformation/parameters/frequency{frequency}")
            grid(doppler)
            doppler["dopplerCentroid"] = np.zeros(shape)
    return path
