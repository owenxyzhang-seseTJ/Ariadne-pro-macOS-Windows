"""Detector input invariants shared by the native proposal and overlap paths."""

import numpy as np
import pytest

from ariadne.spots import propose
from byte_cache import ByteLRU
from overlap_prepare import shared_pixel_noise


SPOT_OPTIONS = {
    "gain": 1.0, "offset": 0.0, "readout_variance": 0.1, "noise_model": "poisson_gaussian",
    "background_width": 7, "exclusion_width": 3, "scales": [3],
    "score_threshold": 3.0, "support_sigma": 1.0, "min_valid_fraction": 0.5,
    "background_iterations": 1,
}


@pytest.mark.parametrize("shape", [(1, 11), (11, 1), (11,), ()])
def test_broadcastable_detector_masks_are_rejected(shape):
    with pytest.raises(ValueError, match="matching mask"):
        propose(np.ones((11, 11)), np.ones(shape, dtype=bool), **SPOT_OPTIONS)


@pytest.mark.parametrize("name,value", [
    ("offset", np.inf), ("readout_variance", np.nan), ("score_threshold", np.nan),
    ("support_sigma", np.inf),
])
def test_nonfinite_detector_settings_are_rejected(name, value):
    image = np.ones((11, 11))
    with pytest.raises(ValueError, match="finite|threshold"):
        propose(image, image.astype(bool), **{**SPOT_OPTIONS, name: value})


def test_pixel_noise_matches_effective_defaults_in_either_order():
    implicit = {"gain": 2, "readout_variance": 1}
    explicit = {"gain": 2.0, "readout_variance": 1.0, "variance_floor": 2.0}
    assert shared_pixel_noise([implicit, explicit]) == shared_pixel_noise([explicit, implicit]) == (2., 1., 2.)
    different = {**explicit, "variance_floor": 3.0}
    for noises in ([implicit, different], [different, implicit]):
        with pytest.raises(ValueError, match="must share"):
            shared_pixel_noise(noises)
    with pytest.raises(ValueError, match="variance floor"):
        shared_pixel_noise([{**explicit, "variance_floor": np.nan}])


def test_frame_cache_respects_bytes_and_does_not_retain_oversize_frames():
    cache = ByteLRU(10, 10)
    first = cache.get("a", lambda: bytearray(4), len)
    cache.get("b", lambda: bytearray(4), len)
    assert cache.get("a", lambda: None, len) is first
    cache.get("c", lambda: bytearray(6), len)
    assert cache.statistics() == {"hits": 1, "loads": 3, "evictions": 1, "oversize_loads": 0,
                                  "current_bytes": 10, "peak_cached_bytes": 10, "max_bytes": 10}
    for _ in range(2):
        assert len(cache.get("large", lambda: bytearray(11), len)) == 11
    assert cache.statistics()["oversize_loads"] == 2
    assert cache.get("a", lambda: None, len) is first
