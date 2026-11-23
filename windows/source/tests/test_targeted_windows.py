import numpy as np

from targeted_windows import choose_windows, preview_information


def test_window_geometry_tracks_contrast_and_charges_frames_once():
    covariance = np.diag([1.4, 1.1, .4])**2
    base = {'entry': {'covariance_px_xyz': covariance}, 'centre': [15., 16., 4.], 'domain': 0}
    other = {**base, 'domain': 1}
    assert preview_information([base, other])[:2] == ('almost_coincident', 0.)
    split = {**other, 'centre': [17., 16., 4.]}
    kind, information, separation = preview_information([base, split])
    assert kind == 'partial_overlap'
    assert 0 < information < 1
    assert np.isclose(separation, 2/1.4)
    common = {'scan': 0, 'voxels': 700, 'domain': 1, 'reciprocal_sq': .5}
    windows = [dict(common, frames={0, 1}, classification=kind, information=information),
               dict(common, frames={1}, classification='isolated', information=0.),
               dict(common, frames={9, 10}, classification='almost_coincident', information=0.)]
    selected, details = choose_windows(windows, 2, random_fraction=0.)
    assert selected == {0, 1}
    assert details['planned_distinct_frames'] == 2
