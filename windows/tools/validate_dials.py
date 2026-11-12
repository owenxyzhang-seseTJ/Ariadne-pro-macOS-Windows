"""核验 Windows DIALS 的 Ariadne 插件导入与真实 flex 阈值接口。"""
import os
import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
dll_handle = os.add_dll_directory(str(Path(sys.prefix) / 'Library/bin'))
for path in reversed([root/'source/python/src', root/'source/native/src', root/'dials-overlay/src', root/'overlay']):
    sys.path.insert(0, str(path))
import json
from types import SimpleNamespace
import numpy as np
import dials, dxtbx, cctbx
import dials.algorithms.indexing.model_evaluation as evaluation
import dials.algorithms.indexing.lattice_search as lattice
from dials.extensions.ariadne_spotfinder_threshold_ext import AriadneThresholdExt
from scitbx.array_family import flex
from ariadne.spots import propose

rng = np.random.default_rng(809)
y, x = np.indices((512, 512))
mean = np.full((512, 512), 12.0)
for cy, cx in [(90,100),(250,280),(390,420)]:
    mean += 350 * np.exp(-((x-cx)**2+(y-cy)**2)/4.0)
mask = np.ones(mean.shape, dtype=bool)
mask[:8] = False
rows=[]
for noise in ['poisson', 'poisson_gaussian']:
    image = rng.poisson(mean).astype(float)
    if noise == 'poisson_gaussian':
        image += rng.normal(0, 2, image.shape)
    p = AriadneThresholdExt.phil().extract()
    p.gain_adu_per_photon=1.0
    p.noise_model=noise
    p.readout_variance_adu=4.0 if noise=='poisson_gaussian' else 0.0
    params=SimpleNamespace(spotfinder=SimpleNamespace(threshold=SimpleNamespace(ariadne=p)))
    actual=AriadneThresholdExt(params).compute_threshold(flex.double(image), flex.bool(mask)).as_numpy_array()
    expected=propose(image,mask,gain=1.0,offset=p.offset_adu,readout_variance=p.readout_variance_adu,noise_model=noise,background_width=p.background_width,exclusion_width=p.exclusion_width,scales=p.scales,score_threshold=p.score_threshold,support_sigma=p.support_sigma,min_valid_fraction=p.min_valid_fraction,background_iterations=p.background_iterations)
    assert np.array_equal(actual, expected)
    assert actual.sum()>0 and not actual[~mask].any()
    rows.append({'noise_model':noise,'selected_pixels':int(actual.sum()),'equals_direct_api':True,'mask_respected':True})
versions={}
for p in (Path(sys.prefix)/'conda-meta').glob('*.json'):
    item=json.loads(p.read_text(encoding='utf-8'))
    if item['name'] in {'dials','dxtbx','cctbx-base','python'}:
        versions[item['name']]={'version':item['version'],'build':item['build']}
result={'python':sys.version,'executable':sys.executable,'versions':versions,'module_paths':{m.__name__:m.__file__ for m in [dials,dxtbx,cctbx,evaluation,lattice]},'threshold':rows,'acceptance':True,'raw_detector_end_to_end_tested':False}
(root/'results/dials-validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
