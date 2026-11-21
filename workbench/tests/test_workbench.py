"""Behavioral checks for the real local API and scientific adapters."""
import json
import os
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient
from ariadne_workbench import app as server


@pytest.fixture
def client(tmp_path,monkeypatch):
    root=tmp_path/'projects';root.mkdir();monkeypatch.setattr(server,'PROJECTS',root)
    with TestClient(server.app) as client:yield client


def finished(client,project_id):
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        result=client.get(f'/api/projects/{project_id}').json()
        job=result['jobs'][0]
        if job['status'] not in {'running','queued'}:return result,job
        time.sleep(.2)
    raise AssertionError('Scientific adapter did not finish in the test budget')


def test_catalog_exposes_real_boundaries(client):
    response=client.get('/api/capabilities');assert response.status_code==200
    data=response.json();ids={t['id'] for t in data['tools']}
    assert {'refine','autosolve','maps','integrate-pixels','dials:index','analysis:joint_fit','legacy:shelxl'}<=ids
    assert data['native']['native_full_shelx_parity'] is False


def test_import_blocks_traversal_and_cross_origin(client):
    result=client.post('/api/projects',files={'files':('../../outside.cif',b'data_x')})
    assert result.status_code==400
    result=client.post('/api/demo',json={'kind':'fen4'},headers={'Origin':'https://external.example'})
    assert result.status_code==403


def test_structure_reflections_and_real_refinement(client):
    project=client.post('/api/demo',json={'kind':'fen4'}).json();project_id=project['id']
    view=client.get(f'/api/projects/{project_id}/view/structure').json()
    assert view['unitCellAtoms']==5
    assert view['composition']=={'Fe':1,'N':4}
    assert all(atom.get('ellipsoid',{}).get('probability')==.5 for atom in view['atoms'])
    reflections=client.get(f'/api/projects/{project_id}/view/reflections').json()
    assert reflections['total']>10 and len(reflections['reflections'])==reflections['shown']
    result=client.post(f'/api/projects/{project_id}/jobs',json={'action':'refine','settings':{'options':{'max_nfev':50}}})
    assert result.status_code==202,result.text
    project,job=finished(client,project_id)
    assert job['status']=='completed',job.get('logTail')
    assert project['structure'].startswith('jobs/') and project['structure'].endswith('.json')
    assert project.get('metrics')
    files=project['artifacts'];assert any(f['name']=='structure.cif' for f in files)
    result_file=next(f['path'] for f in files if f['name']=='result.json')
    result=client.get(f'/api/projects/{project_id}/files/{result_file}')
    assert result.status_code==200 and 'metrics' in result.json()
    assert client.post(f'/api/projects/{project_id}/jobs/{job["id"]}',json={'action':'cancel'}).status_code==400


def test_calibrated_pixel_job_and_input_validation(client):
    project=client.post('/api/demo',json={'kind':'pixels'}).json();pid=project['id']
    profile=client.get(f'/api/projects/{pid}/view/profile').json()
    assert len(profile['x'])==len(profile['observed'])==2
    assert client.patch(f'/api/projects/{pid}/settings',json={'selection':[1,-1]}).status_code==400
    assert client.patch(f'/api/projects/{pid}/settings',json={'cell':{'a':-1,'b':2,'c':3}}).status_code==400
    assert client.post(f'/api/projects/{pid}/jobs',json={'action':'integrate-pixels'}).status_code==202
    _,job=finished(client,pid)
    assert job['status']=='completed',job.get('logTail')


def test_screened_phase_search_serializes_persistent_candidates(client):
    project = client.post('/api/demo', json={'kind':'fen4'}).json()
    project_id = project['id']
    uri = f'/api/projects/{project_id}'
    response = client.post(uri+'/jobs',json={
        'action':'solve',
        'settings': {
            'options': {'trials':4, 'iterations':80},
            'screening': {'coarse_backend':'batched_cpu','coarse_iterations':20,'promoted':2},
        },
    })
    assert response.status_code == 202, response.text
    _, job = finished(client, project_id)
    assert job['status'] == 'completed', job.get('message')
    result_file = next(path for path in job['steps'][0]['artifacts'] if path.endswith('/result.json'))
    result = client.get(uri+'/files/'+result_file).json()
    assert result['screening_runtime']['backend'] == 'batched_cpu'
    assert 1 <= len(result['candidates']) <= 2
    assert result['candidates'][0]['structure']['atoms']


def test_workbench_waits_for_a_shared_host_reservation(client, tmp_path, monkeypatch):
    from ariadne.agent.host_budget import HostBudget

    config = tmp_path/'host.json'
    config.write_text('{"cpu_threads":1,"memory_mb":512}', encoding='utf-8')
    monkeypatch.setenv('ARIADNE_HOST_BUDGET', str(config))
    project_id = client.post('/api/demo', json={'kind':'pixels'}).json()['id']
    with HostBudget(config).request({'cpu_threads':1,'memory_mb':512}) as barrier:
        assert barrier.try_admit()
        result = client.post(f'/api/projects/{project_id}/jobs', json={'action':'integrate-pixels'})
        assert result.status_code == 202, result.text
        deadline = time.monotonic()+12
        while time.monotonic() < deadline:
            job = client.get(f'/api/projects/{project_id}').json()['jobs'][0]
            if job['steps'][0].get('resources'):
                break
            time.sleep(.05)
        assert job['steps'][0]['resources'] == {'cpu_threads':1,'memory_mb':512}
        assert job['status'] == 'queued'
        job_id = job['id']
        first_run = job['runId']
        assert client.post(f'/api/projects/{project_id}/jobs/{job_id}',json={'action':'cancel'}).status_code == 200
        _, stopped = finished(client,project_id)
        assert stopped['status'] == 'cancelled'
        assert stopped['steps'][0]['attempt'] == 0
        resumed=client.post(f'/api/projects/{project_id}/jobs/{job_id}',json={'action':'resume'})
        assert resumed.status_code == 200,resumed.text
        assert resumed.json()['id'] == job_id and resumed.json()['runId'] != first_run
    _, job = finished(client, project_id)
    assert job['status'] == 'completed', job.get('logTail')
    assert job['steps'][0]['attempt'] == 1
    response=client.get(f'/api/projects/{project_id}/jobs/{job_id}/events?after=0&limit=200')
    assert response.status_code == 200,response.text
    events=response.json()['events']
    assert sum(item['kind']=='attempt_started' for item in events) == 1
    assert any(item['kind']=='cancel_requested' for item in events)


def test_dials_point_table_keeps_ids_and_negative_intensities(client):
    from dials.array_family import flex
    import tempfile
    table=flex.reflection_table()
    table['miller_index']=flex.miller_index([(1,0,0),(0,1,0)])
    table['xyzobs.px.value']=flex.vec3_double([(100,110,.5),(200,210,1.5)])
    table['intensity.sum.value']=flex.double([-2.,9.])
    table['intensity.sum.variance']=flex.double([1.,4.])
    table['intensity.prf.value']=flex.double([0.,10.])
    table['intensity.prf.variance']=flex.double([0.,4.])
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'synthetic-api-fixture.refl';table.as_file(str(path))
        response=client.post('/api/projects',files={'files':(path.name,path.read_bytes())})
    assert response.status_code==201,response.text
    pid=response.json()['id']
    result=client.get(f'/api/projects/{pid}/view/reflections').json()
    assert result['negativeCount']==1
    assert result['reflections'][0]['intensity']==-2
    assert result['reflections'][1]['intensity']==10
    assert result['reflections'][1]['frame']==1
    assert result['reflections'][1]['x']==200
    # The richer point table must remain the visual source after HKL export.
    extra={'hkl':[[9,9,9]],'intensity':[42.],'sigma':[1.]}
    response=client.post(f'/api/projects/{pid}/import',files={'files':('exported.json',json.dumps(extra).encode())})
    assert response.status_code==201
    result=client.get(f'/api/projects/{pid}/view/reflections').json()
    assert result['total']==2 and result['reflections'][1]['x']==200


def test_unindexed_points_keep_measured_reciprocal_vectors(client,tmp_path):
    from dials.array_family import flex
    table=flex.reflection_table()
    table['xyzobs.px.value']=flex.vec3_double([(100,110,.5),(200,210,1.5)])
    table['rlp']=flex.vec3_double([(.1,.2,.3),(-.2,.3,.4)])
    table['id']=flex.int([0,0])
    table['panel']=flex.size_t([0,0])
    path=tmp_path/'unindexed.refl';table.as_file(str(path))
    project=client.post('/api/projects',files={'files':(path.name,path.read_bytes())}).json()
    result=client.get(f'/api/projects/{project["id"]}/view/reflections').json()
    assert result['reciprocalUnit']=='Å⁻¹'
    assert result['reflections'][0]['q']==pytest.approx([.1,.2,.3])
    assert result['reflections'][0]['d']==pytest.approx(1/(.14**.5))
    assert result['reflections'][1]['indexed'] is False
    assert 'domain' not in result['reflections'][0]


def test_pixel_profile_follows_selected_reflection_frame(monkeypatch,tmp_path):
    import numpy as np
    from ariadne_workbench import science
    seen=[]
    def pixels(project,folder,index):
        seen.append(index)
        return np.full((12,30),index+1.)
    monkeypatch.setattr(science,'frame_array',pixels)
    profile=science.pixel_profile({'frames':['a','b','c']},tmp_path,{'x':15,'y':6,'frame':2})
    assert seen==[2]
    assert profile['frame']==2 and profile['observed']==[15.]*21


def test_resolution_rings_and_probe_follow_detector_geometry():
    import math
    from dxtbx.model import BeamFactory,DetectorFactory
    from ariadne_workbench.science import detector_geometry
    beam=BeamFactory.simple(1.)
    panel=DetectorFactory.simple('PAD',100.,(50.,50.),'+x','-y',(.1,.1),(1000,1000),(0.,1.e6))[0]
    probe=detector_geometry(panel,beam,(600.,500.))
    expected=1/(2*math.sin(math.atan(.1)/2))
    assert probe['d']==pytest.approx(expected,rel=1.e-9)
    geometry=detector_geometry(panel,beam)
    assert geometry['beamCenter']==pytest.approx([500.,500.])
    assert len(geometry['rings'])>=3
    for ring in geometry['rings']:
        for segment in ring['segments']:
            for point in segment[::90]:
                assert panel.get_resolution_at_pixel(beam.get_s0(),point)==pytest.approx(ring['d'],rel=1.e-8)


def test_density_maps_contours_and_uploaded_ccp4(client,tmp_path):
    import numpy as np
    from ariadne_xtal.maps import DensityMap
    from ariadne_xtal.model import Cell
    from ariadne_workbench.science import run_native
    project=client.post('/api/demo',json={'kind':'fen4'}).json();pid=project['id']
    folder=server.PROJECTS/pid;out=folder/'maps';out.mkdir()
    run_native('maps',project,folder,out,{})
    project['surface']='maps/surface.json';server.dump(folder/'project.json',project)
    result=client.get(f'/api/projects/{pid}/view/density?map=difference&sigma=3').json()
    assert {item['id'] for item in result['availableMaps']}=={'difference','calculated','patterson'}
    stats=result['statistics']
    expected={sign for sign in (1,-1) if stats['min']<stats['mean']+sign*3*stats['sigma']<stats['max']}
    assert {surface['sign'] for surface in result['surfaces']}==expected
    for kind in ('calculated','patterson'):
        result=client.get(f'/api/projects/{pid}/view/density?map={kind}&sigma=1.5').json()
        assert result['surfaces'] and all(s['sign']==1 for s in result['surfaces'])
    assert client.get(f'/api/projects/{pid}/view/density?sigma=0').status_code==400
    assert client.get(f'/api/projects/{pid}/view/density?map=../../private').status_code==400
    grid=np.indices((24,24,24)).astype(float)/24
    values=np.exp(-sum((a-.3)**2 for a in grid)/.015)-np.exp(-sum((a-.7)**2 for a in grid)/.015)
    path=tmp_path/'display.ccp4';DensityMap(values,Cell(10,11,12,90,100,90)).save_ccp4(path)
    imported=client.post('/api/projects',files={'files':(path.name,path.read_bytes())}).json()
    assert imported['files'][0]['role']=='density'
    result=client.get(f'/api/projects/{imported["id"]}/view/density?sigma=1.5').json()
    assert result['surfaces'] and result['cell']['beta']==100
    assert {surface['sign'] for surface in result['surfaces']}=={-1,1}
    assert result['statistics']['grid']==[24,24,24]
    low=client.get(f'/api/projects/{imported["id"]}/view/density?sigma=1').json()
    high=client.get(f'/api/projects/{imported["id"]}/view/density?sigma=4').json()
    assert sum(len(s['vertices']) for s in low['surfaces'])>sum(len(s['vertices']) for s in high['surfaces'])


SHELX_DISPLAY_FIXTURE='''TITL read only SHELX input
CELL .71073 8 9 10 90 90 90
ZERR 1 .001 .001 .002 0 0 0
LATT -1
SFAC C
UNIT 1
L.S. 10
ANIS
ACTA
FMAP 2
PLAN 20
C1 1 .1 .2 .3 11 .02 9.5
HKLF 4
END
'''

FCF_FIXTURE='''data_map
_cell_length_a 8
_cell_length_b 9
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_refln_index_h
_refln_index_k
_refln_index_l
_refln_F_squared_meas
_refln_F_squared_sigma
_refln_F_calc
_refln_phase_calc
1 0 0 9 1 1 0
0 1 0 -4 1 0 0
'''


def test_shelxl_map_coefficients_normalization_and_source_preservation(tmp_path):
    from ariadne_workbench.shelx_engine import input_plan
    from ariadne_workbench.shelx_maps import prepare_map_input,maps_from_fcf
    from ariadne_xtal.shelx import ShelxDocument
    from ariadne_xtal.maps import load_map
    import shutil
    source=tmp_path/'model.ins';source.write_bytes(SHELX_DISPLAY_FIXTURE.replace('\n','\r\n').encode())
    (tmp_path/'data.hkl').write_bytes(b'unchanged HKL bytes\r\n');out=tmp_path/'run';out.mkdir()
    plan=input_plan({'structure':'model.ins','reflections':'data.hkl'},tmp_path,{},'shelxl')
    for target,original in plan['files']:shutil.copyfile(original,out/target)
    recipe=prepare_map_input(plan,out)
    assert source.read_bytes()==(out/'source_instructions/model.ins').read_bytes()
    active=[(x.keyword,x.arguments) for x in ShelxDocument.from_path(out/'model.ins').instructions if x.active]
    assert ('L.S.',('0',)) in active and ('LIST',('6',)) in active
    assert not any(key=='ACTA' for key,args in active)
    assert any(change['original']=='ACTA' for change in recipe['changes'])
    assert ('ANIS',()) in active and ('ZERR',('1','.001','.001','.002','0','0','0')) in active
    assert recipe['changes'][0]['original']=='L.S. 10'
    fcf=out/'model.fcf';fcf.write_text(FCF_FIXTURE);result=maps_from_fcf(fcf,out)
    assert result['reflections']==2
    difference=load_map(out/'difference.npz').values;calculated=load_map(out/'calculated.npz').values;patterson=load_map(out/'patterson.npz').values
    # A P1 +/-h pair contributes 2F/V. Negative measured F² is retained in Patterson.
    assert difference[0,0,0]==pytest.approx(4/720,rel=1.e-6)
    assert difference[difference.shape[0]//2,0,0]==pytest.approx(-4/720,rel=1.e-6)
    assert calculated[0,0,0]==pytest.approx(2/720,rel=1.e-6)
    assert patterson[0,0,0]==pytest.approx(10/720,rel=1.e-6)
    fcf.write_text(FCF_FIXTURE.replace('1 0 0 9 1 1 0','1 0 0 9 1 1 90'))
    maps_from_fcf(fcf,out);shifted=load_map(out/'difference.npz').values
    assert shifted[shifted.shape[0]//4,0,0]==pytest.approx(4/720,rel=1.e-6)


def test_npd_shelx_is_displayed_without_changing_invalid_tensor(client):
    import numpy as np
    npd=SHELX_DISPLAY_FIXTURE.replace('C1 1 .1 .2 .3 11 .02 9.5','C1 1 .1 .2 .3 11 .02 .02 -.01 0 0 0')
    project=client.post('/api/projects',files={'files':('npd.res',npd.encode())}).json()
    assert project['structureCompatibility']['npdAtoms']==['C1']
    view=client.get(f'/api/projects/{project["id"]}/view/structure').json()
    atom=view['atoms'][0]
    assert atom['adpValid'] is False and 'ellipsoid' not in atom
    assert np.linalg.eigvalsh(atom['uCart']).min()==pytest.approx(-.01)
    assert (server.PROJECTS/project['id']/project['structure']).read_text()==npd


def test_standard_shelx_is_viewable_but_native_refinement_stays_strict(client):
    from ariadne_xtal.shelx import ShelxDocument,NativeConversionError
    project=client.post('/api/projects',files={'files':('model.ins',SHELX_DISPLAY_FIXTURE.encode())}).json()
    pid=project['id'];assert project['structureCompatibility']['displayOnly'] is True
    assert not project.get('elements')
    view=client.get(f'/api/projects/{pid}/view/structure').json()
    assert view['asymmetricAtoms']==1 and view['spacegroup']=='P 1'
    assert view['cell']['a']==8
    assert view['atoms'][0]['fractional']==pytest.approx([.1,.2,.3])
    refs={'hkl':[[1,0,0]],'intensity':[20.],'sigma':[1.]}
    client.post(f'/api/projects/{pid}/import',files={'files':('measured.json',json.dumps(refs).encode())})
    result=client.post(f'/api/projects/{pid}/jobs',json={'action':'refine','settings':{'engine':'native'}})
    assert result.status_code==400 and 'SHELXL' in result.json()['error']
    assert client.get(f'/api/projects/{pid}').json()['jobs']==[]
    source=server.PROJECTS/pid/project['structure']
    assert source.read_text()==SHELX_DISPLAY_FIXTURE
    with pytest.raises(NativeConversionError):ShelxDocument.from_path(source).to_structure()


def test_shelx_engine_discovery_respects_explicit_override(tmp_path,monkeypatch):
    from ariadne_workbench.shelx_engine import engine_info
    executable=tmp_path/('shelxl.exe' if os.name=='nt' else 'shelxl')
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o755)
    monkeypatch.setenv('PATH',str(tmp_path));monkeypatch.delenv('ARIADNE_SHELXL_PATH',raising=False)
    assert engine_info('shelxl')['executable']==str(executable)
    monkeypatch.setenv('ARIADNE_SHELXL_PATH',str(tmp_path/'missing'))
    assert engine_info('shelxl')['available'] is False
    if os.name != 'nt':
        executable.chmod(0o644)
        monkeypatch.setenv('ARIADNE_SHELXL_PATH',str(executable))
        assert engine_info('shelxl')['available'] is False


def test_shelx_input_pair_keeps_includes_and_fab_inside_source(tmp_path):
    from ariadne_workbench.shelx_engine import input_plan
    source=tmp_path/'instructions';source.mkdir();(source/'parts').mkdir();data=tmp_path/'data';data.mkdir()
    original=source/'sample.res';original.write_text(SHELX_DISPLAY_FIXTURE.replace('HKLF 4','+parts/extra.ins\nABIN\nHKLF 4'))
    extra=source/'parts/extra.ins';extra.write_text('REM retained include\n')
    fab=source/'sample.fab';fab.write_bytes(b'partial structure factors\r\n')
    hkl=data/'measured.hkl';hkl.write_bytes(b'   1   0   0   20.00    1.00\r\n')
    project={'structure':'instructions/sample.res','reflections':'data/measured.hkl'}
    plan=input_plan(project,tmp_path,{},'shelxl')
    assert dict(plan['files'])=={Path('sample.ins'):original,Path('parts/extra.ins'):extra,Path('sample.hkl'):hkl,Path('sample.fab'):fab}
    fab.unlink()
    with pytest.raises(ValueError,match='ABIN'):input_plan(project,tmp_path,{},'shelxl')
    original.write_text(SHELX_DISPLAY_FIXTURE.replace('HKLF 4','+../outside.ins\nHKLF 4'))
    with pytest.raises(ValueError,match='附件路径'):input_plan(project,tmp_path,{},'shelxl')
    original.write_text(SHELX_DISPLAY_FIXTURE.replace('HKLF 4','+parts/extra.ins\nHKLF 4'));extra.write_text('+parts/extra.ins\n')
    with pytest.raises(ValueError,match='循环'):input_plan(project,tmp_path,{},'shelxl')


@pytest.mark.skipif(os.name == 'nt', reason='POSIX SHELX executable fixture')
@pytest.mark.parametrize('action',['refine','automatic','legacy:shelxl'])
def test_shelx_jobs_run_explicit_pair_without_native_conversion(client,tmp_path,monkeypatch,action):
    import sys
    # A subprocess test double verifies transport and routing, not numerical refinement.
    executable=tmp_path/'shelxl'
    executable.write_text(f'#!{sys.executable}\nfrom pathlib import Path\nimport sys\np=Path(sys.argv[1])\nassert p.with_suffix(".hkl").read_bytes()==b"measured HKL bytes\\r\\n"\np.with_suffix(".res").write_bytes(p.with_suffix(".ins").read_bytes())\np.with_suffix(".lst").write_text("SHELXL-2018/3 test double")\nif b"LIST 6" in p.with_suffix(".ins").read_bytes():p.with_suffix(".fcf").write_text({FCF_FIXTURE!r})\n')
    executable.chmod(0o755);monkeypatch.setenv('ARIADNE_SHELXL_PATH',str(executable))
    response=client.post('/api/projects',files=[('files',('ins/source.res',SHELX_DISPLAY_FIXTURE.encode())),('files',('data/observations.hkl',b'measured HKL bytes\r\n'))])
    assert response.status_code==201,response.text
    project=response.json();pid=project['id'];folder=server.PROJECTS/pid;original=project['structure'];reflections=project['reflections']
    response=client.post(f'/api/projects/{pid}/jobs',json={'action':action})
    assert response.status_code==202,response.text
    accepted=response.json()
    assert [s['action'] for s in accepted['steps']]==(['legacy:shelxl','legacy:shelxl-maps'] if action=='automatic' else ['legacy:shelxl'])
    assert accepted['settings']['shelx_structure']==original
    assert accepted['settings']['shelx_reflections']==reflections
    project,job=finished(client,pid)
    assert job['status']=='completed',job.get('logTail')
    assert project['structure'].endswith('/source.res') and project['structure'].startswith('jobs/')
    assert project['structureCompatibility']['nativeCompatible'] is False
    assert (folder/original).read_text()==SHELX_DISPLAY_FIXTURE
    assert (folder/reflections).read_bytes()==b'measured HKL bytes\r\n'
    result=json.loads((folder/project['lastRefinement']['result']).read_text())
    assert result['scientific_acceptance']=='not_assessed'
    assert result['inputs'][1]['source']==reflections
    assert client.get(f'/api/projects/{pid}/view/structure').status_code==200
    if action=='automatic':
        assert project['surface'].endswith('/surface.json')
        density=client.get(f'/api/projects/{pid}/view/density').json()
        assert {m['id'] for m in density['availableMaps']}=={'difference','calculated','patterson'}
        map_result=json.loads((folder/project['surface']).with_name('result.json').read_text())
        assert map_result['inputs'][0]['source']==project['structure']


@pytest.mark.skipif(os.name == 'nt', reason='POSIX SHELX executable fixture')
def test_shelx_zero_exit_without_result_does_not_complete_job(client,tmp_path,monkeypatch):
    executable=tmp_path/'shelxl';executable.write_text('#!/bin/sh\nexit 0\n');executable.chmod(0o755)
    monkeypatch.setenv('ARIADNE_SHELXL_PATH',str(executable))
    response=client.post('/api/projects',files=[('files',('source.ins',SHELX_DISPLAY_FIXTURE.encode())),('files',('measured.hkl',b'input HKL'))])
    project=response.json();pid=project['id'];original=project['structure']
    assert client.post(f'/api/projects/{pid}/jobs',json={'action':'legacy:shelxl'}).status_code==202
    project,job=finished(client,pid)
    assert job['status']=='failed' and 'RES' in job['message']
    assert project['structure']==original


@pytest.mark.skipif(os.name == 'nt', reason='POSIX SHELX executable fixture')
def test_dag_resumes_failed_map_in_new_attempt_and_keeps_refinement(client,tmp_path,monkeypatch):
    import sys
    from ariadne.agent import engine

    executable=tmp_path/'shelxl'
    gate=tmp_path/'allow_maps'
    executable.write_text(f'#!{sys.executable}\nfrom pathlib import Path\nimport sys\np=Path(sys.argv[1]); data=p.with_suffix(".ins").read_bytes()\np.with_suffix(".res").write_bytes(data)\nif b"LIST 6" in data and Path({str(gate)!r}).exists(): p.with_suffix(".fcf").write_text({FCF_FIXTURE!r})\n')
    executable.chmod(0o755);monkeypatch.setenv('ARIADNE_SHELXL_PATH',str(executable))
    project=client.post('/api/projects',files=[('files',('source.ins',SHELX_DISPLAY_FIXTURE.encode())),
                                             ('files',('measured.hkl',b'measured HKL'))]).json()
    pid=project['id']
    response=client.post(f'/api/projects/{pid}/jobs',json={'action':'automatic'})
    assert response.status_code==202,response.text
    project,failed=finished(client,pid)
    assert failed['status']=='failed' and [s['status'] for s in failed['steps']]==['completed','failed']
    fitted=project['structure']
    jobfolder=server.PROJECTS/pid/'jobs'/failed['id']
    with pytest.raises(ValueError,match='application executor'):
        engine.run(jobfolder/'session')
    gate.touch()
    reply=client.post(f"/api/projects/{pid}/jobs/{failed['id']}",json={'action':'resume'})
    assert reply.status_code==200,reply.text
    project,resumed=finished(client,pid)
    assert resumed['id']==failed['id'] and resumed['runId']!=failed['runId']
    assert resumed['status']=='completed',resumed.get('message')
    assert [s['attempt'] for s in resumed['steps']]==[1,2]
    assert project['structure']==fitted and project['surface'].endswith('/surface.json')
    assert (jobfolder/'session/attempts/step-01/001/result.json').is_file()
    assert (jobfolder/'session/attempts/step-01/002/result.json').is_file()


def test_detached_dag_survives_workbench_submitter_exit(client,tmp_path,monkeypatch):
    import os
    import subprocess
    import sys
    from ariadne.agent.host_budget import HostBudget

    config=tmp_path/'host.json'
    config.write_text('{"cpu_threads":1,"memory_mb":512}',encoding='utf-8')
    monkeypatch.setenv('ARIADNE_HOST_BUDGET',str(config))
    project_id=client.post('/api/demo',json={'kind':'pixels'}).json()['id']
    program='''
import sys
from starlette.testclient import TestClient
from ariadne_workbench.app import app
with TestClient(app) as client:
    response=client.post(f"/api/projects/{sys.argv[1]}/jobs",json={"action":"integrate-pixels"})
    assert response.status_code==202,response.text
print(response.json()["runId"])
'''
    with HostBudget(config).request({'cpu_threads':1,'memory_mb':512}) as barrier:
        assert barrier.try_admit()
        first=subprocess.run([sys.executable,'-c',program,project_id],
                             env={**os.environ,'ARIADNE_PROJECTS_DIR':str(server.PROJECTS)},
                             capture_output=True,text=True,check=True,timeout=20)
        submitted=first.stdout.strip()
        job=client.get(f'/api/projects/{project_id}').json()['jobs'][0]
        assert job['runId']==submitted and job['status']=='queued'
    _,job=finished(client,project_id)
    assert job['runId']==submitted and job['status']=='completed'
    assert job['steps'][0]['attempt']==1


@pytest.mark.skipif(os.name == 'nt', reason='POSIX external process fixture and forced supervisor death')
@pytest.mark.parametrize('interruption', ['cancel', 'crash'])
def test_workbench_recovers_unfinished_node_after_process_stops(client, tmp_path, monkeypatch, interruption):
    import signal
    import sys
    from ariadne.server import pid_alive

    executable = tmp_path/'shelxl'
    marker = tmp_path/'map-process-pid'
    ready = tmp_path/'map-second-attempt'
    executable.write_text(
        f'#!{sys.executable}\n'
        'from pathlib import Path\nimport os, sys, time\n'
        'job = Path(sys.argv[1]); data = job.with_suffix(".ins").read_bytes()\n'
        'job.with_suffix(".res").write_bytes(data)\n'
        'if b"LIST 6" in data:\n'
        f'    if not Path({str(ready)!r}).exists():\n'
        f'        Path({str(marker)!r}).write_text(str(os.getpid()))\n'
        '        time.sleep(120)\n'
        f'    job.with_suffix(".fcf").write_text({FCF_FIXTURE!r})\n')
    executable.chmod(0o755)
    monkeypatch.setenv('ARIADNE_SHELXL_PATH', str(executable))
    project = client.post('/api/projects', files=[
        ('files', ('source.ins', SHELX_DISPLAY_FIXTURE.encode())),
        ('files', ('measured.hkl', b'measured HKL')),
    ]).json()
    project_path = f"/api/projects/{project['id']}"
    response = client.post(project_path+'/jobs', json={'action': 'automatic'})
    assert response.status_code == 202, response.text
    started = response.json()
    job_path = project_path+f"/jobs/{started['id']}"
    deadline = time.monotonic() + 20
    while not marker.is_file() and time.monotonic() < deadline:
        time.sleep(.05)
    external_pid = int(marker.read_text())
    running = client.get(project_path).json()
    model = running['structure']
    assert running['jobs'][0]['steps'][0]['status'] == 'completed'
    head = client.get(job_path+'/events', params={'limit': 2}).json()
    assert head['more']
    if interruption == 'cancel':
        response = client.post(job_path, json={'action': 'cancel'})
        assert response.status_code == 200, response.text
    else:
        os.kill(started['pid'], signal.SIGKILL)
    _, stopped = finished(client, project['id'])
    assert stopped['status'] == ('cancelled' if interruption == 'cancel' else 'interrupted')
    deadline = time.monotonic() + 5
    while pid_alive(external_pid) and time.monotonic() < deadline:
        time.sleep(.05)
    assert not pid_alive(external_pid)
    ready.touch()
    # A fresh API instance can resume the same SQLite session after restart.
    with TestClient(server.app) as reopened:
        response = reopened.post(job_path, json={'action': 'resume'})
        assert response.status_code == 200, response.text
        saved, resumed = finished(reopened, project['id'])
        tail = reopened.get(job_path+'/events', params={'after': head['next_cursor']}).json()
    assert resumed['status'] == 'completed', resumed['message']
    assert resumed['id'] == started['id'] and resumed['runId'] != started['runId']
    assert [step['attempt'] for step in resumed['steps']] == [1, 2]
    assert saved['structure'] == model and saved['surface'].endswith('/surface.json')
    assert tail['events'][0]['seq'] == head['next_cursor'] + 1
    assert any(event['kind'] == 'retry_authorized' for event in tail['events'])


def test_pixel_job_ignores_unrelated_shelx_instructions(client):
    project=client.post('/api/demo',json={'kind':'pixels'}).json();pid=project['id']
    assert client.post(f'/api/projects/{pid}/import',files={'files':('model.res',SHELX_DISPLAY_FIXTURE.encode())}).status_code==201
    assert client.post(f'/api/projects/{pid}/jobs',json={'action':'automatic'}).status_code==202
    project,job=finished(client,pid)
    assert job['status']=='completed',job.get('logTail')
    assert project['structure'].endswith('model.res')


def test_dials_commit_does_not_convert_unrelated_shelx(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from ariadne_workbench import runner, workflow
    from ariadne_workbench.science import dump
    from ariadne.agent.store import Store
    from contextlib import closing

    folder=tmp_path/'project';folder.mkdir();source=folder/'model.ins';source.write_text(SHELX_DISPLAY_FIXTURE)
    project = {'id': 'p', 'structure': source.name, 'cell': None}
    dump(folder/'project.json', project)
    jobfolder = folder/'jobs/run'
    jobfolder.mkdir(parents=True)
    workflow.initialize(jobfolder, project, {'id':'run', 'action':'dials:find_spots','settings':{},
                                            'steps':[{'action':'dials:find_spots'}]})
    def completed_operation(action,project,folder,out,settings,log):
        assert action=='find_spots'
        result = out/'new-strong.refl'
        result.write_text('completed')
        return {'dials_reflections': result.relative_to(folder).as_posix()}
    monkeypatch.setattr(runner,'dials',completed_operation)
    with closing(Store(jobfolder/'session')) as store:
        out = store.begin('step-00')
        args = store.node('step-00')['args']
        args['project'] = store.get('inputs')['project']
        result = workflow.execute_step(SimpleNamespace(root=store.root, directory=out,
                                                       resources={'cpu_threads':1}),
                                       workflow.ACTION, args)
        store.finish('step-00', 'succeeded', .01, result=result)
    project['lastJob'] = 'run'
    project = workflow.advance_project(folder, project)
    assert project['dials_reflections'].endswith('/new-strong.refl') and project['cell'] is None
    assert source.read_text()==SHELX_DISPLAY_FIXTURE
