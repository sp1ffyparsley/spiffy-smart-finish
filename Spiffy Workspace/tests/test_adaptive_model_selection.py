"""Local model auditions: real clocks/pixels, never mock results called Topaz."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import workflow.final_delivery_pipeline  # noqa: F401
import adaptive_enhancement as adaptive
import video_upscale as worker


def selection():
    import adaptive_model_selection
    return adaptive_model_selection


def plan(frames=240, cuts=(0,120,240)):
    shots=[{"start_frame":a,"end_frame":b,"route":"masked_full_frame",
            "motion_peak_frame":a+12,"motion_peak_delta":4,
            "measurements":[{"sharpness":.7,"blockiness":0.1,"texture_residual":3}],
            "effective_controls":{"details":.12,"blend":.5}}
           for a,b in zip(cuts,cuts[1:])]
    return {"schema":adaptive.SCHEMA,"source_fps":"24","frame_count":frames,
            "source_width":320,"source_height":180,"target_width":640,"target_height":360,
            "shots":shots,"route":"masked_full_frame","crop":[0,0,320,180],
            "effective_params":":details=0.12:blend=0.5", "effective_controls":{"details":.12,"blend":.5},
            "policy":{"mode":"natural"}}


def test_auditions_are_bounded_cover_each_admitted_shot_and_do_not_cross_cuts():
    value=plan()
    decision=selection().plan_audition(value)
    assert decision['eligible']
    assert len(decision['windows'])==2
    for window,shot in zip(decision['windows'],value['shots']):
        assert shot['start_frame']<=window[0]<window[1]<=shot['end_frame']
        assert window[1]-window[0]==8
    assert 2*sum(b-a for a,b in decision['windows'])<=value['frame_count']*.2


@pytest.mark.parametrize('change,reason',[
    ({'frame_count':24,'shots':plan(24,(0,24))['shots']},'pilot_cost_budget'),
    ({'shots':plan(240,(0,60,120,240))['shots']},'too_many_distinct_shots'),
    ({'route':'source_preserved'},'no_enhancement_needed'),
    ({'policy':{'mode':'natural','model_search':'off'}},'explicitly_disabled'),
])
def test_no_extra_model_startup_when_pilot_cannot_pay_or_scope_is_unqualified(change,reason):
    value=plan()
    value.update(change)
    result=selection().plan_audition(value)
    assert not result['eligible'] and result['reason']==reason


def test_four_second_real_shorts_do_not_pay_two_native_model_startups():
    # The old five-times-pilot-frame heuristic admitted 96-frame edits and
    # spent 15-20s launching two models despite <1s projected export savings.
    short=plan(96,(0,96))
    assert selection().plan_audition(short)=={
        'eligible':False,'reason':'pilot_cost_budget','windows':[]}
    longer=plan(240,(0,240))
    assert selection().plan_audition(longer)['eligible'] is True


def test_contact_sample_does_not_override_measured_compression_risk():
    value=plan()
    value['shots'][0]['measurements'][0]['blockiness']=8
    result=selection().plan_audition(value)
    assert not result['eligible'] and result['reason']=='retain_tunable_model_for_damage'


def test_real_graphics_signature_prevents_speculative_second_model_in_entire_source():
    value = plan()
    value['shots'][0]['visual_treatment'] = {
        'schema': 'spiffy_visual_treatment_v1', 'mode': 'graphics_edge_guard',
        'genre': 'not_inferred', 'scope': 'measured_risk_not_semantic_genre'}
    result = selection().plan_audition(value)
    assert result['eligible'] is False
    assert result['reason'] == 'graphic_content_preserve_existing_model'


def test_faster_model_requires_all_image_gates_and_pilot_cost_payback():
    evidence={'status':'PASS_SOURCE_AND_REFERENCE_GUARDS','reasons':[],'frames':16}
    good=selection().choose_model(evidence,20.,10.,pilot_frames=16,total_frames=480)
    assert good['model']=='ahq-12'
    bad=selection().choose_model({**evidence,'status':'REJECT','reasons':['texture_loss']},20.,10.,pilot_frames=16,total_frames=480)
    assert bad['model']=='prob-4'
    slow=selection().choose_model(evidence,20.,19.,pilot_frames=16,total_frames=480)
    assert slow['model']=='prob-4'
    short=selection().choose_model(evidence,20.,10.,pilot_frames=16,total_frames=32)
    assert short['model']=='prob-4' and short['reason']=='pilot_not_repaid'


def test_startup_only_advantage_must_not_be_multiplied_by_every_output_frame():
    evidence={'status':'PASS_SOURCE_AND_REFERENCE_GUARDS','reasons':[],'frames':8}
    # Artemis starts dramatically faster, but then renders each additional
    # frame at half Proteus' throughput. The old linear-total calculation
    # falsely favored Artemis for long videos.
    decision=selection().choose_model(
        evidence,20.,5.,pilot_frames=8,total_frames=400,
        reference_first_frame_seconds=19.,candidate_first_frame_seconds=3.)
    assert decision['model']=='prob-4'
    assert decision['reason']=='startup_aware_speed_not_repaid'
    assert decision['projection_scope']=='measured_first_frame_and_steady_state_estimate_not_full_export_proof'
    assert decision['candidate_steady_seconds_per_frame']>decision['reference_steady_seconds_per_frame']
    assert decision['projected_remaining_savings_seconds']<0


def test_genuine_faster_steady_frames_can_pay_for_two_model_startups():
    evidence={'status':'PASS_SOURCE_AND_REFERENCE_GUARDS','reasons':[],'frames':8}
    choice=selection().choose_model(
        evidence,20.,10.,pilot_frames=8,total_frames=400,
        reference_first_frame_seconds=15.,candidate_first_frame_seconds=8.)
    assert choice['model']=='ahq-12'
    assert choice['reason']=='qualified_local_candidate_with_cost_margin'
    assert choice['candidate_steady_seconds_per_frame']<choice['reference_steady_seconds_per_frame']
    assert choice['projected_remaining_savings_seconds']>60


def test_native_cold_start_evidence_is_explicit_and_must_be_finite():
    good={'status':'PASS_SOURCE_AND_REFERENCE_GUARDS','reasons':[],'frames':16}
    choice=selection().choose_model(
        good,40.,20.,pilot_frames=16,total_frames=1000,pilot_starts=2,
        reference_first_frame_seconds=34.,candidate_first_frame_seconds=17.)
    assert choice['pilot_starts']==2
    assert choice['production_starts_assumed']==1
    assert choice['reference_steady_seconds_per_frame']==pytest.approx(6/14)
    assert choice['candidate_steady_seconds_per_frame']==pytest.approx(3/14)
    for broken in (float('nan'),float('inf'),0.,-1.,20.,21.):
        with pytest.raises(ValueError,match='first.frame|startup'):
            selection().choose_model(
                good,20.,10.,pilot_frames=16,total_frames=1000,
                reference_first_frame_seconds=broken,
                candidate_first_frame_seconds=9.)


def test_global_model_choice_rejects_measured_steady_slowdown_on_second_shot():
    # A global model must not be chosen from aggregate time alone when it
    # demonstrably processes one of two different shots materially slower.
    source_shots = [
        {"window": [16, 24], "reference_seconds": 20., "candidate_seconds": 14.,
         "reference_first_frame_seconds": 16., "candidate_first_frame_seconds": 11.,
         "comparison": {"status": "PASS_SOURCE_AND_REFERENCE_GUARDS",
                        "frames": 8, "reasons": []}},
        {"window": [200, 208], "reference_seconds": 20., "candidate_seconds": 12.,
         "reference_first_frame_seconds": 15., "candidate_first_frame_seconds": 5.,
         "comparison": {"status": "PASS_SOURCE_AND_REFERENCE_GUARDS",
                        "frames": 8, "reasons": []}},
    ]
    result, ref, cand, frames = selection()._aggregate(source_shots)
    assert ref == 40 and cand == 26 and frames == 16
    assert result["status"] == "REJECT"
    assert "per_shot_steady_speed_regression" in result["reasons"]
    assert len(result["shot_speed_evidence"]) == 2
    assert result["shot_speed_evidence"][0]["candidate_steady_seconds_per_frame"] < \
        result["shot_speed_evidence"][0]["reference_steady_seconds_per_frame"]
    assert result["shot_speed_evidence"][1]["candidate_steady_seconds_per_frame"] > \
        result["shot_speed_evidence"][1]["reference_steady_seconds_per_frame"]
    decision = selection().choose_model(
        result, ref, cand, pilot_frames=frames, total_frames=10000,
        pilot_starts=2,
        reference_first_frame_seconds=sum(s["reference_first_frame_seconds"] for s in source_shots),
        candidate_first_frame_seconds=sum(s["candidate_first_frame_seconds"] for s in source_shots))
    assert decision["model"] == "prob-4"


def test_global_model_choice_accepts_consistently_faster_steady_processing():
    pilots = [
        {"window": [16, 24], "reference_seconds": 20., "candidate_seconds": 13.,
         "reference_first_frame_seconds": 15., "candidate_first_frame_seconds": 10.,
         "comparison": {"status": "PASS_SOURCE_AND_REFERENCE_GUARDS",
                        "frames": 8, "reasons": []}},
        {"window": [200, 208], "reference_seconds": 18., "candidate_seconds": 12.,
         "reference_first_frame_seconds": 14., "candidate_first_frame_seconds": 9.,
         "comparison": {"status": "PASS_SOURCE_AND_REFERENCE_GUARDS",
                        "frames": 8, "reasons": []}},
    ]
    result, ref, cand, frames = selection()._aggregate(pilots)
    assert result["status"] == "PASS_SOURCE_AND_REFERENCE_GUARDS"
    assert not result["reasons"]
    assert len(result["shot_speed_evidence"]) == 2
    choice = selection().choose_model(
        result, ref, cand, pilot_frames=frames, total_frames=1500,
        pilot_starts=2,
        reference_first_frame_seconds=29., candidate_first_frame_seconds=19.)
    assert choice["model"] == "ahq-12"


def test_real_render_unit_count_changes_the_startup_aware_model_decision():
    evidence = {"status": "PASS_SOURCE_AND_REFERENCE_GUARDS",
                "frames": 8, "reasons": []}
    # A candidate has genuinely faster processing after startup, but is
    # expensive to initialize. Treating two actual render units as one
    # startup produces the wrong source-level choice.
    one = selection().choose_model(
        evidence, 17., 22.1, pilot_frames=8, total_frames=130,
        reference_first_frame_seconds=10., candidate_first_frame_seconds=20.,
        production_starts=1)
    two = selection().choose_model(
        evidence, 17., 22.1, pilot_frames=8, total_frames=130,
        reference_first_frame_seconds=10., candidate_first_frame_seconds=20.,
        production_starts=2)
    assert one["model"] == "ahq-12"
    assert two["model"] == "prob-4"
    assert two["reason"] == "startup_aware_speed_not_repaid"
    assert one["production_starts_assumed"] == 1
    assert two["production_starts_assumed"] == 2
    assert one["projected_remaining_savings_seconds"] > \
        two["projected_remaining_savings_seconds"]


def test_actual_source_render_units_exclude_source_preservation_from_model_starts():
    source = plan()
    assert selection()._production_model_starts(source) == 1
    source.update(route="shot_sequence", render_units=[
        {"route": "source_preserved", "start_frame": 0, "end_frame": 20},
        {"route": "masked_full_frame", "start_frame": 20, "end_frame": 120},
        {"route": "masked_crop", "start_frame": 120, "end_frame": 200},
        {"route": "source_preserved", "start_frame": 200, "end_frame": 240},
    ])
    assert selection()._production_model_starts(source) == 2
    source["render_units"][2]["route"] = "source_preserved"
    assert selection()._production_model_starts(source) == 1
    source["render_units"] = []
    with pytest.raises(ValueError, match="production.*units"):
        selection()._production_model_starts(source)


def test_real_pilot_wrapper_uses_its_own_native_first_frame_timing_not_stale_writer_data(
        tmp_path, monkeypatch):
    import time
    import adaptive_enhancement

    value=plan(8,(0,8))
    value['fingerprint']=adaptive.plan_fingerprint(value)
    writer=SimpleNamespace(
        temp_dir=tmp_path,cancel_requested=False,
        process_timings=[{'phase':'adaptive_enhancement_pilot','status':'completed',
                          'emitted_frames':8,'first_frame_seconds':999.}])
    monkeypatch.setattr(adaptive_enhancement,'verify_artifacts',lambda *_:None)
    calls=[]

    def fake_native(source,dest,ffmpeg,info,preset,width,height,w,*,recipe,adaptive_plan,**kwargs):
        calls.append(recipe.up_model)
        time.sleep(.022)
        dest.write_bytes(b'owned-test-video-fixture')
        w.process_timings.append({
            'phase':'adaptive_enhancement_pilot',
            'status':'completed','emitted_frames':8,
            'first_frame_seconds':.006 if recipe.up_model=='prob-4' else .009})

    monkeypatch.setattr(worker,'topaz_upscale_to_prores',fake_native)
    monkeypatch.setattr(selection(),'compare_pilot',lambda *args,**kwargs:
        {'status':'PASS_SOURCE_AND_REFERENCE_GUARDS','frames':8,'reasons':[]})
    trial=selection().run_pilot_pair(
        tmp_path/'source',{},value,[0,8],{'ffmpeg':'only-fixture'},writer,'auto')
    assert calls==['prob-4','ahq-12']
    assert trial['reference_first_frame_seconds']==.006
    assert trial['candidate_first_frame_seconds']==.009
    assert trial['reference_seconds']>.006 and trial['candidate_seconds']>.009
    assert not list(tmp_path.glob('model_pilot_*.mov'))


@pytest.mark.parametrize('value',[True,0,-1,float('nan'),float('inf')])
def test_invalid_timing_cannot_select_a_model(value):
    with pytest.raises(ValueError):
        selection().choose_model({'status':'PASS_SOURCE_AND_REFERENCE_GUARDS','reasons':[],'frames':8},value,1.,pilot_frames=8,total_frames=240)


def test_masked_graph_does_not_send_proteus_controls_to_artemis():
    value=plan()
    value['effective_params']=''
    graph=adaptive.enhancement_filter_graph(value,'ahq-12','-2')
    assert 'model=ahq-12' in graph and 'maskedmerge' in graph and 'download=0' in graph
    assert ':details=' not in graph
    value['effective_params']=':details=.2'
    with pytest.raises(ValueError,match='controls|Artemis'):
        adaptive.enhancement_filter_graph(value,'ahq-12','-2')
    with pytest.raises(ValueError):
        adaptive.enhancement_filter_graph(value,'iris-3','-2')


def test_nonadaptive_manual_recipe_and_short_input_do_not_start_any_pilot(tmp_path,monkeypatch):
    import delivery_insights
    monkeypatch.setattr(delivery_insights,'resource_snapshot',lambda:pytest.fail('No eligible audition'))
    writer=SimpleNamespace(temp_dir=tmp_path,cancel_requested=False)
    recipe=worker.EnhancementRecipe('auto','prob-4',':details=.2',explicit=False)
    value=plan(24,(0,24))
    unchanged,decision=selection().select_for_source(tmp_path/'source',{},value,{},writer,recipe)
    assert unchanged==value and decision['reason']=='pilot_cost_budget'


def test_resource_gate_is_explicit_and_portable_without_stopping_other_apps():
    result=selection().resource_admission({'platform':'darwin','ram_available_bytes':1*1024**3},plan())
    assert not result['allowed'] and result['reason']=='system_memory_headroom'
    result=selection().resource_admission({'platform':'darwin','ram_available_bytes':16*1024**3},plan())
    assert not result['allowed'] and result['reason']=='native_model_audition_not_qualified_on_platform'


def write_frames(path:Path, frames:list[np.ndarray],rate=Fraction(24)):
    import av
    with av.open(str(path),'w') as out:
        stream=out.add_stream('png',rate=rate)
        stream.width,stream.height=frames[0].shape[1],frames[0].shape[0]
        stream.pix_fmt='rgb24'
        for i,pixels in enumerate(frames):
            frame=av.VideoFrame.from_ndarray(pixels,format='rgb24')
            frame.pts,frame.time_base=i,1/rate
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def test_real_frame_comparison_rejects_local_texture_loss_and_clock_drift(tmp_path):
    import cv2
    yy,xx=np.mgrid[:90,:160]
    gray=np.clip(110+20*np.sin(xx*.9)+15*np.cos(yy*.8),0,255).astype(np.uint8)
    frame=np.repeat(gray[...,None],3,axis=2)
    pixels=[np.roll(frame,i,axis=1) for i in range(8)]
    source=tmp_path/'source.mov'
    reference=tmp_path/'reference.mov'
    candidate=tmp_path/'candidate.mov'
    write_frames(source,pixels)
    write_frames(reference,pixels)
    write_frames(candidate,pixels)
    writer=SimpleNamespace(cancel_requested=False)
    value=plan(8,(0,8))
    value.update(source_width=160,source_height=90,target_width=160,target_height=90)
    evidence=selection().compare_pilot(source,reference,candidate,value,(0,8),writer)
    assert evidence['status']=='PASS_SOURCE_AND_REFERENCE_GUARDS'
    write_frames(candidate,[cv2.GaussianBlur(p,(0,0),2) for p in pixels])
    evidence=selection().compare_pilot(source,reference,candidate,value,(0,8),writer)
    assert evidence['status']=='REJECT'
    assert any(r in evidence['reasons'] for r in ('texture_loss_or_oversharpening','local_detail_changed'))
    write_frames(candidate,pixels,rate=Fraction(30))
    with pytest.raises(RuntimeError,match='clock|time'):
        selection().compare_pilot(source,reference,candidate,value,(0,8),writer)


def test_actual_upscale_native_pixels_reject_checkerboard_that_downscale_hides(tmp_path):
    """A 1-pixel hallucinated texture must not disappear at source-size QA."""
    import cv2
    rng=np.random.default_rng(9021)
    gray=cv2.GaussianBlur(rng.integers(45,220,(90,160),dtype=np.uint8),(0,0),1.0)
    original=np.repeat(gray[...,None],3,axis=2)
    originals=[np.roll(original,i,axis=1) for i in range(8)]
    high_res=[cv2.resize(frame,(320,180),interpolation=cv2.INTER_LANCZOS4) for frame in originals]
    yy,xx=np.mgrid[:180,:320]
    disturbance=np.where((xx+yy)%2==0,16,-16).astype(np.int16)
    altered=[np.uint8(np.clip(frame.astype(np.int16)+disturbance[...,None],0,255)) for frame in high_res]
    source=tmp_path/'source.mov'
    reference=tmp_path/'proteus_reference.mov'
    candidate=tmp_path/'overtextured_candidate.mov'
    write_frames(source,originals)
    write_frames(reference,high_res)
    write_frames(candidate,altered)
    value=plan(8,(0,8))
    value.update(source_width=160,source_height=90,target_width=320,target_height=180)
    result=selection().compare_pilot(source,reference,candidate,value,(0,8),
        SimpleNamespace(cancel_requested=False))
    assert result['status']=='REJECT'
    assert 'native_high_frequency_artifacts' in result['reasons']
    assert result['native_detail']['screened_frames']>=7
    assert (result['native_detail']['screened_frames'] +
            result['native_detail']['skipped_no_active_mask'])==8
    assert result['native_detail']['worst_energy_ratio']>1.5


def test_native_upscale_quality_gate_accepts_matched_detail_and_rejects_tiny_rgb_damage(tmp_path):
    import cv2
    rng=np.random.default_rng(287)
    gray=cv2.GaussianBlur(rng.integers(45,220,(90,160),dtype=np.uint8),(0,0),1.0)
    rgb=np.repeat(gray[...,None],3,axis=2)
    frames=[np.roll(rgb,i,axis=1) for i in range(8)]
    enlarged=[cv2.resize(f,(320,180),interpolation=cv2.INTER_LANCZOS4) for f in frames]
    src,reference,candidate=(tmp_path/name for name in ('source.mov','reference.mov','candidate.mov'))
    for path,pixels in ((src,frames),(reference,enlarged),(candidate,enlarged)):
        write_frames(path,pixels)
    value=plan(8,(0,8))
    value.update(source_width=160,source_height=90,target_width=320,target_height=180)
    result=selection().compare_pilot(src,reference,candidate,value,(0,8),
        SimpleNamespace(cancel_requested=False))
    assert result['status']=='PASS_SOURCE_AND_REFERENCE_GUARDS',result['reasons']
    assert result['native_detail']['screened_frames']==8
    assert all(abs(row['native_detail_energy_ratio']-1)<.02 for row in result['samples']
               if row['native_detail_energy_ratio'] is not None)


def test_native_protected_caption_check_is_not_diluted_by_large_static_background():
    import av
    reference=np.full((180,320,3),108,dtype=np.uint8)
    changed=reference.copy()
    # Less than 1% of image area, but ALL the model-changed pixels belong to
    # glyph protection. A global background p99 would misleadingly be zero.
    changed[35:45,80:100]=137
    no_detail=np.zeros((90,160),np.uint8)
    glyph=np.zeros_like(no_detail)
    glyph[17:23,39:51]=255
    actual,_=selection()._native_resolution_detail_witness(
        av.VideoFrame.from_ndarray(reference,format='rgb24'),
        av.VideoFrame.from_ndarray(changed,format='rgb24'),
        no_detail,glyph,None)
    assert actual['eligible_pixels']==0
    assert actual['status']=='REJECT'
    assert 'native_protected_pixels_changed' in actual['reasons']
    assert actual['native_glyph_p99']>20


def test_no_native_detail_witness_cannot_approve_a_different_model_on_flat_upscale(tmp_path):
    flat=np.full((90,160,3),90,np.uint8)
    target=np.full((180,320,3),90,np.uint8)
    source=tmp_path/'flat_source.mov'
    reference=tmp_path/'reference.mov'
    candidate=tmp_path/'candidate.mov'
    write_frames(source,[flat]*8)
    write_frames(reference,[target]*8)
    write_frames(candidate,[target]*8)
    value=plan(8,(0,8))
    value.update(source_width=160,source_height=90,target_width=320,target_height=180)
    result=selection().compare_pilot(source,reference,candidate,value,(0,8),
        SimpleNamespace(cancel_requested=False))
    assert result['status']=='REJECT'
    assert 'native_detail_evidence_insufficient' in result['reasons']
    assert result['native_detail']['screened_frames']==0
    assert result['native_detail']['skipped_no_active_mask']==8


def test_auto_model_policy_round_trips_to_existing_delivery_plan(tmp_path):
    import workflow.final_delivery_pipeline as pipeline
    from tools import run_spiffy_final_delivery as runner
    value={'mode':'natural','model_search':'off'}
    assert adaptive.normalize_policy(value)==value
    result=pipeline.build_final_delivery_plan(tmp_path/'source',tmp_path/'out',
        profile_ids=['smart_finish_master'],adaptive_enhancement=value)
    assert result['adaptive_enhancement']==value
    args=runner.parse_args(['--input','input.mp4','--enhancement-model-search','auto'])
    assert args.enhancement_model_search=='auto'
    with pytest.raises(ValueError):
        adaptive.normalize_policy({'mode':'natural','model_search':'cloud'})


def test_model_witness_uses_actual_batched_recipe_not_unexecuted_shot_suggestion():
    value=plan()
    value['shots'][0]['effective_controls']={'details':.18,'blend':.5}
    value.update(effective_params=':details=0.08:blend=0.65',effective_controls={'details':.08,'blend':.65})
    actual=selection()._pilot_plan(value,[8,16],'prob-4')
    assert actual['effective_params']==':details=0.08:blend=0.65'
    assert actual['execution_window']=={'start_frame':8,'end_frame':16}
    assert actual['crop']==value['crop']


def _selector_fixture(tmp_path,monkeypatch):
    import hashlib
    import delivery_insights
    source=tmp_path/'source.mov'
    source.write_bytes(b'Source-bound control fixture, not native inference.')
    runtime=tmp_path/'topaz.exe'
    runtime.write_bytes(b'Runtime identity fixture, never executed.')
    mask=tmp_path/'mask.nut'
    mask.write_bytes(b'Integrity-only mask fixture.')
    value=plan()
    value.update(mask_path=str(mask),mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest(),
                 identity={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    value['fingerprint']=adaptive.plan_fingerprint(value)
    assets={m:{'resume_safe':True,'identity_sha256':hashlib.sha256(m.encode()).hexdigest()} for m in ('prob-4','ahq-12')}
    monkeypatch.setattr(worker,'topaz_model_asset_identity',lambda root,model:dict(assets[model]))
    monkeypatch.setattr(delivery_insights,'resource_snapshot',lambda:{'platform':'win32',
        'ram_available_bytes':16*1024**3,'gpu_memory_free_mib':8000,'gpu_utilization_percent':0})
    writer=SimpleNamespace(temp_dir=tmp_path,cancel_requested=False,update=lambda **k:None,log=lambda *a:None)
    topaz={'ffmpeg':str(runtime),'model_dir':str(tmp_path)}
    recipe=worker.EnhancementRecipe('auto','prob-4',':details=.2',explicit=False)
    variants=[{'width':320,'height':180,'layout':'keep_source_aspect'}]
    return source,runtime,mask,value,assets,writer,topaz,recipe,variants


def _trial(window,status='PASS_SOURCE_AND_REFERENCE_GUARDS'):
    return {'window':window,'reference_seconds':20.,'candidate_seconds':5.,
        'comparison':{'status':status,'frames':8,'reasons':[] if status.startswith('PASS') else ['texture_loss_or_oversharpening']}}


def test_real_selector_cache_revalidates_decision_and_never_reexecutes_qualified_evidence(tmp_path,monkeypatch):
    import delivery_insights
    source,runtime,mask,value,assets,writer,topaz,recipe,variants=_selector_fixture(tmp_path,monkeypatch)
    calls=[]
    def pilot(src,info,p,w,t,wr,preset,**kw):
        calls.append(w)
        assert kw['deadline']>selection().time.monotonic()
        return _trial(w)
    monkeypatch.setattr(selection(),'run_pilot_pair',pilot)
    result,receipt=selection().select_for_source(source,{},value,topaz,writer,recipe,delivery_variants=variants)
    assert receipt['model']=='ahq-12' and len(calls)==2
    assert result['fingerprint']!=value['fingerprint']
    assert selection().execution_model(result)=='ahq-12'
    assert value['effective_params']==':details=0.12:blend=0.5'
    monkeypatch.setattr(delivery_insights,'resource_snapshot',lambda:pytest.fail('Validated evidence requires no new GPU trial'))
    again,reused=selection().select_for_source(source,{},value,topaz,writer,recipe,delivery_variants=variants)
    assert reused['cache_hit'] and again==result and len(calls)==2
    assets['ahq-12']['identity_sha256']='a'*64
    with pytest.raises(selection().ModelSelectionIntegrityError):
        selection().validate_selected_assets(result,str(tmp_path))


def test_first_shot_rejection_is_cached_instead_of_retesting_other_shots(tmp_path,monkeypatch):
    source,runtime,mask,value,assets,writer,topaz,recipe,variants=_selector_fixture(tmp_path,monkeypatch)
    calls=[]
    def reject(src,info,p,w,t,wr,preset,**kw):
        calls.append(w)
        return _trial(w,'REJECT')
    monkeypatch.setattr(selection(),'run_pilot_pair',reject)
    result,receipt=selection().select_for_source(source,{},value,topaz,writer,recipe,delivery_variants=variants)
    assert result==value and receipt['model']=='prob-4' and len(calls)==1
    again,reused=selection().select_for_source(source,{},value,topaz,writer,recipe,delivery_variants=variants)
    assert reused['cache_hit'] and len(calls)==1 and again==value


@pytest.mark.parametrize('drift',['source','runtime','mask','asset'])
def test_integrity_drift_during_candidate_failure_is_never_treated_as_optional(tmp_path,monkeypatch,drift):
    source,runtime,mask,value,assets,writer,topaz,recipe,variants=_selector_fixture(tmp_path,monkeypatch)
    def corrupt(*args,**kwargs):
        if drift=='asset':
            assets['ahq-12']['identity_sha256']='d'*64
        else:
            {'source':source,'runtime':runtime,'mask':mask}[drift].write_bytes(b'changed')
        raise RuntimeError('Candidate failed')
    monkeypatch.setattr(selection(),'run_pilot_pair',corrupt)
    with pytest.raises(selection().ModelSelectionIntegrityError):
        selection().select_for_source(source,{},value,topaz,writer,recipe,delivery_variants=variants)


def test_optional_timeout_preserves_reference_but_cancel_propagates(tmp_path,monkeypatch):
    import subprocess
    from shared.gpu_lock import GpuLockCancelled
    source,runtime,mask,value,assets,writer,topaz,recipe,variants=_selector_fixture(tmp_path,monkeypatch)
    def expired(*a,**kw):
        raise subprocess.TimeoutExpired('owned-native-pilot',1)
    monkeypatch.setattr(selection(),'run_pilot_pair',expired)
    result,receipt=selection().select_for_source(source,{},value,topaz,writer,recipe,delivery_variants=variants)
    assert result==value and receipt['reason']=='candidate_trial_not_qualified'
    writer.cancel_requested=True
    with pytest.raises(GpuLockCancelled):
        selection().select_for_source(source,{},value,topaz,writer,recipe,delivery_variants=variants)


def test_high_resolution_master_does_not_inherit_source_scale_quality_approval(tmp_path,monkeypatch):
    source,runtime,mask,value,assets,writer,topaz,recipe,variants=_selector_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(selection(),'run_pilot_pair',lambda *a,**k:pytest.fail('Unqualified final canvas'))
    variants[0].update(width=640,height=360)
    result,receipt=selection().select_for_source(source,{},value,topaz,writer,recipe,delivery_variants=variants)
    assert result==value and receipt['reason'].startswith('delivery_geometry_not_qualified')


def test_changed_source_cannot_borrow_an_older_mask_plan(tmp_path,monkeypatch):
    source,runtime,mask,value,assets,writer,topaz,recipe,variants=_selector_fixture(tmp_path,monkeypatch)
    source.write_bytes(b'New source, same path')
    monkeypatch.setattr(selection(),'run_pilot_pair',lambda *a,**k:pytest.fail('Changed source must not render'))
    with pytest.raises(selection().ModelSelectionIntegrityError,match='analyzed masks'):
        selection().select_for_source(source,{},value,topaz,writer,recipe,delivery_variants=variants)


def test_cache_timing_or_decision_tamper_reexecutes_instead_of_accepting_false_evidence(tmp_path,monkeypatch):
    import json
    source,runtime,mask,value,assets,writer,topaz,recipe,variants=_selector_fixture(tmp_path,monkeypatch)
    calls=[]
    def trial(src,info,p,w,t,wr,preset,**kw):
        calls.append(w)
        return _trial(w)
    monkeypatch.setattr(selection(),'run_pilot_pair',trial)
    selection().select_for_source(source,{},value,topaz,writer,recipe,delivery_variants=variants)
    cache=next(tmp_path.glob('enhancement_choice_*.json'))
    data=json.loads(cache.read_text())
    data['seconds']=0.00001
    cache.write_text(json.dumps(data))
    result,receipt=selection().select_for_source(source,{},value,topaz,writer,recipe,delivery_variants=variants)
    assert len(calls)==4 and not receipt['cache_hit']


def test_real_worker_argv_uses_selected_model_blank_controls_and_hard_pilot_deadline(tmp_path,monkeypatch):
    import hashlib
    mask=tmp_path/'mask.nut'
    mask.write_bytes(b'command-only mask identity')
    value=plan(8,(0,8))
    value.update(mask_path=str(mask),mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest(),
        effective_params='',execution_window={'start_frame':0,'end_frame':8})
    value['fingerprint']=adaptive.plan_fingerprint(value)
    calls=[]
    monkeypatch.setattr(worker,'run_process',lambda argv,writer,phase,**kw:calls.append((argv,phase,kw)))
    writer=SimpleNamespace(cancel_requested=False)
    worker.topaz_upscale_to_prores(tmp_path/'src',tmp_path/'out','native.exe',{},'auto',640,360,writer,
        recipe=worker.EnhancementRecipe('auto','ahq-12','',False),adaptive_plan=value,audition=True,audition_timeout=12)
    argv,phase,kw=calls[0]
    graph=argv[argv.index('-filter_complex')+1]
    assert 'tvai_up=model=ahq-12' in graph and ':details=' not in graph and ':download=0' in graph
    assert phase=='adaptive_enhancement_pilot' and kw['timeout']==12 and kw['priority']=='below_normal'


def test_parent_worker_selected_model_reaches_render_and_checkpoint_with_real_cpu_delivery(tmp_path,monkeypatch):
    """Controlled selected-model result; CPU encode proves wiring, not Topaz fidelity."""
    import hashlib
    import json
    import subprocess
    import sys
    import workflow.final_delivery_pipeline as pipeline
    from tests.test_adaptive_video_enhancement import write_fixture
    source=tmp_path/'input.mov'
    write_fixture(source,3)
    original=hashlib.sha256(source.read_bytes()).hexdigest()
    ffmpeg,ffprobe=worker.standard_ffmpeg(),worker.standard_ffprobe()
    info=worker.probe_video(source,ffprobe,inspect_timestamps=True)
    runtime=tmp_path/'identity_only.exe'
    runtime.write_bytes(b'never executed; integration identity')
    monkeypatch.setattr(worker,'find_topaz_video',lambda:{'ffmpeg':str(runtime),'ffprobe':ffprobe,
        'tvai_up':True,'tvai_fi':True,'model_dir':str(tmp_path)})
    monkeypatch.setattr(worker,'configure_topaz_model_environment',lambda p:str(tmp_path))
    monkeypatch.setattr(worker,'wait_for_render_ready',lambda *a:(source,info))
    monkeypatch.setattr(worker,'topaz_model_asset_identity',lambda root,model:{'resume_safe':True,
        'identity_sha256':'f'*64,'descriptor_present':True,'weight_or_engine_count':1,'asset_files':[],'model':model})
    def selected(src,source_info,value,topaz,writer,recipe,**kw):
        assert kw['delivery_variants']
        receipt={'fingerprint':'b'*64,'contract':{'models':{'ahq-12':'f'*64}},'decision':{'model':'ahq-12'}}
        return selection()._apply(value,receipt),{'model':'ahq-12','reason':'test_injected_selection_not_native_proof'}
    monkeypatch.setattr(selection(),'select_for_source',selected)
    observed=[]
    def cpu_standin(src,dest,exe,source_info,preset,width,height,writer,*a,**kw):
        observed.append((kw['recipe'].up_model,kw['recipe'].params,kw['adaptive_plan']['fingerprint']))
        subprocess.run([ffmpeg,'-v','error','-nostdin','-i',str(src),'-vf',f'scale={width}:{height}:flags=lanczos,setsar=1',
            '-an','-c:v','prores_ks','-profile:v','3','-pix_fmt','yuv422p10le',str(dest)],
            check=True,capture_output=True,timeout=30)
        return False
    monkeypatch.setattr(worker,'topaz_upscale_to_prores',cpu_standin)
    checkpoints=[]
    original_checkpoint=worker.build_topaz_checkpoint_fingerprint
    def capture(**kw):
        checkpoints.append((kw['stage'],kw['model'],kw.get('model_params')))
        return original_checkpoint(**kw)
    monkeypatch.setattr(worker,'build_topaz_checkpoint_fingerprint',capture)
    def child(command,child_dir,*a,**kw):
        with monkeypatch.context() as scoped:
            scoped.setattr(sys,'argv',command[1:])
            return worker.main()
    monkeypatch.setattr(pipeline,'_run_child',child)
    code,result=pipeline.run_final_delivery([source],jobs_root=tmp_path/'jobs',job_id='selected',
        output_root=tmp_path/'out',profile_ids=['smart_finish_master'],
        frame_rate={'mode':'cfr','fps':30},adaptive_enhancement={'mode':'natural','working_long_edge':1024},execute=True)
    assert code==0,result.get('errors')
    assert observed and observed[0][:2]==('ahq-12','')
    assert ('topaz_upscale','ahq-12','') in checkpoints
    assert result['state']=='passed' and hashlib.sha256(source.read_bytes()).hexdigest()==original
    child_manifest=json.loads(Path(result['sources'][0]['child_manifest_path']).read_text())
    assert next(iter(child_manifest['adaptive_enhancement_calibration'].values()))['model']=='ahq-12'
    assert next(iter(result['performance']['summary']['enhancement_model_trials'].values()))['model']=='ahq-12'
    assert all(v['model']=='ahq-12' for v in result['performance']['summary']['shots'][0]['execution']['units'])
