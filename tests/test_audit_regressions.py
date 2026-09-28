"""Regression cases found during the code/result/presentation audit."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import h5py
import numpy as np

from event4d.adapters.dsec import DSECAdapter
from event4d.geometry.esvo2 import ESVO2GeometryBackend
from event4d.geometry.projection import ground_detection_to_3d
from event4d.memory.world_memory import WorldMemory
from event4d.query.engine import LanguageAnswerer, QueryEngine, parse_query
from event4d.tracking.tracker import AssociationTracker
from event4d.types import Detection, GeometryFrame, ObjectObservation


def observation(seconds, xyz, track_id=1, label="car"):
    position=None if xyz is None else np.asarray(xyz,dtype=float)
    return ObjectObservation(int(seconds*1e6),track_id,label,.9,np.array([10.,10.,30.,30.]),
        position,None,None if position is None else float(position[2]),1 if position is not None else 0,
        1 if position is not None else 0,"test_fixture")


class AuditRegressions(unittest.TestCase):
    def test_velocity_uses_elapsed_time_between_valid_geometry(self):
        memory=WorldMemory(ema_alpha=1)
        for t,xyz in [(0,[0,0,10]),(1,None),(2,[0,0,8])]:
            memory.update([observation(t,xyz)])
        state=memory.objects[1]
        self.assertEqual(state.last_geometry_seen,2_000_000)
        np.testing.assert_allclose(state.velocity_camera,[0,0,-1])
        self.assertEqual(state.radial_velocity,-1)

    def test_empty_frames_expire_current_queries_and_history(self):
        memory=WorldMemory(history_s=3,ema_alpha=1)
        memory.update([observation(0,[0,0,5])])
        memory.update([],4_000_000)
        self.assertEqual(len(memory.objects[1].trajectory),0)
        engine=QueryEngine({"timestamp_us":4_000_000,"objects":memory.snapshot()})
        self.assertEqual(engine.execute(parse_query("What is closest?")),[])
        self.assertEqual(len(engine.execute(parse_query("Where is object 1?"))),1)

    def test_historical_movement_uses_window_not_latest_motion(self):
        memory=WorldMemory(ema_alpha=1)
        for t,x in [(0,0),(1,2),(4,2)]:
            memory.update([observation(t,[x,0,5])])
        engine=QueryEngine(memory.snapshot())
        self.assertEqual(memory.objects[1].motion_state,"stationary")
        self.assertEqual(len(engine.execute(parse_query("What moved in the last five seconds?"))),1)
        self.assertEqual(engine.execute(parse_query("What moved in the last 2 seconds?")),[])

    def test_lateral_movement_is_not_filtered_out(self):
        memory=WorldMemory(ema_alpha=1)
        memory.update([observation(0,[0,0,5])])
        memory.update([observation(1,[3,0,4])])
        self.assertEqual(memory.objects[1].motion_state,"stable_distance")
        engine=QueryEngine(memory.snapshot())
        self.assertEqual(len(engine.execute(parse_query("What is moving?"))),1)
        self.assertEqual(engine.execute(parse_query("What is moving within 4 meters?")),[])
        self.assertEqual(engine.execute(parse_query("What people are moving?")),[])

    def test_historical_nearby_answer_uses_matching_old_position(self):
        memory=WorldMemory(ema_alpha=1)
        memory.update([observation(0,[0,0,3])])
        memory.update([observation(1,[0,0,20])])
        hits=QueryEngine(memory.snapshot(),timestamp_us=2_000_000).execute(
            parse_query("What objects were near me during the last five seconds?"))
        self.assertEqual(hits[0]["distance_to_camera"],3)
        self.assertEqual(hits[0]["measurement_age_s"],2)

    def test_history_returns_points_and_unknown_query_is_explicit(self):
        memory=WorldMemory(ema_alpha=1)
        memory.update([observation(0,[0,0,5])])
        memory.update([observation(1,[0,0,4])])
        spec=parse_query("history of object 1 in the last five seconds")
        hits=QueryEngine(memory.snapshot()).execute(spec)
        answer=LanguageAnswerer().answer(spec,hits)
        self.assertIn("t=0 us",answer)
        self.assertIn("t=1000000 us",answer)
        unsupported=parse_query("What is the weather?")
        self.assertIn("Unsupported",LanguageAnswerer().answer(unsupported,[]))

    def test_projection_rejects_offscreen_and_zero_confidence(self):
        depth=np.full((40,40),5.,np.float32)
        g=GeometryFrame(0,depth,np.ones_like(depth),np.array([[100,0,20],[0,100,20],[0,0,1.]]),np.eye(4),False)
        outside=Detection(np.array([-30,10,-10,30]),"car",.9)
        self.assertFalse(ground_detection_to_3d(outside,g,0).geometry_valid)
        g.depth_confidence[:]=0
        inside=Detection(np.array([10,10,30,30]),"car",.9)
        self.assertFalse(ground_detection_to_3d(inside,g,0,min_valid_ratio=0).geometry_valid)

    def test_assignment_gates_invalid_edges_before_optimization(self):
        tracker=AssociationTracker(match_cost=.7)
        tracker.tracks={1:observation(0,None,1),2:observation(0,None,2)}
        tracker.next_id=3
        incoming=[observation(.5,None,-1),observation(.5,None,-1)]
        with patch("event4d.tracking.tracker.iou",side_effect=[.875,.1875,.1875,.1]):
            result=tracker.update(incoming,500_000)
        self.assertEqual([o.track_id for o in result],[2,1])

    def test_event_window_keeps_all_events_after_last_index_entry(self):
        with TemporaryDirectory() as temp:
            source=Path(temp)/"events.h5"
            times=np.r_[0,1000,2000,np.full(100_001,2001)].astype(np.int64)
            with h5py.File(source,"w") as f:
                f["t_offset"]=100_000
                f["ms_to_idx"]=np.array([0,1,2])
                f["events/t"]=times
                for key in ("x","y","p"): f[f"events/{key}"]=np.zeros(len(times),dtype=np.uint16)
            adapter=DSECAdapter.__new__(DSECAdapter)
            adapter.event_files={"left":source}
            packet=adapter.events("left",102_000,103_000)
            self.assertEqual(len(packet.x),100_002)
            self.assertEqual(len(adapter.events("left",103_000,104_000).x),0)

    def test_esvo_export_validates_confidence_and_pose(self):
        with TemporaryDirectory() as temp:
            target=Path(temp)/"000000.npz"
            kwargs=dict(depth_map=np.ones((2,2)),depth_confidence=np.array([[np.nan,1],[0,1]]),K=np.eye(3),T_world_cam=np.eye(4),timestamp_us=0)
            np.savez(target,**kwargs)
            frame=ESVO2GeometryBackend(temp).frame(0)
            self.assertEqual(np.isfinite(frame.depth_map).sum(),2)
            kwargs["T_world_cam"][0,0]=2
            np.savez(target,**kwargs)
            with self.assertRaises(ValueError): ESVO2GeometryBackend(temp).frame(0)


if __name__=="__main__": unittest.main()
