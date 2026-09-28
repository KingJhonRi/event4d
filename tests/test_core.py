import unittest
import numpy as np
from event4d.types import Detection,GeometryFrame
from event4d.geometry.projection import ground_detection_to_3d
from event4d.tracking.tracker import AssociationTracker
from event4d.memory.world_memory import WorldMemory
from event4d.query.engine import parse_query,QueryEngine

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[100,0,50],[0,100,50],[0,0,1.]],float)
        self.box=Detection(np.array([40,40,60,60]),"car",.9)

    def test_projection_uses_median_and_rejects_sparse_depth(self):
        d=np.full((100,100),np.nan,np.float32);d[46:54,46:54]=5;d[49,49]=50
        g=GeometryFrame(0,d,np.isfinite(d).astype(float),self.K,np.eye(4),False)
        ob=ground_detection_to_3d(self.box,g,0)
        self.assertAlmostEqual(ob.depth_m,5)
        self.assertIsNone(ob.position_world_xyz)
        g.depth_map[:]=np.nan
        self.assertFalse(ground_detection_to_3d(self.box,g,0).geometry_valid)

    def test_tracking_memory_query(self):
        tracker=AssociationTracker(); memory=WorldMemory(ema_alpha=1)
        for i,z in enumerate((6.,5.,4.)):
            d=np.full((100,100),z,np.float32)
            g=GeometryFrame(i*1_000_000,d,np.ones_like(d),self.K,np.eye(4),False)
            ob=tracker.update([ground_detection_to_3d(self.box,g,i*1_000_000)],i*1_000_000)[0]
            self.assertEqual(ob.track_id,1)
            memory.update([ob])
        self.assertEqual(memory.objects[1].motion_state,"approaching")
        hits=QueryEngine(memory.snapshot()).execute(parse_query("What is approaching me within 5 meters?"))
        self.assertEqual(len(hits),1)
        spec=parse_query("What objects were near me during the last five seconds?")
        self.assertEqual(spec.intent,"recent_events")
        self.assertEqual(spec.radius_m,10)
        self.assertEqual(len(QueryEngine(memory.snapshot()).execute(spec)),1)

if __name__=="__main__": unittest.main()
