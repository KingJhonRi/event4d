from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(a, b):
    x1,y1 = np.maximum(a[:2],b[:2]); x2,y2 = np.minimum(a[2:],b[2:])
    inter = max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter+1e-9)

class AssociationTracker:
    def __init__(self, max_age_s=1.0, match_cost=0.99):
        self.tracks={}; self.next_id=1; self.max_age_us=int(max_age_s*1e6); self.match_cost=match_cost

    def update(self, observations, timestamp_us):
        live={k:v for k,v in self.tracks.items() if timestamp_us-v.timestamp_us<=self.max_age_us}
        ids=list(live)
        cost=np.full((len(ids),len(observations)), 1e4)
        for i,tid in enumerate(ids):
            old=live[tid]
            for j,new in enumerate(observations):
                if old.semantic_label != new.semantic_label: continue
                overlap=iou(old.bbox_xyxy,new.bbox_xyxy)
                dist=0.0
                if old.position_camera_xyz is not None and new.position_camera_xyz is not None:
                    dist=float(np.linalg.norm(old.position_camera_xyz-new.position_camera_xyz))
                if overlap==0 and (old.position_camera_xyz is None or new.position_camera_xyz is None or dist>1.5):
                    continue
                cost[i,j]=0.8*(1-overlap)+0.2*min(dist/5,3)
        if cost.size:
            # Ineligible pairs must not steal assignments before thresholding.
            cost[cost >= self.match_cost] = 1e4
            rows,cols=linear_sum_assignment(cost)
            for i,j in zip(rows,cols):
                if cost[i,j] < self.match_cost:
                    observations[j].track_id=ids[i]
        for ob in observations:
            if ob.track_id<0:
                ob.track_id=self.next_id; self.next_id+=1
            self.tracks[ob.track_id]=ob
        self.tracks={k:v for k,v in self.tracks.items() if timestamp_us-v.timestamp_us<=self.max_age_us}
        return observations
