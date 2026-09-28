"""Adapter for ESVO2 exports. ESVO2 must provide calibrated depth and pose."""
from pathlib import Path
import numpy as np
from event4d.types import GeometryFrame

class ESVO2GeometryBackend:
    def __init__(self, export_dir: str | Path):
        self.root=Path(export_dir)
        if not self.root.exists(): raise FileNotFoundError(self.root)

    def frame(self,index:int)->GeometryFrame:
        path=self.root/f"{index:06d}.npz"
        with np.load(path) as data:
            depth=data["depth_map"].astype(np.float32)
            confidence=data["depth_confidence"].astype(np.float32)
            K=data["K"].astype(np.float64)
            pose=data["T_world_cam"].astype(np.float64)
            timestamp=int(data["timestamp_us"])
        if depth.ndim!=2 or confidence.shape!=depth.shape:
            raise ValueError("depth and confidence must be matching 2D maps")
        if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:
            raise ValueError("invalid calibrated K")
        if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]):
            raise ValueError("T_world_cam must be a finite homogeneous transform")
        rotation=pose[:3,:3]
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(rotation),1,atol=1e-4):
            raise ValueError("T_world_cam must contain a proper rotation")
        invalid=(confidence<=0)|~np.isfinite(confidence)|~np.isfinite(depth)|(depth<=0)
        depth[invalid]=np.nan
        confidence[invalid]=0
        return GeometryFrame(timestamp,depth,confidence,K,pose,True)
