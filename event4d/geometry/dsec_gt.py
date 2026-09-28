import numpy as np
from event4d.types import GeometryFrame

class DSECGroundTruthGeometryBackend:
    def __init__(self, adapter, min_depth=0.2, max_depth=100.0):
        self.adapter = adapter
        intr = adapter.calibration["intrinsics"]["camRect1"]["camera_matrix"]
        self.K = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]], dtype=np.float64)
        q = np.asarray(adapter.calibration["disparity_to_depth"]["cams_12"], dtype=np.float64)
        self.baseline_m = 1.0 / abs(q[3, 2])
        self.min_depth, self.max_depth = min_depth, max_depth

    def frame(self, disparity_index: int) -> GeometryFrame:
        disparity = self.adapter.disparity(disparity_index)
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        valid = disparity > 0
        depth[valid] = self.K[0, 0] * self.baseline_m / disparity[valid]
        valid &= np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
        depth[~valid] = np.nan
        return GeometryFrame(int(self.adapter.disparity_timestamps[disparity_index]), depth, valid.astype(np.float32), self.K, np.eye(4), False)
