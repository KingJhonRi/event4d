import numpy as np
from event4d.types import ObjectObservation

def ground_detection_to_3d(detection, geometry, timestamp_us, track_id=-1, central_fraction=0.4, min_valid_ratio=0.1, source_sensor="dsec_rectified_frame_camera"):
    if not 0 < central_fraction <= 1 or not 0 <= min_valid_ratio <= 1:
        raise ValueError("invalid central fraction or depth ratio")
    box = np.asarray(detection.bbox_xyxy, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bbox must contain finite ordered xyxy coordinates")
    x1, y1, x2, y2 = box
    w, h = geometry.depth_map.shape[1], geometry.depth_map.shape[0]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    sx, sy = max(0, int(cx-(x2-x1)*central_fraction/2)), max(0, int(cy-(y2-y1)*central_fraction/2))
    ex, ey = max(0,min(w, int(cx+(x2-x1)*central_fraction/2))), max(0,min(h, int(cy+(y2-y1)*central_fraction/2)))
    roi = geometry.depth_map[sy:ey, sx:ex]
    confidence = geometry.depth_confidence[sy:ey, sx:ex]
    valid = np.isfinite(roi) & (roi > 0) & np.isfinite(confidence) & (confidence > 0)
    ratio = float(valid.mean()) if roi.size else 0.0
    position = world = depth = None
    if valid.any() and ratio >= min_valid_ratio and 0 <= cx < w and 0 <= cy < h:
        depth = float(np.median(roi[valid]))
        fx, fy, ox, oy = geometry.K[0,0], geometry.K[1,1], geometry.K[0,2], geometry.K[1,2]
        position = np.array([(cx-ox)*depth/fx, (cy-oy)*depth/fy, depth], dtype=np.float64)
        if geometry.has_world_pose:
            world = (geometry.T_world_cam @ np.r_[position, 1])[:3]
    return ObjectObservation(timestamp_us, track_id, detection.semantic_label, detection.semantic_confidence, box, position, world, depth, ratio, ratio if position is not None else 0.0, source_sensor)
