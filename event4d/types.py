from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class EventPacket:
    camera_id: str
    timestamps_us: np.ndarray
    x: np.ndarray
    y: np.ndarray
    polarity: np.ndarray
    width: int
    height: int

@dataclass
class StereoEventSlice:
    t_start_us: int
    t_end_us: int
    left: EventPacket
    right: EventPacket

@dataclass
class GeometryFrame:
    timestamp_us: int
    depth_map: np.ndarray
    depth_confidence: np.ndarray
    K: np.ndarray
    T_world_cam: np.ndarray
    has_world_pose: bool

@dataclass
class ReconstructedFrame:
    timestamp_us: int
    image: np.ndarray
    source: str

@dataclass
class Detection:
    bbox_xyxy: np.ndarray
    semantic_label: str
    semantic_confidence: float

@dataclass
class ObjectObservation:
    timestamp_us: int
    track_id: int
    semantic_label: str
    semantic_confidence: float
    bbox_xyxy: np.ndarray
    position_camera_xyz: np.ndarray | None
    position_world_xyz: np.ndarray | None
    depth_m: float | None
    depth_valid_ratio: float
    geometry_confidence: float
    source_sensor: str

    @property
    def geometry_valid(self) -> bool:
        return self.position_camera_xyz is not None
