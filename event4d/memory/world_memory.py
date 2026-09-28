from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import numpy as np

@dataclass
class ObjectState:
    track_id: int
    semantic_label: str
    semantic_confidence: float
    first_seen: int
    last_seen: int
    last_geometry_seen: int | None = None
    current_position_camera: list | None = None
    current_position_world: list | None = None
    distance_to_camera: float | None = None
    velocity_camera: list | None = None
    velocity_world: list | None = None
    speed_mps: float | None = None
    speed_state: str = "unknown"
    radial_velocity: float | None = None
    motion_state: str = "unknown"
    trajectory: deque = field(default_factory=lambda: deque(maxlen=1000))
    geometry_confidence: float = 0.0
    observation_count: int = 0

    def as_dict(self):
        d=self.__dict__.copy(); d["trajectory"]=list(self.trajectory); return d

class WorldMemory:
    def __init__(self, history_s=30, ema_alpha=0.5, radial_threshold=0.3, speed_threshold=0.1, fast_threshold=0.5, max_trajectory_samples=1000):
        if history_s <= 0 or not 0 < ema_alpha <= 1 or max_trajectory_samples < 1:
            raise ValueError("history, EMA alpha, and trajectory capacity must be valid")
        self.objects={}; self.history_us=int(history_s*1e6)
        self.alpha=ema_alpha; self.radial_threshold=radial_threshold; self.speed_threshold=speed_threshold; self.fast_threshold=fast_threshold
        self.max_trajectory_samples=max_trajectory_samples
        self.timestamp_us=None

    def update(self, observations, timestamp_us=None):
        observations=list(observations)
        now=timestamp_us if timestamp_us is not None else max((o.timestamp_us for o in observations),default=self.timestamp_us)
        if now is not None and self.timestamp_us is not None and now < self.timestamp_us:
            raise ValueError("memory updates must be chronological")
        if any(now is not None and o.timestamp_us > now for o in observations):
            raise ValueError("observation cannot be newer than the memory frame")
        for ob in observations:
            state=self.objects.get(ob.track_id)
            if state is None:
                state=ObjectState(ob.track_id,ob.semantic_label,ob.semantic_confidence,ob.timestamp_us,ob.timestamp_us)
                state.trajectory=deque(maxlen=self.max_trajectory_samples)
                self.objects[ob.track_id]=state
            if ob.timestamp_us < state.last_seen:
                raise ValueError("object observations must be chronological")
            previous_t=state.last_geometry_seen
            previous=np.asarray(state.current_position_camera) if state.current_position_camera is not None else None
            previous_d=state.distance_to_camera
            state.semantic_label=ob.semantic_label
            state.semantic_confidence=max(state.semantic_confidence,ob.semantic_confidence)
            state.last_seen=ob.timestamp_us
            state.observation_count+=1
            if ob.position_camera_xyz is not None:
                raw=ob.position_camera_xyz
                current=raw if previous is None else self.alpha*raw+(1-self.alpha)*previous
                distance=float(np.linalg.norm(current))
                dt=(ob.timestamp_us-previous_t)/1e6 if previous_t is not None else 0
                if previous is not None and dt>0:
                    velocity=(current-previous)/dt
                    state.velocity_camera=velocity.tolist()
                    state.speed_mps=float(np.linalg.norm(velocity))
                    state.speed_state=("stationary" if state.speed_mps<self.speed_threshold else "moving" if state.speed_mps<=self.fast_threshold else "fast_moving")
                    state.radial_velocity=(distance-previous_d)/dt
                    if state.radial_velocity < -self.radial_threshold: state.motion_state="approaching"
                    elif state.radial_velocity > self.radial_threshold: state.motion_state="moving_away"
                    elif np.linalg.norm(velocity) < self.speed_threshold: state.motion_state="stationary"
                    else: state.motion_state="stable_distance"
                state.current_position_camera=current.tolist()
                state.distance_to_camera=distance
                previous_world=state.current_position_world
                state.current_position_world=ob.position_world_xyz.tolist() if ob.position_world_xyz is not None else None
                state.velocity_world=((np.asarray(state.current_position_world)-previous_world)/dt).tolist() if previous_world is not None and state.current_position_world is not None and dt>0 else None
                state.last_geometry_seen=ob.timestamp_us
                state.geometry_confidence=ob.geometry_confidence
                state.trajectory.append({"timestamp_us":ob.timestamp_us,"xyz_camera":current.tolist(),"xyz_world":state.current_position_world,
                    "velocity_camera":state.velocity_camera,"speed_mps":state.speed_mps,"radial_velocity":state.radial_velocity,
                    "motion_state":state.motion_state,"interval_start_us":previous_t})
        if now is not None:
            self.timestamp_us=int(now)
            self._prune_history(self.timestamp_us)

    def _prune_history(self, timestamp_us):
        for state in self.objects.values():
            while state.trajectory and timestamp_us-state.trajectory[0]["timestamp_us"]>self.history_us:
                state.trajectory.popleft()

    def snapshot(self, timestamp_us=None):
        if timestamp_us is not None:
            self.update([],timestamp_us)
        return {str(k):v.as_dict() for k,v in self.objects.items()}
