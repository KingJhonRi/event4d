from __future__ import annotations
from dataclasses import dataclass
import re
import math

@dataclass
class QuerySpec:
    intent: str
    radius_m: float=10.0
    time_window_s: float=5.0
    track_id: int | None=None
    label_filter: str | None=None
    motion_filter: str | None=None
    limit: int=20

def parse_query(text: str) -> QuerySpec:
    s=text.lower(); intent="unsupported"
    if "history" in s: intent="object_history"
    elif "closest" in s or "nearest" in s: intent="closest_object"
    elif "approach" in s: intent="approaching_objects"
    elif "where" in s or "location" in s: intent="object_location"
    elif "moved" in s and ("last" in s or "past" in s or "during" in s): intent="recent_events"
    elif "moved" in s or "moving" in s: intent="moving_objects"
    elif "happened" in s or "during" in s or "were near" in s: intent="recent_events"
    elif any(word in s for word in ("near", "within", "under", "less than")): intent="nearby_objects"
    radius=re.search(r"(?:within|under|less than)\s+(\d+(?:\.\d+)?)\s*(?:m|meters?)",s)
    time=re.search(r"(?:last|past)\s+(\d+(?:\.\d+)?|five|ten)\s+seconds?",s)
    track=re.search(r"(?:object|person|car|truck|bus|bicycle|sign)\s*#?\s*(\d+)",s)
    num={"five":5,"ten":10}
    default_radius=10.0 if intent=="nearby_objects" or (intent=="recent_events" and "near" in s) else float("inf")
    motion="moving" if intent=="recent_events" and "moved" in s else None
    label=re.search(r"\b(traffic signs?|traffic lights?|people|persons?|cars?|trucks?|buses|bus|bicycles?|motorcycles?|backpacks?)\b",s)
    aliases={"people":"person","persons":"person","buses":"bus"}
    label_filter=None
    if label:
        raw=label.group(1)
        label_filter=aliases.get(raw,raw[:-1] if raw.endswith("s") and raw!="bus" else raw)
    return QuerySpec(intent,float(radius.group(1)) if radius else default_radius,float(num.get(time.group(1),time.group(1))) if time else 5.0,int(track.group(1)) if track else None,label_filter,motion)

class QueryEngine:
    def __init__(self, snapshot, timestamp_us=None):
        if isinstance(snapshot,dict) and "objects" in snapshot:
            timestamp_us=snapshot["timestamp_us"] if timestamp_us is None else timestamp_us
            snapshot=snapshot["objects"]
        self.objects=list(snapshot.values()) if isinstance(snapshot,dict) else snapshot
        # Record time advances even on frames without detections.
        self.timestamp_us=timestamp_us if timestamp_us is not None else max((o["last_seen"] for o in self.objects),default=0)

    def execute(self, spec: QuerySpec):
        if spec.intent=="unsupported": return []
        now=self.timestamp_us
        objects=[o for o in self.objects if o.get("distance_to_camera") is not None and math.isfinite(o["distance_to_camera"])]
        if spec.track_id is not None: objects=[o for o in objects if o["track_id"]==spec.track_id]
        if spec.label_filter: objects=[o for o in objects if o["semantic_label"]==spec.label_filter]
        current=spec.intent in ("nearby_objects","closest_object","approaching_objects","moving_objects")
        historical=spec.intent in ("recent_events","object_history")
        results=[]
        for obj in objects:
            trajectory=obj.get("trajectory",[])
            measured_at=obj.get("last_geometry_seen")
            if measured_at is None: measured_at=trajectory[-1]["timestamp_us"] if trajectory else None
            if measured_at is None: continue
            if current and not (0<=now-obj["last_seen"]<=2_000_000 and 0<=now-measured_at<=2_000_000): continue
            result=dict(obj)
            if historical:
                cutoff=now-int(spec.time_window_s*1e6)
                matches=[]
                for index,sample in enumerate(trajectory):
                    if not cutoff<=sample["timestamp_us"]<=now: continue
                    if math.dist(sample["xyz_camera"],(0,0,0))>spec.radius_m: continue
                    if spec.motion_filter=="moving":
                        # Both measured endpoints must fall in the requested window.
                        previous=trajectory[index-1] if index else None
                        if previous is None or previous["timestamp_us"]<cutoff: continue
                        dt=(sample["timestamp_us"]-previous["timestamp_us"])/1e6
                        if dt<=0 or math.dist(sample["xyz_camera"],previous["xyz_camera"])/dt<=0.1: continue
                    matches.append(sample)
                if not matches: continue
                sample=matches[-1]; measured_at=sample["timestamp_us"]
                result.update(current_position_camera=sample["xyz_camera"],distance_to_camera=math.dist(sample["xyz_camera"],(0,0,0)),
                    motion_state=sample.get("motion_state","unknown"),speed_mps=sample.get("speed_mps"),matched_trajectory=matches)
            else:
                if obj["distance_to_camera"]>spec.radius_m: continue
                if spec.intent=="approaching_objects" and obj["motion_state"]!="approaching": continue
                if spec.intent=="moving_objects" and (obj.get("speed_mps") is None or obj["speed_mps"]<=0.1): continue
            result["measurement_age_s"]=(now-measured_at)/1e6
            results.append(result)
        results.sort(key=lambda o:o["distance_to_camera"])
        return results[:1 if spec.intent=="closest_object" else spec.limit]

class LanguageAnswerer:
    def answer(self, spec, objects):
        if spec.intent=="unsupported": return "Unsupported query. Ask about nearby, closest, approaching, moving, object location, or recent history."
        if not objects: return "No matching objects with valid metric depth were recorded."
        lines=[f"{len(objects)} object(s) matched."]
        for o in objects:
            xyz=o["current_position_camera"]
            age=o.get("measurement_age_s",0)
            when=f"last measured {age:.1f}s ago; " if age>0.5 else ""
            lines.append(f'{o["semantic_label"]} #{o["track_id"]}: {when}distance = {o["distance_to_camera"]:.1f} m; camera XYZ = ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) m; state = {o["motion_state"]} (camera-relative); observations = {o["observation_count"]}')
            if spec.intent=="object_history":
                for sample in o.get("matched_trajectory",[]):
                    xyz=sample["xyz_camera"]
                    lines.append(f'  t={sample["timestamp_us"]} us: camera XYZ = ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) m')
        return "\n".join(lines)
