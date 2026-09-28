"""Cross-check saved real outputs against source geometry and derived state."""
from pathlib import Path
import argparse
import csv
import json
import sys
from collections import Counter

import cv2
import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from event4d.adapters.dsec import DSECAdapter
from event4d.geometry.dsec_gt import DSECGroundTruthGeometryBackend
from event4d.geometry.projection import ground_detection_to_3d
from event4d.types import Detection

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    output=args.output
    summary=json.loads((output/"demo_summary.json").read_text())
    memories=[json.loads(line) for line in (output/"memory.jsonl").read_text().splitlines()]
    frames=[json.loads(line) for line in (output/"observations.jsonl").read_text().splitlines()]
    with (output/"tracks.csv").open(newline="") as f: rows=list(csv.DictReader(f))
    counts=Counter(int(r["track_id"]) for r in rows)
    xyz_counts=Counter(int(r["track_id"]) for r in rows if r["z_cam"])
    assert summary["frames"]==len(memories)==len(frames)
    assert summary["detections"]==len(rows)==sum(len(f["observations"]) for f in frames)
    assert summary["valid_depth_observations"]==sum(xyz_counts.values())
    assert summary["tracks"]==len(counts)
    assert summary["max_observations_per_track"]==max(counts.values())
    assert summary["max_valid_xyz_trajectory"]==max(xyz_counts.values())
    assert summary["has_world_pose"] is False and summary["events_used_for_inference"] is False
    adapter=DSECAdapter(ROOT/"data/dsec"/summary["sequence"])
    settings=yaml.safe_load((ROOT/"configs/geometry.yaml").read_text())["geometry"]
    backend=DSECGroundTruthGeometryBackend(adapter,settings["min_depth_m"],settings["max_depth_m"])
    geometric_checks=0
    for frame in frames:
        index=int(np.argmin(abs(adapter.disparity_timestamps-frame["timestamp_us"])))
        geometry=backend.frame(index)
        assert geometry.timestamp_us==frame["timestamp_us"]
        assert abs(int(adapter.image_timestamps[frame["image_index"]])-frame["timestamp_us"])<=50_000
        for ob in frame["observations"]:
            det=Detection(np.asarray(ob["bbox_xyxy"]),ob["semantic_label"],ob["semantic_confidence"])
            measured=ground_detection_to_3d(det,geometry,frame["timestamp_us"],central_fraction=settings["central_fraction"],min_valid_ratio=settings["min_depth_valid_ratio"])
            assert measured.geometry_valid==(ob["position_camera_xyz"] is not None)
            if measured.geometry_valid:
                np.testing.assert_allclose(measured.position_camera_xyz,ob["position_camera_xyz"],rtol=1e-6)
                geometric_checks+=1
            assert ob["position_world_xyz"] is None
    velocity_checks=0
    final=memories[-1]["objects"]
    for state in final.values():
        assert state["observation_count"]==counts[state["track_id"]]
        for previous,current in zip(state["trajectory"],state["trajectory"][1:]):
            dt=(current["timestamp_us"]-previous["timestamp_us"])/1e6
            velocity=(np.array(current["xyz_camera"])-previous["xyz_camera"])/dt
            np.testing.assert_allclose(current["velocity_camera"],velocity)
            velocity_checks+=1
        if state["trajectory"]:
            assert state["last_geometry_seen"]==state["trajectory"][-1]["timestamp_us"]
    video_frames={}
    extracts=ROOT/".codex-build/frames"
    extracts.mkdir(parents=True,exist_ok=True)
    for name in ("overlay","bev"):
        cap=cv2.VideoCapture(str(output/f"{name}.mp4"))
        assert cap.isOpened()
        count=0
        while True:
            ok,image=cap.read()
            if not ok: break
            if count in (5,7,12,16,20): cv2.imwrite(str(extracts/f"{name}_{count:02d}.png"),image)
            count+=1
        cap.release()
        assert count==summary["frames"],(name,count)
        video_frames[name]=count
    report={"status":"pass","summary":summary,"recomputed_3d_observations":geometric_checks,
        "checked_velocity_intervals":velocity_checks,"decoded_video_frames":video_frames}
    (output/"audit_verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
