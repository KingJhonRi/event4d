"""Run an offline DSEC geometry, semantics, tracking, memory, query demo."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,csv,json,traceback
import cv2
import numpy as np
import yaml
from event4d.adapters.dsec import DSECAdapter
from event4d.geometry.dsec_gt import DSECGroundTruthGeometryBackend
from event4d.geometry.projection import ground_detection_to_3d
from event4d.semantics.detectors import GroundingDINODetector,TransformersGroundingDINODetector,MockDetector
from event4d.tracking.tracker import AssociationTracker
from event4d.memory.world_memory import WorldMemory
from event4d.visualization.draw import overlay,bev

ROOT=Path(__file__).resolve().parents[1]

def make_detector(kind):
    semantic=yaml.safe_load((ROOT/"configs/semantic.yaml").read_text())
    options={"labels":semantic["labels"],"box_threshold":semantic["box_threshold"]}
    errors=[]
    if kind in ("auto","groundingdino"):
        try: return GroundingDINODetector(ROOT,**options,text_threshold=semantic["text_threshold"]),"GroundingDINO official"
        except Exception as e:
            errors.append(f"official: {type(e).__name__}: {e}")
            if kind=="groundingdino": raise
    if kind in ("auto","transformers"):
        try: return TransformersGroundingDINODetector(**options),"GroundingDINO HF authors"
        except Exception as e:
            errors.append(f"transformers: {type(e).__name__}: {e}")
            if kind=="transformers": raise
    (ROOT/"logs").mkdir(exist_ok=True)
    (ROOT/"logs/detector_fallback.log").write_text("\n".join(errors))
    print("Detector initialization failed; diagnostic mock active. " + "; ".join(errors),flush=True)
    return MockDetector(),"MOCK (synthetic detections)"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sequence",default="thun_00_a")
    ap.add_argument("--duration",type=float,default=20)
    ap.add_argument("--start-time",type=float,default=0,help="seconds after first GT disparity")
    ap.add_argument("--fps",type=float,default=1)
    ap.add_argument("--detector",choices=("auto","groundingdino","transformers","mock"),default="auto")
    ap.add_argument("--geometry",choices=("dsec_gt","esvo2","mock"),default="dsec_gt")
    ap.add_argument("--display",action="store_true")
    ap.add_argument("--save-video",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--output-dir",type=Path,default=ROOT/"outputs/demo")
    args=ap.parse_args()
    if not np.isfinite(args.fps) or args.fps<=0 or not np.isfinite(args.duration) or args.duration<=0 or not np.isfinite(args.start_time) or args.start_time<0:
        ap.error("fps and duration must be positive; start-time must be nonnegative")
    if args.geometry!="dsec_gt": raise NotImplementedError(f"{args.geometry} needs a corresponding geometry source")
    adapter=DSECAdapter(ROOT/"data/dsec"/args.sequence)
    gc=yaml.safe_load((ROOT/"configs/geometry.yaml").read_text())["geometry"]
    mc=yaml.safe_load((ROOT/"configs/memory.yaml").read_text())
    geometry_backend=DSECGroundTruthGeometryBackend(adapter,gc["min_depth_m"],gc["max_depth_m"])
    detector,detector_name=make_detector(args.detector)
    tracker=AssociationTracker(max_age_s=max(3,2/args.fps))
    memory=WorldMemory(mc["history_seconds"],mc["ema_alpha"],mc["radial_velocity_threshold_mps"],mc["stationary_speed_threshold_mps"],mc["fast_speed_threshold_mps"],mc["max_trajectory_samples"])
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    start=int(adapter.disparity_timestamps[0]+args.start_time*1e6)
    end=int(start+args.duration*1e6)
    selected=[]; next_t=start
    for i,t in enumerate(adapter.disparity_timestamps):
        if t>=next_t and t<end:
            selected.append(i);next_t=t+int(1e6/args.fps)
    if not selected: raise ValueError("No disparity frames in requested interval")
    writer=bev_writer=None; rows=[]; frame_count=0; valid_count=0
    with (out/"memory.jsonl").open("w",encoding="utf-8") as memory_file, (out/"observations.jsonl").open("w",encoding="utf-8") as observation_file:
        for di in selected:
            timestamp=int(adapter.disparity_timestamps[di])
            ii=int(np.argmin(abs(adapter.image_timestamps-timestamp)))
            if abs(int(adapter.image_timestamps[ii])-timestamp)>50_000: continue
            image=adapter.image(ii).image
            geometry=geometry_backend.frame(di)
            detections=detector.detect(image)
            observations=[ground_detection_to_3d(d,geometry,timestamp,central_fraction=gc["central_fraction"],min_valid_ratio=gc["min_depth_valid_ratio"]) for d in detections]
            observations=tracker.update(observations,timestamp)
            memory.update(observations,timestamp)
            serialized=[]
            for ob in observations:
                record={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in vars(ob).items()}
                serialized.append(record)
            observation_file.write(json.dumps({"timestamp_us":timestamp,"image_index":ii,"observations":serialized},allow_nan=False)+"\n")
            valid_count+=sum(ob.geometry_valid for ob in observations)
            for ob in observations:
                rows.append({"timestamp_us":timestamp,"track_id":ob.track_id,"label":ob.semantic_label,"confidence":ob.semantic_confidence,"depth_m":ob.depth_m,"depth_valid_ratio":ob.depth_valid_ratio,"x_cam":float(ob.position_camera_xyz[0]) if ob.geometry_valid else None,"y_cam":float(ob.position_camera_xyz[1]) if ob.geometry_valid else None,"z_cam":float(ob.position_camera_xyz[2]) if ob.geometry_valid else None})
            memory_file.write(json.dumps({"timestamp_us":timestamp,"objects":memory.snapshot()},allow_nan=False)+"\n")
            view=overlay(image,observations,memory,semantic_name=detector_name)
            map_view=bev(memory)
            if frame_count==0:
                cv2.imwrite(str(out/"first_overlay.png"),view)
                cv2.imwrite(str(out/"first_bev.png"),map_view)
            if args.save_video:
                if writer is None:
                    writer=cv2.VideoWriter(str(out/"overlay.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),args.fps,(image.shape[1],image.shape[0]))
                    bev_writer=cv2.VideoWriter(str(out/"bev.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),args.fps,(800,800))
                    if not writer.isOpened() or not bev_writer.isOpened(): raise RuntimeError("OpenCV mp4v writer unavailable")
                writer.write(view);bev_writer.write(map_view)
            if args.display:
                cv2.imshow("4D Objects",cv2.resize(view,(960,720)));cv2.waitKey(1)
            frame_count+=1
            print(f"Frame {frame_count}/{len(selected)} | Objects: {len(observations)} | Tracks: {len(memory.objects)} | Valid depth objects: {sum(ob.geometry_valid for ob in observations)}",flush=True)
    if writer: writer.release();bev_writer.release()
    if args.display: cv2.destroyAllWindows()
    with (out/"tracks.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["timestamp_us","track_id","label","confidence","depth_m","depth_valid_ratio","x_cam","y_cam","z_cam"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary={"sequence":args.sequence,"geometry":"DSEC GT image disparity","semantic":detector_name,"has_world_pose":False,"coordinate_frame":"rectified left image camera","frames":frame_count,"detections":len(rows),"valid_depth_observations":valid_count,"tracks":len(memory.objects),"max_observations_per_track":max((s.observation_count for s in memory.objects.values()),default=0),"download_bytes":sum(p.stat().st_size for p in (ROOT/"data/dsec"/args.sequence).glob("*.zip"))}
    summary.update(max_valid_xyz_trajectory=max((len(s.trajectory) for s in memory.objects.values()),default=0),
        events_used_for_inference=False,semantic_input="rectified frame-camera image converted to grayscale",query_backend="local rules and templates",requested_fps=args.fps,
        first_timestamp_us=int(adapter.disparity_timestamps[selected[0]]),last_timestamp_us=timestamp)
    (out/"demo_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
