"""Single-frame open-vocabulary detection and DSEC GT grounding sanity check."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import yaml
from event4d.adapters.dsec import DSECAdapter
from event4d.geometry.dsec_gt import DSECGroundTruthGeometryBackend
from event4d.geometry.projection import ground_detection_to_3d
from event4d.semantics.detectors import GroundingDINODetector

ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/"configs/semantic.yaml").read_text())
adapter=DSECAdapter(ROOT/"data/dsec/thun_00_a")
geometry=DSECGroundTruthGeometryBackend(adapter).frame(0)
image_index=int(np.argmin(np.abs(adapter.image_timestamps-geometry.timestamp_us)))
if abs(int(adapter.image_timestamps[image_index])-geometry.timestamp_us)>50_000:
    raise ValueError("no synchronized image for first GT disparity")
image=adapter.image(image_index).image
detector=GroundingDINODetector(ROOT,cfg["labels"],cfg["box_threshold"],cfg["text_threshold"])
detections=detector.detect(image)
plain=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
grounded=plain.copy()
plain_lines=[];grounded_lines=[]
for i,det in enumerate(detections,1):
    ob=ground_detection_to_3d(det,geometry,geometry.timestamp_us,track_id=i)
    x1,y1,x2,y2=map(int,det.bbox_xyxy)
    cv2.rectangle(plain,(x1,y1),(x2,y2),(0,230,255),2)
    cv2.rectangle(grounded,(x1,y1),(x2,y2),(0,230,255) if not ob.geometry_valid else (0,240,0),2)
    label=f"{det.semantic_label} {det.semantic_confidence:.2f}"
    xyz=ob.position_camera_xyz
    label3=f"{label} Z={ob.depth_m:.1f}m XYZ=({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f})" if xyz is not None else f"{label} depth unavailable"
    plain_lines.append((f"#{i} {label}",(0,230,255)))
    grounded_lines.append((f"#{i} {label3}",(0,230,255) if xyz is None else (0,240,0)))
    cv2.putText(plain,f"#{i}",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,230,255),2)
    cv2.putText(grounded,f"#{i}",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,230,255) if xyz is None else (0,240,0),2)

def draw_panel(frame,title,lines):
    panel=frame.copy()
    cv2.rectangle(panel,(8,8),(970,35+23*len(lines)),(0,0,0),-1)
    frame=cv2.addWeighted(panel,.68,frame,.32,0)
    cv2.putText(frame,title,(18,29),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    for i,(line,color) in enumerate(lines):
        cv2.putText(frame,line,(18,54+23*i),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1)
    return frame

plain=draw_panel(plain,"GroundingDINO open-vocabulary detections",plain_lines)
grounded=draw_panel(grounded,"GroundingDINO + DSEC GT metric geometry (camera frame)",grounded_lines)
out=ROOT/"outputs/sanity";out.mkdir(parents=True,exist_ok=True)
cv2.imwrite(str(out/"detection.png"),plain)
cv2.imwrite(str(out/"detection_3d.png"),grounded)
print(f"detections={len(detections)} valid_3d={sum(ground_detection_to_3d(d,geometry,geometry.timestamp_us).geometry_valid for d in detections)}")
