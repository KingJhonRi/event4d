from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from event4d.adapters.dsec import DSECAdapter
from event4d.geometry.dsec_gt import DSECGroundTruthGeometryBackend
from event4d.visualization.draw import event_accumulation

root=Path(__file__).resolve().parents[1]
seq=root/"data/dsec/thun_00_a"
out=root/"outputs/sanity"; out.mkdir(parents=True,exist_ok=True)
adapter=DSECAdapter(seq)
backend=DSECGroundTruthGeometryBackend(adapter)
di=len(adapter.disparities)//2
timestamp=int(adapter.disparity_timestamps[di])
ii=int(np.argmin(abs(adapter.image_timestamps-timestamp)))
image=adapter.image(ii).image
disparity=adapter.disparity(di)
geometry=backend.frame(di)
if image.shape!=disparity.shape: raise ValueError(f"image {image.shape} versus disparity {disparity.shape}")
events=adapter.stereo_events(timestamp,20)
cv2.imwrite(str(out/"left_image.png"),image)
disp_vis=cv2.applyColorMap(cv2.convertScaleAbs(disparity,alpha=2),cv2.COLORMAP_TURBO)
disp_vis[disparity==0]=0
cv2.imwrite(str(out/"disparity.png"),disp_vis)
depth=geometry.depth_map
depth_vis=cv2.applyColorMap(cv2.convertScaleAbs(np.nan_to_num(depth,nan=0),alpha=8),cv2.COLORMAP_TURBO)
depth_vis[~np.isfinite(depth)]=0
cv2.imwrite(str(out/"depth.png"),depth_vis)
cv2.imwrite(str(out/"event_accumulation_left.png"),event_accumulation(events.left))
cv2.imwrite(str(out/"event_accumulation_right.png"),event_accumulation(events.right))
valid=depth[np.isfinite(depth)]
print(f"image resolution: {image.shape[1]}x{image.shape[0]}")
print(f"event count left/right: {len(events.left.x)}/{len(events.right.x)}")
print(f"fx fy cx cy: {geometry.K[0,0]:.3f} {geometry.K[1,1]:.3f} {geometry.K[0,2]:.3f} {geometry.K[1,2]:.3f}")
print(f"baseline: {backend.baseline_m:.5f} m")
print(f"valid disparity/depth: {np.mean(np.isfinite(depth))*100:.2f}%")
print(f"depth min/median/max: {np.min(valid):.2f}/{np.median(valid):.2f}/{np.max(valid):.2f} m")
print(f"saved {out}")
