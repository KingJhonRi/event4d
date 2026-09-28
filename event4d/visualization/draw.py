import cv2
import numpy as np

def overlay(image, observations, memory, semantic_name="GroundingDINO"):
    out=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    lines=[]
    for ob in observations:
        x1,y1,x2,y2=map(int,ob.bbox_xyxy)
        color=(0,220,0) if ob.geometry_valid else (0,120,240)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        state=memory.objects[ob.track_id].motion_state
        if ob.geometry_valid:
            px,py,pz=ob.position_camera_xyz
            label=f"{ob.semantic_label} #{ob.track_id} {ob.semantic_confidence:.2f} Z={pz:.1f}m XYZ=({px:.1f},{py:.1f},{pz:.1f}) {state}"
        else:
            label=f"{ob.semantic_label} #{ob.track_id} {ob.semantic_confidence:.2f} depth unknown"
        lines.append((label,color))
        cv2.putText(out,f"#{ob.track_id}",(max(0,x1),max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    if lines:
        panel=out.copy()
        cv2.rectangle(panel,(8,8),(min(out.shape[1]-8,970),min(out.shape[0]-8,35+23*len(lines))),(0,0,0),-1)
        out=cv2.addWeighted(panel,.68,out,.32,0)
        cv2.putText(out,f"{semantic_name} + DSEC GT; raw XYZ / smoothed camera-relative motion",(18,29),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        for i,(label,color) in enumerate(lines):
            cv2.putText(out,label,(18,54+23*i),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1)
    return out

def bev(memory,width=800,height=800,max_range=50):
    canvas=np.full((height,width,3),28,np.uint8)
    center=(width//2,height-60); scale=(height-100)/max_range
    for radius in (3,5,10):
        cv2.circle(canvas,center,int(radius*scale),(65,65,65),1)
        cv2.putText(canvas,f"{radius}m",(center[0]+5,center[1]-int(radius*scale)),cv2.FONT_HERSHEY_SIMPLEX,.45,(160,160,160),1)
    cv2.circle(canvas,center,7,(255,255,255),-1)
    cv2.putText(canvas,"CAMERA",(center[0]+10,center[1]+4),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    latest=memory.timestamp_us if memory.timestamp_us is not None else max((s.last_seen for s in memory.objects.values()),default=0)
    cv2.putText(canvas,"Camera-relative history (no ego-motion compensation)",(20,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(220,220,220),1)
    for state in memory.objects.values():
        if latest-state.last_seen>2_000_000: continue
        if state.last_geometry_seen is None or latest-state.last_geometry_seen>2_000_000: continue
        points=[]
        for sample in state.trajectory:
            xyz=sample["xyz_camera"]
            points.append((int(center[0]+xyz[0]*scale),int(center[1]-xyz[2]*scale)))
        if len(points)>1:
            cv2.polylines(canvas,[np.asarray(points,np.int32)],False,(70,150,255),2)
        if points:
            p=points[-1]
            if 0<=p[0]<width and 0<=p[1]<height:
                cv2.circle(canvas,p,6,(0,220,255),-1)
                cv2.putText(canvas,f"{state.semantic_label} #{state.track_id}",(p[0]+8,p[1]),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    return canvas

def event_accumulation(packet):
    canvas=np.full((packet.height,packet.width,3),127,np.uint8)
    if len(packet.x):
        valid=(packet.x<packet.width)&(packet.y<packet.height)
        x,y,p=packet.x[valid],packet.y[valid],packet.polarity[valid]
        canvas[y[p>0],x[p>0]]=(20,20,255)
        canvas[y[p<=0],x[p<=0]]=(255,40,40)
    return canvas
