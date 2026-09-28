"""Upgrade a saved Demo A memory log with derived velocity magnitude fields."""
from pathlib import Path
import json,math,os,sys
path=Path(sys.argv[1])
tmp=path.with_suffix(path.suffix+".tmp")
with path.open(encoding="utf-8") as source,tmp.open("w",encoding="utf-8") as out:
    for line in source:
        row=json.loads(line)
        for state in row["objects"].values():
            velocity=state.get("velocity_camera")
            speed=math.dist(velocity,(0,0,0)) if velocity is not None else None
            state["speed_mps"]=speed
            state["speed_state"]=("unknown" if speed is None else "stationary" if speed<0.1 else "moving" if speed<=0.5 else "fast_moving")
        out.write(json.dumps(row,allow_nan=False)+"\n")
os.replace(tmp,path)
print(path)
