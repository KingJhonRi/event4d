from pathlib import Path
import platform,shutil,subprocess,sys
root=Path(__file__).resolve().parents[1]
print("OS:",platform.platform())
print("Python:",sys.version.split()[0],sys.executable)
try:
 import torch
 print("PyTorch:",torch.__version__,"CUDA available:",torch.cuda.is_available(),"CUDA version:",torch.version.cuda)
 print("GPU:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU mode)")
except ImportError: print("PyTorch: unavailable")
try:
 import cv2
 print("OpenCV:",cv2.__version__)
except ImportError: print("OpenCV: unavailable")
print("ROS:",shutil.which("roslaunch") or "unavailable")
print("free disk space:",round(shutil.disk_usage(root).free/1e9,2),"GB")
