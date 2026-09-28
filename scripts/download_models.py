"""Download the official GroundingDINO Swin-T checkpoint."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_demo_data import download

ROOT=Path(__file__).resolve().parents[1]
URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
target=ROOT/"models/groundingdino/groundingdino_swint_ogc.pth"
target.parent.mkdir(parents=True,exist_ok=True)
download(URL,target)
print(target, target.stat().st_size)
