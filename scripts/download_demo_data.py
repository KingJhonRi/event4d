"""Download one official DSEC training sequence with resumable verified archives."""
from __future__ import annotations

import argparse
import html.parser
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = "https://dsec.ifi.uzh.ch/dsec-datasets/download/"
DEFAULT_SEQUENCE = "thun_00_a"
PARTS = (
    "calibration.zip", "events_left.zip", "events_right.zip",
    "images_rectified_left.zip", "images_rectified_right.zip",
    "disparity_image.zip", "image_timestamps.txt", "disparity_timestamps.txt",
)


class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            url = dict(attrs).get("href", "")
            if url.startswith("https://download.ifi.uzh.ch/rpg/DSEC/train/"):
                self.urls.append(url)


def official_links(sequence: str) -> dict[str, str]:
    parser = Links()
    parser.feed(urllib.request.urlopen(PAGE, timeout=60).read().decode("utf-8"))
    prefix = f"/train/{sequence}/{sequence}_"
    links = {u.rsplit("_", 1)[-1]: u for u in parser.urls if prefix in u}
    # Match full suffix because names such as images_rectified_left contain underscores.
    return {part: next((u for u in parser.urls if u.endswith(f"/{sequence}_{part}") and prefix in u), "") for part in PARTS}


def download(url: str, target: Path) -> None:
    if target.is_file() and target.stat().st_size:
        if target.suffix != ".zip" or zipfile.is_zipfile(target):
            print(f"skip {target.name} ({target.stat().st_size:,} bytes)", flush=True)
            return
    tmp = target.with_name(target.name + ".part")
    offset = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(req, timeout=90) as response:
        status = response.status
        if offset and status != 206:
            offset = 0
        expected = response.headers.get("Content-Length")
        expected = int(expected) + offset if expected else None
        with tmp.open("ab" if offset else "wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                if out.tell() // (50 * 1024 * 1024) != (out.tell() - len(block)) // (50 * 1024 * 1024):
                    print(f"  {target.name}: {out.tell()/1e6:.0f} MB", flush=True)
    if expected is not None and tmp.stat().st_size != expected:
        raise IOError(f"incomplete {target.name}: {tmp.stat().st_size} != {expected}")
    if target.suffix == ".zip" and not zipfile.is_zipfile(tmp):
        raise IOError(f"invalid zip: {tmp}")
    tmp.replace(target)


def unpack(archive: Path, destination: Path) -> None:
    marker = destination / (archive.stem + ".unpacked")
    if marker.exists():
        return
    with zipfile.ZipFile(archive) as zf:
        for entry in zf.infolist():
            resolved = (destination / entry.filename).resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise ValueError(f"unsafe archive member {entry.filename}")
        zf.extractall(destination)
    marker.write_text(str(archive.stat().st_size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "dsec")
    ap.add_argument("--minimal", action="store_true", help="Only calibration, left image, and disparity")
    args = ap.parse_args()
    destination = args.data_root / args.sequence
    destination.mkdir(parents=True, exist_ok=True)
    links = official_links(args.sequence)
    parts = ("calibration.zip", "images_rectified_left.zip", "disparity_image.zip", "image_timestamps.txt", "disparity_timestamps.txt") if args.minimal else PARTS
    free = shutil.disk_usage(destination).free
    if free < 5_000_000_000:
        raise RuntimeError(f"Only {free/1e9:.1f} GB free; at least 5 GB required")
    for part in parts:
        url = links.get(part)
        if not url:
            raise RuntimeError(f"{part} not found on {PAGE}")
        target = destination / f"{args.sequence}_{part}"
        print(f"fetch {url}", flush=True)
        download(url, target)
        if target.suffix == ".zip":
            subdirs = {
                "events_left.zip": "events/left", "events_right.zip": "events/right",
                "images_rectified_left.zip": "images/left/rectified",
                "images_rectified_right.zip": "images/right/rectified",
                "disparity_image.zip": "disparity/image",
            }
            unpack(target, destination / subdirs.get(part, ""))
    required = [destination / "cam_to_cam.yaml", destination / f"{args.sequence}_image_timestamps.txt",
                destination / f"{args.sequence}_disparity_timestamps.txt"]
    required += [destination / "images/left/rectified/000000.png", destination / "disparity/image/000000.png"]
    if not args.minimal:
        required += [destination / "events/left/events.h5", destination / "events/right/events.h5",
                     destination / "images/right/rectified/000000.png"]
    missing = [str(p) for p in required if not p.is_file() or not p.stat().st_size]
    if missing:
        raise RuntimeError("Missing extracted DSEC files: " + ", ".join(missing))
    print(f"Downloaded and unpacked official DSEC sequence at {destination}", flush=True)


if __name__ == "__main__":
    main()
