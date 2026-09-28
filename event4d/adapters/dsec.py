from __future__ import annotations
from pathlib import Path
import cv2
import h5py
import hdf5plugin  # noqa: F401 registers DSEC's Blosc/ZSTD filter
import numpy as np
import yaml
from event4d.types import EventPacket, StereoEventSlice, ReconstructedFrame

class DSECAdapter:
    def __init__(self, sequence_dir: str | Path):
        self.root = Path(sequence_dir)
        self.calibration = yaml.safe_load((self.root / "cam_to_cam.yaml").read_text())
        self.image_timestamps = np.loadtxt(self.root / "thun_00_a_image_timestamps.txt", dtype=np.int64) if (self.root / "thun_00_a_image_timestamps.txt").exists() else np.loadtxt(next(self.root.glob("*_image_timestamps.txt")), dtype=np.int64)
        self.disparity_timestamps = np.loadtxt(next(self.root.glob("*_disparity_timestamps.txt")), dtype=np.int64)
        self.left_images = sorted((self.root / "images/left/rectified").rglob("*.png"))
        self.right_images = sorted((self.root / "images/right/rectified").rglob("*.png"))
        self.disparities = sorted((self.root / "disparity/image").rglob("*.png"))
        self.event_files = {side: self.root / "events" / side / "events.h5" for side in ("left", "right")}

    def image(self, index: int, camera: str = "left") -> ReconstructedFrame:
        files = self.left_images if camera == "left" else self.right_images
        image = cv2.imread(str(files[index]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(files[index])
        return ReconstructedFrame(int(self.image_timestamps[index]), image, "dsec_image")

    def disparity_index(self, image_index: int) -> int:
        t = self.image_timestamps[image_index]
        idx = int(np.argmin(np.abs(self.disparity_timestamps - t)))
        if abs(int(self.disparity_timestamps[idx]) - int(t)) > 50_000:
            raise ValueError("no GT disparity within 50 ms of image")
        return idx

    def disparity(self, index: int) -> np.ndarray:
        raw = cv2.imread(str(self.disparities[index]), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16:
            raise ValueError(f"expected 16-bit disparity PNG: {self.disparities[index]}")
        return raw.astype(np.float32) / 256.0

    def events(self, camera: str, start_us: int, end_us: int) -> EventPacket:
        if end_us < start_us:
            raise ValueError("event window end must not precede its start")
        path = self.event_files[camera]
        with h5py.File(path) as h5:
            offset = int(h5["t_offset"][()])
            times = h5["events/t"]
            # The millisecond index keeps HDF5 reads local even for long sequences.
            mapping = h5["ms_to_idx"]
            lo_ms = max(0, (start_us - offset) // 1000)
            hi_ms = max(0, (end_us - offset + 999) // 1000)
            lo = int(mapping[min(lo_ms, len(mapping)-1)])
            hi = int(mapping[hi_ms]) if hi_ms < len(mapping) else len(times)
            t = times[lo:hi].astype(np.int64) + offset
            keep = (t >= start_us) & (t < end_us)
            return EventPacket(camera, t[keep], h5["events/x"][lo:hi][keep], h5["events/y"][lo:hi][keep], h5["events/p"][lo:hi][keep], 640, 480)

    def stereo_events(self, end_us: int, window_ms: int = 20) -> StereoEventSlice:
        start = end_us - window_ms * 1000
        return StereoEventSlice(start, end_us, self.events("left", start, end_us), self.events("right", start, end_us))
