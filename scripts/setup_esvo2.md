# ESVO2 integration

The current machine runs Windows 11 and has no ROS. ESVO2's official tested setup is Ubuntu 20.04, ROS Noetic, OpenCV 4.2, Eigen 3.3.9, and Ceres 1.14.0. Demo A was verified, and the [official ESVO2 repository](https://github.com/NAIL-HNU/ESVO2) is cloned under `third_party/ESVO2`. Build it with catkin on an Ubuntu/ROS host and run the official DSEC launch and repacked bag described in its README. If playback lags, use `rosbag play <bag> --clock -r 0.5`.

The README's repacked bag link currently points to a passworded Baidu Pan share, which does not expose a direct anonymous file URL to the automated downloader. No ESVO2 bag was downloaded on this host. The original official DSEC files are already available locally and may be repacked using the tool referenced by ESVO2 on an Ubuntu/ROS host.

Export timestamped depth (or inverse depth converted at valid pixels), confidence, calibrated `K`, and `T_world_cam` into sequential `.npz` files with those names. `ESVO2GeometryBackend` then produces the same `GeometryFrame` as DSEC GT. Do not interpolate empty regions to fabricate dense depth. This export bridge has not been exercised on this Windows host.

ESVO2 is a visual-inertial system. Verify the required IMU stream and camera/IMU calibration before assuming the downloaded DSEC subset is sufficient. The current subset contains no IMU stream.

The depth map, `K`, `T_world_cam`, and semantic bounding boxes must share a pixel coordinate system and camera frame. DSEC's `camRect0` is the left event camera (640×480); `camRect1` is the left frame camera (1440×1080), which Demo A uses. If depth/pose are exported in the event-camera frame, explicitly transform and project valid 3D samples into the frame camera or move the semantic input to the event view, respecting occlusion and timestamps. Reusing frame-camera boxes directly on event-camera depth would be incorrect. The runner has no live ESVO2 route yet; the adapter only validates external NPZ exports.
