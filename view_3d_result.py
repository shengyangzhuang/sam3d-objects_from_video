import open3d as o3d
import os
import glob
import time
import numpy as np
import cv2  # <-- NEW

ply_dir = "CheetahRunning_plys"
ply_files = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
print("Found", len(ply_files), "PLY frames")

if not ply_files:
    raise RuntimeError("No .ply files found")

# ---------- 1) PICK CAMERA ----------
print("Step 1: pick camera view on first frame, then press 'q'.")

vis1 = o3d.visualization.Visualizer()
vis1.create_window(window_name="Pick camera (frame 0)", width=1280, height=720)
pcd1 = o3d.io.read_point_cloud(ply_files[0])
vis1.add_geometry(pcd1)
vis1.run()
ctr1 = vis1.get_view_control()
cam_params = ctr1.convert_to_pinhole_camera_parameters()
vis1.destroy_window()

# ---------- 2) PRELOAD ALL FRAMES ----------
print("Preloading all point clouds into memory...")
all_pcds = [o3d.io.read_point_cloud(p) for p in ply_files]
print("Done preloading.")

# ---------- 3) PLAYBACK + RECORD ----------
output_video = "cheetah_3d_reconstruction.mp4"
width, height = 1280, 720
fps_save = 1     # output video FPS (could match target_fps or lower)

# Define codec + writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video, fourcc, fps_save, (width, height))

print("Step 2: playing + recording.")

vis2 = o3d.visualization.Visualizer()
vis2.create_window(window_name="Cheetah playback", width=width, height=height)

pcd2 = all_pcds[0]
vis2.add_geometry(pcd2)

ctr2 = vis2.get_view_control()
ctr2.convert_from_pinhole_camera_parameters(cam_params)

vis2.poll_events()
vis2.update_renderer()

target_fps = 1
frame_interval = 1.0 / target_fps

for i, frame_pcd in enumerate(all_pcds):
    t0 = time.perf_counter()

    # update geometry
    pcd2.points = frame_pcd.points
    if frame_pcd.has_colors():
        pcd2.colors = frame_pcd.colors

    vis2.update_geometry(pcd2)
    vis2.poll_events()
    vis2.update_renderer()

    # ------- CAPTURE FRAME AND SAVE -------
    img = vis2.capture_screen_float_buffer(False)  # False = no UI save
    img = (np.asarray(img) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)     # convert to OpenCV format
    writer.write(img)

    dt = time.perf_counter() - t0
    remaining = frame_interval - dt
    if remaining > 0:
        time.sleep(remaining)

    print(f"Frame {i+1}/{len(all_pcds)} saved")

# Clean up
writer.release()
vis2.destroy_window()

print("Video saved to:", output_video)
