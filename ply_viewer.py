import open3d as o3d
import glob
import os
import numpy as np
import cv2

ply_dir = "CheetahRunning_plys"
ply_files = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
print("Found", len(ply_files), "PLY frames")

output_video = "CheetahRunning_plys_play.mp4"

if not ply_files:
    raise RuntimeError("No PLY files found")

width, height = 1280, 720
fps = 5  # adjust to control playback speed

# ---------- 1) ONE VISUALIZER: pick view + reuse for recording ----------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Cheetah playback", width=width, height=height, visible=True)

# Load first frame
pcd = o3d.io.read_point_cloud(ply_files[0])
vis.add_geometry(pcd)
vis.poll_events()
vis.update_renderer()

print("Rotate/zoom to the view you like, then press 'q' to close the interactive phase.")
# This lets you choose the camera/view; when you press 'q', the window stays open,
# but vis.run() returns and we regain control.
vis.run()

# At this point, the camera is exactly where you left it.
# We do NOT convert camera parameters or create a new window.

# ---------- 2) VIDEO WRITER ----------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# ---------- 3) PLAYBACK LOOP WITH THE SAME CAMERA ----------
for i, ply_path in enumerate(ply_files):
    print(f"Rendering frame {i+1}/{len(ply_files)}: {ply_path}")
    new_pcd = o3d.io.read_point_cloud(ply_path)

    # Update the same geometry object in-place
    pcd.points = new_pcd.points
    if new_pcd.has_colors():
        pcd.colors = new_pcd.colors

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # Capture the current view
    img = vis.capture_screen_float_buffer(do_render=True)
    img = (np.asarray(img) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    writer.write(img_bgr)

writer.release()
vis.destroy_window()
print(f"Video saved to {output_video}")
