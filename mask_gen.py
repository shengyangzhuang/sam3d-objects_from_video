# mask_Gen.py

import os
import sam3
import torch
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

from PIL import Image
import numpy as np
import glob
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12


def gen_mask(
    video_path,
    save_dir="xxx_mask",
    prompt_text="cheetah",
    frame_interval=1,   # <- NEW: save every N-th frame
):

    save_dir = str(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[MASK_GEN] Running SAM3 on: {video_path}")
    print(f"[MASK_GEN] Saving results to: {save_dir}")
    print(f"[MASK_GEN] Saving every {frame_interval} frame(s).")

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

    # GPU selection
    gpus_to_use = range(torch.cuda.device_count())
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    # -------------------------
    # Load video frames
    # -------------------------
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
        try:
            video_frames_for_vis.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            video_frames_for_vis.sort()

    # -------------------------
    # Run session
    # -------------------------
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]

    predictor.handle_request(
        request=dict(
            type="reset_session",
            session_id=session_id,
        )
    )

    # Add prompt to frame 0
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt_text,
        )
    )
    out = response["outputs"]

    # Visualize initial result (still on frame 0)
    plt.close("all")
    visualize_formatted_frame_output(
        0,
        video_frames_for_vis,
        outputs_list=[prepare_masks_for_visualization({0: out})],
        titles=["SAM3 Dense Tracking"],
        figsize=(6, 4),
    )

    # Propagate through all frames
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    # -------------------------
    # Save masks (subsampled)
    # -------------------------
    count = 0
    num_frames = len(outputs_per_frame)

    for frame_idx in range(num_frames):
        # only keep every `frame_interval`-th frame
        if frame_idx % frame_interval != 0:
            continue

        img = load_frame(video_frames_for_vis[frame_idx])
        h, w, _ = img.shape

        img_uint8 = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
        outputs = outputs_per_frame[frame_idx]

        combined_mask = np.zeros((h, w), dtype=bool)
        for obj_id, mask in outputs.items():
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            combined_mask |= mask.astype(bool)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[combined_mask, :3] = img_uint8[combined_mask]
        rgba[combined_mask, 3] = 255

        Image.fromarray(rgba).save(
            os.path.join(save_dir, f"frame_{frame_idx:06d}.png")
        )

        count += 1

    # Shut down session
    predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )
    predictor.shutdown()

    print(f"[MASK_GEN] Saved {count} RGBA masked frames (out of {num_frames}).")
    return count
