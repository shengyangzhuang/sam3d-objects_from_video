from extract_frames import extract_frames
from mask_gen import gen_mask
from plys_gen import gen_plys

def main():
    video_file = "CheetahRunning.mp4"
    target_fps = 2  # the number of frames per second to extract
    frames_dir = "CheetahRunning_rgba"
    mask_dir = "CheetahRunning_masks"
    ply_out_dir = "CheetahRunning_plys"
    sam3d_root = "/home/admin-op/szhuang/sam-3d-objects"
    sam3d_checkpoint_config = "/home/admin-op/szhuang/sam-3d-objects/checkpoints/hf/pipeline.yaml"

    # Step 1: Extract Image Frames to Folder
    original_fps, _ = extract_frames(
        video_path=video_file,
        output_dir=frames_dir,
        target_fps=target_fps,
        image_ext="png",
    )


    # Step 2: Generate Image Mask with SAM3
    if target_fps is None or target_fps >= original_fps:
        frame_interval = 1
    else:
        frame_interval = round(original_fps / target_fps)

    gen_mask(
        video_path=video_file,
        save_dir=mask_dir,
        prompt_text="cheetah",
        frame_interval=frame_interval,  # <- key line
    )


    # Step 3: Generate Mesh with SAM-3D-Objects Frame by Frame
    gen_plys(
        rgba_dir=frames_dir,
        mask_dir=mask_dir,
        ply_out_dir=ply_out_dir,
        sam3d_root=sam3d_root,  # modify
        config_path=sam3d_checkpoint_config,
    )



if __name__ == "__main__":
    main()