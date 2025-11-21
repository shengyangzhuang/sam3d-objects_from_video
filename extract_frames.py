import cv2
from pathlib import Path


def extract_frames(video_path, output_dir="frames", target_fps=None, image_ext="png"):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")

    # Compute step size
    if target_fps is None or target_fps >= original_fps:
        frame_interval = 1
    else:
        frame_interval = round(original_fps / target_fps)

    print(f"Extracting {target_fps} FPS (saving every {frame_interval} frames)")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = f"frame_{frame_idx:06d}.{image_ext}"
            cv2.imwrite(str(output_dir / filename), frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved_count} frames.")
    return original_fps, saved_count


if __name__ == "__main__":
    video_file = "cam1amelia22.mp4"
    output_folder = "cheetah_frames"

    target_fps = 240  # extract 5 frames per second

    extract_frames(
        video_path=video_file,
        output_dir=output_folder,
        target_fps=target_fps,
        image_ext="png"
    )