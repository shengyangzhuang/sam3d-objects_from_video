import os
import sys
import glob
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask_from_png(path):
    """Return boolean mask (H, W) from a single-channel png."""
    m = Image.open(path).convert("L")
    m_np = np.array(m)
    mask_bool = m_np > 0   # True = object, False = background
    return mask_bool


def gen_plys(
    rgba_dir,
    mask_dir,
    ply_out_dir,
    sam3d_root,
    config_path="checkpoints/hf/pipeline.yaml",
    seed=42,
):
    """
    Generate Gaussian splat PLYs using SAM-3D-Objects inference.

    Parameters
    ----------
    rgba_dir : folder with rgba images (RGBA PNGs)
    mask_dir : folder with masks (same filenames as rgba)
    ply_out_dir : output folder for .ply
    sam3d_root : path to sam-3d-objects repo, MUST contain `notebook/` & `checkpoints/`
    config_path : config path RELATIVE to sam3d_root OR absolute
    seed : random seed
    """

    rgba_dir = Path(rgba_dir)
    mask_dir = Path(mask_dir)
    ply_out_dir = Path(ply_out_dir)
    sam3d_root = Path(sam3d_root).resolve()

    # Basic checks
    if not rgba_dir.is_dir():
        raise RuntimeError(f"[PLYS_GEN] RGBA folder not found: {rgba_dir}")
    if not mask_dir.is_dir():
        raise RuntimeError(f"[PLYS_GEN] Mask folder not found: {mask_dir}")

    ply_out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Ensure `sam3d_objects` package is importable
    # -----------------------------
    # Add the sam3d repo path.
    pkg_parent = None
    candidates = [
        sam3d_root,
        sam3d_root / "src",   # in case the repo uses a src layout
    ]
    for parent in candidates:
        pkg_dir = parent / "sam3d_objects"
        if pkg_dir.is_dir():
            pkg_parent = parent
            break

    if pkg_parent is None:
        raise RuntimeError(
            f"[PLYS_GEN] Could not find 'sam3d_objects' package under {sam3d_root}.\n"
            f"Looked in: {[str(c) for c in candidates]}"
        )

    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))

    # -----------------------------
    # Add inference notebook to sys.path
    # -----------------------------
    notebook_dir = sam3d_root / "notebook"
    if not notebook_dir.is_dir():
        raise RuntimeError(f"[PLYS_GEN] notebook/ not found at {notebook_dir}")

    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))

    # Now safe to import; this will import `sam3d_objects` internally
    from inference import Inference, load_image

    # -----------------------------
    # Resolve config path properly
    # -----------------------------
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = sam3d_root / config_path

    config_path = config_path.resolve()
    print(f"[PLYS_GEN] Loading model from {config_path}")

    inference = Inference(str(config_path), compile=False)

    # -----------------------------
    # Iterate frames
    # -----------------------------
    rgba_paths = sorted(glob.glob(str(rgba_dir / "*.png")))
    print(f"[PLYS_GEN] Found {len(rgba_paths)} frames in {rgba_dir}")

    count = 0
    for i, rgba_path in enumerate(rgba_paths):
        filename = os.path.basename(rgba_path)  # e.g. frame_000000.png
        mask_path = mask_dir / filename

        if not mask_path.exists():
            print(f"[PLYS_GEN] WARNING: mask not found for {rgba_path}, expected {mask_path}")
            continue

        print(f"[PLYS_GEN] Processing frame {i}: image={rgba_path}, mask={mask_path}")

        # RGBA image
        image = load_image(rgba_path)
        # binary mask
        mask = load_mask_from_png(mask_path)

        # run model
        output = inference(image, mask, seed=seed)

        # save PLY
        ply_path = ply_out_dir / f"splat_{i:03d}.ply"
        output["gs"].save_ply(str(ply_path))
        print(f"[PLYS_GEN] Saved {ply_path}")

        count += 1

    print(f"[PLYS_GEN] Done. Total PLYs saved: {count}")
    return count
