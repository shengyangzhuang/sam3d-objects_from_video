# sam3d-objects_from_video

This tool uses SAM3D to continuously reconstruct an object from a video, exporting a sequence of gaussian splatting point clouds.

---

## Example

**Input Video → Segmented Frames**

![CheetahRunning](https://github.com/user-attachments/assets/04240806-12af-4d23-bfff-779bd7e27bbd)  
*Figure 1 — Original cheetah video frames.*

**Generated 3D PLY Sequence**

![CheetahRunning_plys](https://github.com/user-attachments/assets/26d69344-ef0b-4b82-9302-31bb3ea36b1b)  
*Figure 2 — 3D reconstruction exported as sequential `.ply` point clouds.*

---

## Installation

Requires **[SAM3](https://github.com/facebookresearch/sam3)** and **[SAM-3D-Objects](https://github.com/facebookresearch/sam-3d-objects)**.

---

## Usage

Run the pipeline by modifying `main.py`.

## View the generated PLY Sequence

Run `ply_viewer.py`.

