#!/usr/bin/env python
# Run CoTracker3 on videos with per-frame masks for VisualSync preprocessing

import os
import sys
import torch
import argparse
import numpy as np
import glob
from PIL import Image

# Add co-tracker to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'co-tracker'))

from cotracker.utils.visualizer import Visualizer, read_video_from_path

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def load_masks_from_dir(mask_dir, num_frames):
    """Load per-frame masks from directory."""
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    if len(mask_paths) == 0:
        print(f"Warning: No mask files found in {mask_dir}")
        return None

    if len(mask_paths) != num_frames:
        print(f"Warning: Found {len(mask_paths)} masks but {num_frames} video frames")

    # Load all masks
    masks = []
    for path in mask_paths[:num_frames]:
        mask = np.array(Image.open(path))
        # If RGB/RGBA, convert to grayscale by taking first channel
        if mask.ndim == 3:
            mask = mask[..., 0]
        masks.append(mask)

    return np.stack(masks)  # Shape: (T, H, W)

def get_unique_instances(masks):
    """Get unique instance IDs from masks (excluding background=0)."""
    unique_ids = np.unique(masks)
    # Remove background (0)
    unique_ids = unique_ids[unique_ids > 0]
    return unique_ids.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        required=True,
        help="path to video file",
    )
    parser.add_argument(
        "--mask_dir",
        default=None,
        help="path to directory containing per-frame mask images",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory to save tracking results (default: same as video directory)",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Frame to sample points from",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Track in both directions",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization videos",
    )

    args = parser.parse_args()

    # Load video
    print(f"Loading video from {args.video_path}...")
    video = read_video_from_path(args.video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()  # B T C H W
    num_frames = video.shape[1]
    print(f"Loaded video with {num_frames} frames, shape: {video.shape}")

    # Load masks if provided
    masks = None
    instance_ids = [None]  # Default: process without mask
    if args.mask_dir and os.path.isdir(args.mask_dir):
        print(f"Loading masks from {args.mask_dir}...")
        masks = load_masks_from_dir(args.mask_dir, num_frames)
        if masks is not None:
            instance_ids = get_unique_instances(masks)
            print(f"Found {len(instance_ids)} dynamic object instances: {instance_ids}")

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default: save in cotracker/ subdirectory next to video
        video_dir = os.path.dirname(args.video_path)
        scene_name = os.path.basename(args.video_path).replace('.mp4', '')
        output_dir = os.path.join(video_dir, scene_name, "cotracker")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load CoTracker3 model
    print("Loading CoTracker3 model...")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)
    print(f"Using device: {DEFAULT_DEVICE}")

    # Process each instance (or process once without mask)
    for instance_id in instance_ids:
        if instance_id is None:
            print(f"\nTracking without mask (grid_size={args.grid_size})...")
            segm_mask = None
            suffix = ""
        else:
            print(f"\nTracking instance {instance_id} (grid_size={args.grid_size})...")
            # Create binary mask for this instance
            instance_mask = (masks == instance_id).astype(np.uint8)
            segm_mask = torch.from_numpy(instance_mask)[None].to(DEFAULT_DEVICE)  # B T H W
            suffix = f"_instance{instance_id}"

        # Run tracking
        pred_tracks, pred_visibility = model(
            video,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
            backward_tracking=args.backward_tracking,
            segm_mask=segm_mask,
        )
        num_tracks = pred_tracks.shape[2]
        print(f"  Computed tracks: {pred_tracks.shape}, visibility: {pred_visibility.shape}")

        # Skip if no tracks found
        # NOTE: This can happen when the masked region is very small
        if num_tracks == 0:
            print(f"  ⚠ Skipping instance {instance_id} - no tracks found (mask may be too small)")
            continue

        # Save results
        output_file = os.path.join(output_dir, f"tracks{suffix}.npz")
        np.savez(
            output_file,
            tracks=pred_tracks.cpu().numpy(),
            visibility=pred_visibility.cpu().numpy(),
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
        )
        print(f"  ✓ Saved to {output_file}")

        # Visualize if requested
        if args.visualize:
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            vis = Visualizer(save_dir=vis_dir, pad_value=120, linewidth=3)
            vis.visualize(
                video,
                pred_tracks,
                pred_visibility,
                query_frame=0 if args.backward_tracking else args.grid_query_frame,
                filename=f"tracks{suffix}",
            )
            print(f"  ✓ Saved visualization to {vis_dir}")

    print(f"\n✓ All tracking complete! Results saved to {output_dir}")
