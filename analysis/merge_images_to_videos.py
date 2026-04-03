#!/usr/bin/env python3
"""
Combine image sequences into videos for all subdirectories.

This script processes a directory containing scene subdirectories,
each with an 'rgb' folder containing sequential image frames.
It creates MP4 videos from these image sequences using OpenCV.

Usage:
    python merge_images_to_videos.py <directory>

Example:
    python merge_images_to_videos.py test-rectified-images
"""

from typing import *

import os
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np


def get_image_files(rgb_dir: str) -> List[str]:
    """Get sorted list of image files from directory."""
    # Support common image extensions
    patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    image_files = []

    for pattern in patterns:
        image_files.extend(glob.glob(os.path.join(rgb_dir, pattern)))

    return sorted(image_files)


def get_video_dimensions(first_image_path: str) -> Tuple[int, int]:
    """Read first image to get video dimensions."""
    img = cv2.imread(first_image_path)
    if img is None:
        raise ValueError(f"Could not read image: {first_image_path}")
    height, width = img.shape[:2]
    return width, height


def create_video_from_images(
    image_files: List[str],
    output_path: str,
    fps: int = 30,
    codec: str = 'mp4v'
) -> bool:
    """
    Create video from sequence of images using OpenCV.

    Args:
        image_files: List of image file paths (sorted)
        output_path: Output video file path
        fps: Frames per second (default: 30)
        codec: Video codec fourcc code (default: 'mp4v')

    Returns:
        bool: True if successful, False otherwise
    """
    if not image_files:
        return False

    try:
        # Get video dimensions from first image
        width, height = get_video_dimensions(image_files[0])

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"  ✗ Could not open video writer for {output_path}")
            return False

        # Write each frame
        for i, image_file in enumerate(image_files):
            img = cv2.imread(image_file)

            if img is None:
                print(f"  ⚠ Warning: Could not read frame {i}: {image_file}")
                continue

            # Ensure consistent dimensions
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height))

            out.write(img)

        out.release()
        return True

    except Exception as e:
        print(f"  ✗ Error creating video: {e}")
        return False


def process_directory(
    workdir: Union[str, Path],
    fps: int = 30,
    codec: str = 'mp4v'
) -> int:
    """
    Process all subdirectories in workdir, creating videos from image sequences.

    Args:
        workdir: Parent directory containing scene subdirectories
        fps: Frames per second for output videos
        codec: Video codec to use

    Returns:
        int: Number of successfully created videos
    """
    workdir = Path(workdir)

    if not workdir.exists() or not workdir.is_dir():
        print(f"Error: Directory '{workdir}' does not exist")
        return 0

    success_count = 0

    # Find all subdirectories
    subdirs = [d for d in workdir.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"No subdirectories found in {workdir}")
        return 0

    for subdir in sorted(subdirs):
        dirname = subdir.name
        rgb_dir = subdir / 'rgb'

        # Check if rgb folder exists
        if not rgb_dir.exists():
            print(f"Skipping '{dirname}' - no rgb folder found")
            continue

        # Get image files
        image_files = get_image_files(str(rgb_dir))

        if not image_files:
            print(f"Skipping '{dirname}' - no image files found in rgb folder")
            continue

        print(f"Processing '{dirname}' ({len(image_files)} frames)...")

        # Create output video path
        output_path = workdir / f"{dirname}.mp4"

        # Create video
        success = create_video_from_images(
            image_files,
            str(output_path),
            fps=fps,
            codec=codec
        )

        if success:
            print(f"  ✓ Created {output_path}")
            success_count += 1
        else:
            print(f"  ✗ Failed to create video for '{dirname}'")

    return success_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert image sequences in rgb/ subdirectories into videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s test-rectified-images
  %(prog)s /path/to/data --fps 60
  %(prog)s ./scenes --codec avc1
        """
    )

    parser.add_argument(
        'directory',
        help='Parent directory containing scene subdirectories with rgb/ folders'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for output videos (default: 30)'
    )

    parser.add_argument(
        '--codec',
        type=str,
        default='mp4v',
        choices=['mp4v', 'avc1', 'h264', 'xvid'],
        help='Video codec fourcc code (default: mp4v)'
    )

    args = parser.parse_args()

    print(f"Processing directory: {args.directory}")
    print(f"Settings: {args.fps} fps, codec: {args.codec}")
    print()

    success_count = process_directory(args.directory, fps=args.fps, codec=args.codec)

    print()
    print("=" * 60)
    print(f"Video creation complete!")
    print(f"Successfully created {success_count} video(s)")
    print(f"Videos saved to: {args.directory}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
