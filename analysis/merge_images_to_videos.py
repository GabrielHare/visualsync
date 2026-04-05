#!/usr/bin/env python3
"""
Combine image sequences into videos for all subdirectories.

This script processes a directory containing scene subdirectories,
each with a 'rgb' folder containing sequential image frames.
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


# Mapping of codec to file extension
CODEC_EXTENSIONS = {
    'mp4v': '.mp4',
    'avc1': '.mp4',
    'h264': '.mp4',
    'xvid': '.avi',
}



def get_video_dimensions(first_image_path: Path) -> Tuple[int, int]:
    """Read the first image to get video dimensions."""
    img = cv2.imread(first_image_path)
    if img is None:
        raise ValueError(f"Could not read image: {first_image_path}")
    height, width = img.shape[:2]
    return width, height


def create_video_from_images(
    image_files_enum: List[Tuple[int, Path]],
    output_path: str,
    fps: int = 30,
    codec: str = 'mp4v'
) -> bool:
    """
    Create a video from the sequence of images using OpenCV.

    Args:
        image_files_enum: List of image file paths (enumerated and sorted)
        output_path: Output video file path
        fps: Frames per second (default: 30)
        codec: Video codec fourcc code (default: 'mp4v')

    Returns:
        bool: True if successful, False otherwise
    """
    if not image_files_enum:
        return False

    try:
        # Get video dimensions from the first image
        width, height = get_video_dimensions(image_files_enum[0][1])

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"  ✗ Could not open video writer for {output_path}")
            return False

        # Write each frame
        for i, image_file in image_files_enum:
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
    video_codec: str = 'mp4v',
    image_format: str = 'jpg',
    image_subdir: str = 'rgb',
    image_prefix: str = 'frame'
) -> int:
    """
    Process all subdirectories in the workdir, creating videos from image sequences.

    Args:
        workdir: Parent directory containing scene subdirectories
        fps: Frames per second for output videos
        video_codec: Video codec to use
        image_format: Image file extension to process
        image_subdir: Name of the subdirectory containing image frames
        image_prefix: Uniform prefix for image filenames

    Returns:
        int: Number of successfully created videos, < 0 if an error occurred
    """
    workdir = Path(workdir)

    if not workdir.exists() or not workdir.is_dir():
        print(f"Error: Directory '{workdir}' does not exist")
        return -1

    extension = CODEC_EXTENSIONS.get(video_codec)
    if extension is None:
        print(f"Warning: Unsupported video codec: {video_codec}")
        return -1

    success_count = 0

    # Find all subdirectories
    subdirs = [d for d in workdir.iterdir() if d.is_dir()]
    if len(subdirs) == 0:
        print(f"No subdirectories found in {workdir}")
        return -1

    for subdir in sorted(subdirs):
        video_name = subdir.name
        # The expected image frames directory
        images_root = subdir / image_subdir

        # Check if images_dir_path exists
        if not images_root.exists():
            print(f"Skipping '{video_name}' - no images folder found")
            continue

        # Get image files
        image_suffix = f".{image_format}"
        image_files_str = glob.glob(os.path.join(images_root, image_suffix))
        if not image_files_str:
            print(f"Skipping '{video_name}' - no image files found in images folder")
            continue
        print(f"Processing '{video_name}' ({len(image_files_str)} frames)...")

        image_files_enum = [
            (
                int(Path(s).name.lstrip(image_prefix).rstrip(image_suffix)),
                Path(s)
            )
            for s in image_files_str
        ]
        image_files_enum.sort(key=lambda p: p[0])
        if len(image_files_enum) -1 != image_files_enum[-1][0] - image_files_enum[-2][0]:
            print(f"Warning: Missing frames in '{video_name}'")

        # Create output video path with appropriate extension for codec
        output_path = workdir / f"{video_name}{extension}"

        # Create video
        success = create_video_from_images(
            image_files_enum,
            str(output_path),
            fps=fps,
            codec=video_codec
        )

        if success:
            print(f"  ✓ Created {output_path}")
            success_count += 1
        else:
            print(f"  ✗ Failed to create video for '{video_name}'")

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
        help='Parent directory containing scene subdirectories video frame images'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for output videos (default: 30)'
    )

    parser.add_argument(
        '--video_codec',
        type=str,
        default='mp4v',
        choices=['mp4v', 'avc1', 'h264', 'xvid'],
        help='Video codec fourcc code (default: mp4v)'
    )

    parser.add_argument(
        '--image_format',
        type=str,
        default='jpg',
        choices=['jpg', 'png', 'bmp'],
        help='Image format (default: jpg)'
    )

    parser.add_argument(
        '--image_subdir',
        type=str,
        default='rgb',
        help='Images subdirectory name (default: rgb)'
    )

    parser.add_argument(
        '--image_prefix',
        type=str,
        default='frame',
        help='Images name prefix (default: frame)'
    )

    args = parser.parse_args()

    print(f"Processing directory: {args.directory}")
    print(f"Settings: {args.fps} fps, codec: {args.video_codec}")
    print()

    success_count = process_directory(
        args.directory,
        fps=args.fps,
        video_codec=args.video_codec,
        image_format=args.image_format,
        image_subdir=args.image_subdir,
        image_prefix=args.image_prefix
    )

    print()
    print("=" * 60)
    print(f"Video creation complete!")
    if success_count > 0:
        print(f"Successfully created {success_count} video(s)")
        print(f"Videos saved to: {args.directory}/")
    else:
        print(f"Unable to create video(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
