#!/bin/bash
# Combine video images into videos for all subdirectories

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    echo "Example: $0 test-rectified-images"
    exit 1
fi

WORKDIR="$1"

# Check if directory exists
if [ ! -d "$WORKDIR" ]; then
    echo "Error: Directory '$WORKDIR' does not exist"
    exit 1
fi

# Loop through all subdirectories
for subdir in "$WORKDIR"/*/; do
    # Remove trailing slash
    subdir="${subdir%/}"

    # Get the subdirectory name
    dirname=$(basename "$subdir")

    # Check if rgb folder exists
    if [ ! -d "$subdir/rgb" ]; then
        echo "Skipping '$dirname' - no rgb folder found"
        continue
    fi

    # Check if there are any jpg files
    jpg_count=$(find "$subdir/rgb" -name "*.jpg" -o -name "*.JPG" | wc -l)
    if [ $jpg_count -eq 0 ]; then
        echo "Skipping '$dirname' - no jpg files found in rgb folder"
        continue
    fi

    echo "Processing '$dirname'..."

    # Create video from images
    ffmpeg -framerate 30 -pattern_type glob -i "$subdir/rgb/*.jpg" \
        -c:v libx264 -pix_fmt yuv420p "$WORKDIR/${dirname}.mp4" -y

    if [ $? -eq 0 ]; then
        echo "✓ Created $WORKDIR/${dirname}.mp4"
    else
        echo "✗ Failed to create video for '$dirname'"
    fi
done

echo ""
echo "Video creation complete!"
echo "Videos saved to: $WORKDIR/"
