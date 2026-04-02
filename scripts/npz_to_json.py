#!/usr/bin/env python
"""
Convert NPZ file to JSON format.

Usage:
    python npz_to_json.py input.npz [output.json]
"""

import sys
import json
import numpy as np
import argparse


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    # Check for numpy array first
    if isinstance(obj, np.ndarray):
        # Check if it's an object array (contains dicts or other objects)
        if obj.dtype == np.object_:
            # Recursively convert each element
            return [convert_to_json_serializable(item) for item in obj]
        else:
            # Numeric array - convert to list
            return obj.tolist()
    # Check for numpy scalar types
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Check for dict
    elif isinstance(obj, dict):
        return {str(key): convert_to_json_serializable(value) for key, value in obj.items()}
    # Check for list/tuple
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    # Check for bytes
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    # Everything else
    else:
        return obj


def npz_to_dict(npz_file):
    """Load NPZ file and convert to dictionary."""
    data = np.load(npz_file, allow_pickle=True)

    result = {}
    for key in data.files:
        value = data[key]
        result[key] = convert_to_json_serializable(value)

    return result


def main():
    parser = argparse.ArgumentParser(description='Convert NPZ to JSON')
    parser.add_argument('input_npz', help='Input NPZ file')
    parser.add_argument('output_json', nargs='?', help='Output JSON file (optional)')
    parser.add_argument('--pretty', action='store_true', help='Pretty print JSON')
    parser.add_argument('--inspect', action='store_true', help='Just inspect contents without converting')

    args = parser.parse_args()

    # Load NPZ file
    print(f"Loading {args.input_npz}...")
    data = np.load(args.input_npz, allow_pickle=True)

    # Inspect mode
    if args.inspect:
        print(f"\nContents of {args.input_npz}:")
        print("=" * 60)
        for key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}:")
                print(f"    Type: {value.dtype}")
                print(f"    Shape: {value.shape}")
                if value.size < 10:
                    print(f"    Value: {value}")
                else:
                    print(f"    (array too large to display)")
            else:
                print(f"  {key}: {value}")
        print("=" * 60)

        # If it's correspondences, show more details
        if 'correspondences' in data.files:
            corr = data['correspondences']
            if len(corr) > 0:
                print(f"\nFirst correspondence entry:")
                print(f"  Keys: {corr[0].keys() if hasattr(corr[0], 'keys') else 'N/A'}")
                if hasattr(corr[0], 'keys'):
                    for k, v in corr[0].items():
                        if isinstance(v, np.ndarray):
                            print(f"    {k}: shape {v.shape}, dtype {v.dtype}")
                        else:
                            print(f"    {k}: {v}")
        return

    # Convert to dict
    result = npz_to_dict(args.input_npz)

    # Determine output file
    if args.output_json:
        output_file = args.output_json
    else:
        output_file = args.input_npz.replace('.npz', '.json')

    # Write JSON
    print(f"Converting to JSON...")
    with open(output_file, 'w') as f:
        if args.pretty:
            json.dump(result, f, indent=2)
        else:
            json.dump(result, f)

    print(f"✓ Saved to {output_file}")

    # Print summary
    print(f"\nSummary:")
    for key, value in result.items():
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: list of {len(value)} elements")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
