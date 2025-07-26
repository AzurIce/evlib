#!/usr/bin/env python3
"""
Test script to verify the Prophesee ECF decoder fix.
Tests the bit-packed header format and multi-chunk decoding.
"""

import h5py
import numpy as np
from pathlib import Path

# Test file path
test_file = "/Users/tallam/github/tallamjr/origin/evlib/data/prophersee/samples/hdf5/pedestrians.hdf5"

if not Path(test_file).exists():
    print(f"Test file not found: {test_file}")
    exit(1)

print(f"Testing Prophesee ECF decoder with: {test_file}")

# Open the file
with h5py.File(test_file, "r") as f:
    # Navigate to the events dataset
    events_dataset = f["CD/events"]

    # Get dataset info
    total_events = events_dataset.shape[0]
    chunk_shape = events_dataset.chunks[0] if events_dataset.chunks else 16384
    num_chunks = (total_events + chunk_shape - 1) // chunk_shape

    print("\nDataset info:")
    print(f"  Total events: {total_events:,}")
    print(f"  Chunk size: {chunk_shape:,}")
    print(f"  Number of chunks: {num_chunks}")

    # Get filter info
    filters = events_dataset._filters
    print(f"\nFilters applied: {filters}")

    # Try to read raw chunk data
    print("\nAttempting to read raw chunks...")

    # For testing, try to decode first few chunks
    chunks_to_test = min(5, num_chunks)

    for chunk_idx in range(chunks_to_test):
        print(f"\nChunk {chunk_idx}:")

        # Calculate chunk bounds
        start_idx = chunk_idx * chunk_shape
        end_idx = min(start_idx + chunk_shape, total_events)
        chunk_size = end_idx - start_idx

        print(f"  Range: [{start_idx:,} - {end_idx:,})")
        print(f"  Events in chunk: {chunk_size:,}")

        # Try reading the chunk through h5py (will use ECF codec if available)
        try:
            chunk_data = events_dataset[start_idx:end_idx]
            print(f"  ✓ Successfully read chunk (shape: {chunk_data.shape})")

            # Show first few events
            if len(chunk_data) > 0:
                print(
                    f"  First event: x={chunk_data[0]['x']}, y={chunk_data[0]['y']}, "
                    f"t={chunk_data[0]['t']}, p={chunk_data[0]['p']}"
                )
                print(
                    f"  Last event: x={chunk_data[-1]['x']}, y={chunk_data[-1]['y']}, "
                    f"t={chunk_data[-1]['t']}, p={chunk_data[-1]['p']}"
                )
        except Exception as e:
            print(f"  ✗ Failed to read chunk: {e}")

print("\n" + "=" * 60)
print("Now testing with evlib...")

try:
    import evlib

    # Try loading with evlib
    print("\nAttempting to load with evlib.load_events()...")

    # Load just first 100ms to avoid memory issues
    events_df = evlib.load_events(test_file, t_end=0.1)  # First 100ms only

    print(f"✓ Successfully loaded {len(events_df)} events")
    print("\nFirst few events:")
    print(events_df.head())

    print("\nEvent statistics:")
    print(f"  Time range: {events_df['timestamp'].min()} - {events_df['timestamp'].max()}")
    print(f"  X range: {events_df['x'].min()} - {events_df['x'].max()}")
    print(f"  Y range: {events_df['y'].min()} - {events_df['y'].max()}")
    print(f"  Polarities: {events_df['polarity'].unique()}")

except Exception as e:
    print(f"✗ Failed to load with evlib: {e}")
    import traceback

    traceback.print_exc()

print("\nTest complete.")
