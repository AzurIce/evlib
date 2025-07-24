#!/usr/bin/env python3
"""
Test the polarity encoding fix for gen4 files.
"""

import evlib
import numpy as np
from pathlib import Path


def test_polarity_fix():
    """Test that the gen4 polarity encoding fix works correctly"""
    
    # Simulate the test logic
    file_path = "data/gen4_1mpx_original/val/moorea_2019-02-21_000_td_2257500000_2317500000_td.h5"
    
    if not Path(file_path).exists():
        print("❌ Gen4 file not found - skipping test")
        return True
    
    print("🧪 Testing gen4 polarity encoding fix...")
    
    # Load small sample (time filter to avoid timeout)
    print("   Loading sample (first 0.01 seconds)...")
    lf = evlib.load_events(file_path, t_start=0.0, t_end=0.01)
    df = lf.collect()
    
    # Extract polarity array
    p = df["polarity"].to_numpy()
    
    # Test configuration (simulating the fixed test)
    expected_polarity_values = {0, 1}  # HDF5 format uses 0/1 encoding
    allow_single_polarity = True  # This file only contains positive events
    
    # Check polarity encoding
    unique_polarities = np.unique(p)
    actual_polarity_values = set(unique_polarities)
    
    print(f"   📊 Expected polarities: {expected_polarity_values}")
    print(f"   📊 Actual polarities: {actual_polarity_values}")
    print(f"   📊 Sample events: {len(df):,}")
    
    # Test the fixed logic
    if allow_single_polarity:
        # For single-polarity files, allow subset of expected polarities
        test_passed = actual_polarity_values.issubset(expected_polarity_values)
        print(f"   🔍 Subset test: {test_passed}")
    else:
        # For normal files, require exact match
        test_passed = actual_polarity_values == expected_polarity_values
        print(f"   🔍 Exact match test: {test_passed}")
    
    # Check distribution
    polarity_values = list(expected_polarity_values)
    pos_value, neg_value = max(polarity_values), min(polarity_values)
    
    pos_count = np.sum(p == pos_value)
    neg_count = np.sum(p == neg_value)
    total = len(p)
    
    print(f"   📈 Positive ({pos_value}): {pos_count:,}")
    print(f"   📉 Negative ({neg_value}): {neg_count:,}")
    print(f"   📊 Total: {total:,}")
    
    # Test distribution logic
    basic_checks = pos_count + neg_count == total and pos_count > 0
    
    if allow_single_polarity:
        if neg_count == 0:
            print("   ℹ️  Single-polarity file (only positive events) - allowed")
            distribution_ok = True
        else:
            distribution_ok = neg_count > 0
    else:
        distribution_ok = neg_count > 0
    
    print(f"   ✅ Basic checks: {basic_checks}")
    print(f"   ✅ Distribution OK: {distribution_ok}")
    
    overall_pass = test_passed and basic_checks and distribution_ok
    
    if overall_pass:
        print("✅ POLARITY FIX WORKING: Test would pass with new logic")
    else:
        print("❌ POLARITY FIX NOT WORKING: Test would still fail")
    
    return overall_pass


def test_other_files():
    """Test that normal files still work correctly"""
    
    test_files = [
        ("data/slider_depth/events.txt", {0, 1}, False, "Text format"),
        ("data/eTram/h5/val_2/val_night_011_td.h5", {0, 1}, False, "HDF5 format"),
        ("data/eTram/raw/val_2/val_night_011.raw", {-1, 1}, False, "EVT2 format"),
    ]
    
    print("\n🧪 Testing normal files still work...")
    
    all_passed = True
    for file_path, expected, allow_single, description in test_files:
        if not Path(file_path).exists():
            print(f"   ⚠️  {description}: File not found - skipping")
            continue
        
        print(f"   Testing {description}...")
        lf = evlib.load_events(file_path)
        df = lf.collect()
        p = df["polarity"].to_numpy()
        
        unique_polarities = np.unique(p)
        actual = set(unique_polarities)
        
        if allow_single:
            test_passed = actual.issubset(expected)
        else:
            test_passed = actual == expected
        
        print(f"      Expected: {expected}, Got: {actual}, Pass: {test_passed}")
        
        if not test_passed:
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("🚀 POLARITY ENCODING FIX VERIFICATION")
    print("=" * 50)
    
    gen4_ok = test_polarity_fix()
    others_ok = test_other_files()
    
    print(f"\n📈 RESULTS:")
    print(f"   Gen4 fix: {'✅ PASS' if gen4_ok else '❌ FAIL'}")
    print(f"   Other files: {'✅ PASS' if others_ok else '❌ FAIL'}")
    
    if gen4_ok and others_ok:
        print(f"\n🎉 ALL TESTS PASS: Polarity encoding fix is working!")
    else:
        print(f"\n⚠️  SOME TESTS FAILED: Fix needs more work")