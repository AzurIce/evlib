"""
HDF5 Plugin Diagnostic and Setup Helper

This module helps diagnose and fix HDF5 compression plugin issues,
particularly for Prophesee HDF5 files that require BLOSC compression.
"""

import os
import sys


def diagnose_hdf5_plugins():
    """
    Diagnose HDF5 plugin setup and provide fixes.
    """
    print(" HDF5 Plugin Diagnostic")
    print("=" * 50)

    # Check if hdf5plugin is installed
    try:
        import hdf5plugin

        print(" hdf5plugin is installed")
        print(f"   Version: {hdf5plugin.version}")
        print(f"   Plugin path: {hdf5plugin.PLUGIN_PATH}")

        # Check if plugin directory exists and has files
        if os.path.exists(hdf5plugin.PLUGIN_PATH):
            plugins = os.listdir(hdf5plugin.PLUGIN_PATH)
            print(f"   Found {len(plugins)} plugin files:")
            for plugin in sorted(plugins):
                print(f"     - {plugin}")
        else:
            print("   ERROR: Plugin directory doesn't exist!")

    except ImportError:
        print("ERROR: hdf5plugin is not installed")
        print("   Fix: pip install hdf5plugin")
        return False

    # Check h5py version
    try:
        import h5py

        print(" h5py is installed")
        print(f"   Version: {h5py.version.version}")
        print(f"   HDF5 version: {h5py.version.hdf5_version}")
    except ImportError:
        print("ERROR: h5py is not installed")
        print("   Fix: pip install h5py")
        return False

    # Check environment variable
    plugin_path_env = os.environ.get("HDF5_PLUGIN_PATH")
    if plugin_path_env:
        print(f" HDF5_PLUGIN_PATH is set: {plugin_path_env}")
    else:
        print("WARNING:  HDF5_PLUGIN_PATH is not set")
        print(f"   Recommended: export HDF5_PLUGIN_PATH={hdf5plugin.PLUGIN_PATH}")

    return True


def setup_hdf5_plugins():
    """
    Attempt to set up HDF5 plugins for the current session.
    """
    print("\n Setting up HDF5 plugins...")

    try:
        import hdf5plugin

        plugin_path = hdf5plugin.PLUGIN_PATH

        # Set environment variable
        os.environ["HDF5_PLUGIN_PATH"] = plugin_path
        print(f" Set HDF5_PLUGIN_PATH to: {plugin_path}")

        # Register plugins
        if hasattr(hdf5plugin, "register"):
            hdf5plugin.register()
            print(" Registered hdf5plugin filters")

        return True

    except ImportError:
        print("ERROR: Cannot set up plugins - hdf5plugin not installed")
        return False


def test_prophesee_file(file_path: str):
    """
    Test if a Prophesee HDF5 file can be read.
    """
    print(f"\n Testing Prophesee HDF5 file: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return False

    try:
        import h5py

        with h5py.File(file_path, "r") as f:
            print(" File opened successfully")

            if "CD" in f and "events" in f["CD"]:
                cd_events = f["CD"]["events"]
                print(f" Found CD/events dataset with {len(cd_events)} events")
                print(f"   Data type: {cd_events.dtype}")

                # Try to read a small sample
                try:
                    sample = cd_events[:1]
                    print(" Successfully read compressed data!")
                    print(f"   First event: {dict(sample[0])}")
                    return True
                except Exception as read_error:
                    print(f"ERROR: Cannot read data: {read_error}")
                    return False
            else:
                print("ERROR: Not a Prophesee format file (no CD/events)")
                return False

    except Exception as e:
        print(f"ERROR: Cannot open file: {e}")
        return False


def print_solutions():
    """
    Print comprehensive solutions for HDF5 plugin issues.
    """
    print("\n SOLUTIONS")
    print("=" * 50)

    print("1. RECOMMENDED: Set environment variable before starting Python:")
    print("   export HDF5_PLUGIN_PATH=$(python -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)')")
    print("   python your_script.py")

    print("\n2. For Jupyter notebooks, restart kernel and run:")
    print("   import os, hdf5plugin")
    print("   os.environ['HDF5_PLUGIN_PATH'] = hdf5plugin.PLUGIN_PATH")
    print("   # Then import evlib and load data")

    print("\n3. Clean reinstall of HDF5 packages:")
    print("   pip uninstall h5py hdf5plugin")
    print("   pip install --no-cache-dir h5py hdf5plugin")

    print("\n4. For conda environments:")
    print("   conda install -c conda-forge h5py hdf5plugin")

    print("\n5. System-level fix (macOS with Homebrew):")
    print("   brew reinstall hdf5")


if __name__ == "__main__":
    # Run diagnostic
    plugins_ok = diagnose_hdf5_plugins()

    if plugins_ok:
        setup_ok = setup_hdf5_plugins()

        # Test with a Prophesee file if provided
        if len(sys.argv) > 1:
            test_file = sys.argv[1]
            test_prophesee_file(test_file)

    print_solutions()
