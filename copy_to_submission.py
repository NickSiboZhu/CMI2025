#!/usr/bin/env python3
"""
Helper script to copy trained models from development to submission directory
"""

import os
import shutil
from pathlib import Path

def copy_models_to_submission():
    """Copy trained models and preprocessing objects to submission directory"""
    
    print("🚀 Copying trained models to submission directory...")
    
    # Source and destination paths
    source_dir = Path("development/outputs")
    dest_dir = Path("cmi-submission/weights")
    
    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all .pth & .pkl objects lying in outputs/ (handles _full / _imu suffixes)
    files_to_copy = list(source_dir.glob("*.pth")) + list(source_dir.glob("*.pkl"))

    if not files_to_copy:
        print("⚠️  No weight or preprocessing files found in development/outputs.")
        return False

    for src in files_to_copy:
        dst = dest_dir / src.name
        shutil.copy2(src, dst)
        print(f"✅ Copied {src.name}")

    print(f"\n🎯 Submission ready! Copied {len(files_to_copy)} files → {dest_dir}")
    return True

if __name__ == "__main__":
    success = copy_models_to_submission()
    if success:
        print("\n✅ Ready for Kaggle submission!")
    else:
        print("\n❌ No models to copy. Train models first.") 