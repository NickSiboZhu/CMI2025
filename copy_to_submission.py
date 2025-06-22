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
    
    # Files to copy
    files_to_copy = [
        "best_model.pth",
        "model_fold_1.pth", 
        "model_fold_2.pth",
        "model_fold_3.pth", 
        "model_fold_4.pth",
        "model_fold_5.pth",
        "label_encoder.pkl",
        "scaler.pkl"
    ]
    
    copied_files = []
    missing_files = []
    
    for filename in files_to_copy:
        source_file = source_dir / filename
        dest_file = dest_dir / filename
        
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            copied_files.append(filename)
            print(f"✅ Copied {filename}")
        else:
            missing_files.append(filename)
            print(f"⚠️  Missing {filename}")
    
    print(f"\n📊 Summary:")
    print(f"  Copied: {len(copied_files)} files")
    print(f"  Missing: {len(missing_files)} files")
    
    if missing_files:
        print(f"\n💡 Missing files: {missing_files}")
        print("   Run training first: cd development && python train.py --epochs 1")
    
    if copied_files:
        print(f"\n🎯 Submission ready! Files in: {dest_dir}")
    
    return len(copied_files) > 0

if __name__ == "__main__":
    success = copy_models_to_submission()
    if success:
        print("\n✅ Ready for Kaggle submission!")
    else:
        print("\n❌ No models to copy. Train models first.") 