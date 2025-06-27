#!/usr/bin/env python3
"""
Copy trained models and preprocessing objects to submission directory
"""
import os
import shutil
import glob

def copy_weights():
    """Copy all necessary files for submission"""
    source_dir = "development/outputs"
    target_dir = "cmi-submission/weights"
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Files to copy
    patterns = [
        "label_encoder_*.pkl",
        "scaler_fold_*_*.pkl", 
        "model_fold_*_*.pth",
        "best_model_*.pth"  # Optional: single models
    ]
    
    copied_files = []
    
    for pattern in patterns:
        files = glob.glob(os.path.join(source_dir, pattern))
        for file_path in files:
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(file_path, target_path)
            copied_files.append(filename)
            print(f"✅ Copied: {filename}")
    
    if not copied_files:
        print("❌ No files found to copy. Make sure training completed successfully.")
        return False
    
    print(f"\n✅ Successfully copied {len(copied_files)} files to {target_dir}/")
    
    # Verify critical files exist
    critical_files = ["label_encoder_full.pkl", "label_encoder_imu.pkl"]
    missing = []
    
    for variant in ["full", "imu"]:
        # Check for either fold models or single model
        fold_models = glob.glob(os.path.join(target_dir, f"model_fold_*_{variant}.pth"))
        single_model = os.path.join(target_dir, f"best_model_{variant}.pth")
        
        if not fold_models and not os.path.exists(single_model):
            missing.append(f"Models for variant '{variant}'")
            
        # Check for fold scalers
        fold_scalers = glob.glob(os.path.join(target_dir, f"scaler_fold_*_{variant}.pkl"))
        if not fold_scalers:
            missing.append(f"Fold scalers for variant '{variant}'")
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        print("Make sure to train both 'full' and 'imu' variants before submission.")
        return False
    
    print("\n🎉 All required files present for submission!")
    return True

if __name__ == "__main__":
    copy_weights() 