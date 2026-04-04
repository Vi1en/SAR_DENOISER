#!/usr/bin/env python3
"""
Complete workflow script for ADMM-PnP-DL SAR image denoising
Demonstrates the full pipeline from dataset preparation to evaluation
"""
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False


def main():
    """Run complete workflow"""
    print("🚀 ADMM-PnP-DL SAR Image Denoising - Complete Workflow")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Setup
    print("\n📋 Step 1: Project Setup")
    if not run_command("python setup.py", "Setting up project structure"):
        print("❌ Setup failed!")
        return False
    
    # Step 2: Test setup
    print("\n🧪 Step 2: Testing Setup")
    if not run_command("python scripts/test_setup.py", "Testing project setup"):
        print("❌ Setup test failed!")
        return False
    
    # Step 3: Download SAMPLE dataset
    print("\n📥 Step 3: Downloading SAMPLE SAR Dataset")
    if not run_command(
        "python scripts/download_sample_dataset.py --visualize --test_loader",
        "Downloading and organizing SAMPLE dataset",
    ):
        print("❌ Dataset download failed!")
        return False
    
    # Step 4: Train denoiser
    print("\n🧠 Step 4: Training Denoiser")
    if not run_command(
        "python scripts/train_sample.py --mode denoiser --epochs 10",
        "Training U-Net denoiser on SAMPLE dataset",
    ):
        print("❌ Denoiser training failed!")
        return False
    
    # Step 5: Train unrolled ADMM
    print("\n🔄 Step 5: Training Unrolled ADMM")
    if not run_command(
        "python scripts/train_sample.py --mode unrolled --epochs 5",
        "Training unrolled ADMM on SAMPLE dataset",
    ):
        print("❌ Unrolled ADMM training failed!")
        return False
    
    # Step 6: Evaluate models
    print("\n📊 Step 6: Evaluating Models")
    if not run_command(
        "python scripts/evaluate_sample.py --methods all",
        "Evaluating all methods on SAMPLE dataset",
    ):
        print("❌ Evaluation failed!")
        return False
    
    # Step 7: Create summary
    print("\n📋 Step 7: Creating Summary")
    create_workflow_summary()
    
    print("\n" + "=" * 70)
    print("🎉 Complete Workflow Finished Successfully!")
    print("=" * 70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📁 Generated Files:")
    print("  - data/sample_sar/ (SAMPLE dataset)")
    print("  - checkpoints_sample_unet/ (trained denoiser)")
    print("  - checkpoints_unrolled_sample_unet/ (trained unrolled ADMM)")
    print("  - results_sample/ (evaluation results)")
    print("  - workflow_summary.txt (workflow summary)")
    
    print("\n🚀 Next Steps:")
    print("1. View results: open results_sample/")
    print("2. Run interactive demo: streamlit run demo/streamlit_app.py")
    print("3. Explore notebooks: jupyter notebook notebooks/")
    
    return True


def create_workflow_summary():
    """Create workflow summary"""
    summary = f"""
ADMM-PnP-DL SAR Image Denoising - Workflow Summary
==================================================

Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Workflow Steps:
1. ✅ Project setup and dependency installation
2. ✅ Setup verification and testing
3. ✅ SAMPLE SAR dataset download and organization
4. ✅ U-Net denoiser training
5. ✅ Unrolled ADMM training
6. ✅ Model evaluation and comparison
7. ✅ Results visualization and summary

Generated Artifacts:
- SAMPLE SAR dataset (data/sample_sar/)
- Trained denoiser (checkpoints_sample_unet/)
- Trained unrolled ADMM (checkpoints_unrolled_sample_unet/)
- Evaluation results (results_sample/)
- Dataset visualizations
- Performance metrics and comparisons

Key Features Demonstrated:
- Real SAR dataset integration
- Deep learning denoiser training
- ADMM-PnP optimization
- Unrolled end-to-end training
- Comprehensive evaluation
- Interactive visualization

The complete ADMM-PnP-DL system is now ready for:
- Research and development
- Educational purposes
- Production deployment
- Further experimentation

For more information, see README.md and PROJECT_SUMMARY.md
"""
    
    with open('workflow_summary.txt', 'w') as f:
        f.write(summary)
    
    print("✅ Workflow summary created: workflow_summary.txt")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


