"""
Download and load the 46-epoch checkpoint from Kaggle
"""
import kagglehub
import os

print("Downloading 46-epoch checkpoint from Kaggle...")
print("="*80)

# Download latest version
path = kagglehub.model_download("santhankarnala/46th-epoch-model-checkpoint/keras/default")

print(f"\n✅ Downloaded to: {path}")
print("\nListing files:")
for root, dirs, files in os.walk(path):
    for file in files:
        filepath = os.path.join(root, file)
        size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"  - {file} ({size_mb:.2f} MB)")

print("\n" + "="*80)
print(f"Path to model files: {path}")
