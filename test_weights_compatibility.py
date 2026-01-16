"""
Quick test to check weight compatibility
"""
import tensorflow as tf
from src.model import build_model

print("Building model...")
model = build_model()

print("\nModel architecture:")
model.summary()

print("\n" + "="*80)
print("Attempting to load pretrained weights...")
print("="*80)

try:
    model.load_weights('pretrained_weights/lipnet_weights.h5')
    print("✅ SUCCESS! Weights loaded perfectly!")
    print("\nModel is ready for testing.")
except ValueError as e:
    print(f"❌ Weight shape mismatch: {e}")
    print("\nThis means the architectures don't match exactly.")
    print("Let's try loading with skip_mismatch=True...")
    
    try:
        model.load_weights('pretrained_weights/lipnet_weights.h5', 
                          by_name=True, skip_mismatch=True)
        print("✅ Partial success! Some layers loaded.")
        print("We may need to adjust the architecture.")
    except Exception as e2:
        print(f"❌ Complete failure: {e2}")
        print("\nWe need to inspect the weight file structure.")

print("\n" + "="*80)
print("Weight file inspection:")
print("="*80)

# Inspect the weight file
try:
    import h5py
    with h5py.File('pretrained_weights/lipnet_weights.h5', 'r') as f:
        print(f"\nLayers in weight file:")
        for key in f.keys():
            print(f"  - {key}")
            if isinstance(f[key], h5py.Group):
                for subkey in f[key].keys():
                    print(f"      └─ {subkey}")
except Exception as e:
    print(f"Could not inspect: {e}")
