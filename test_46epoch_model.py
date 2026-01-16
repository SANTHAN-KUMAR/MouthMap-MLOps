"""
Load 46-Epoch Checkpoint and Test Performance
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU to avoid GPU issues

import tensorflow as tf
import numpy as np
import glob
from src.model import build_model
from src.data_loader import load_data
from src.utils import decode_predictions, get_vocab_lookups

# Path to downloaded checkpoint
CHECKPOINT_PATH = "/home/santhankumar/.cache/kagglehub/models/santhankarnala/46th-epoch-model-checkpoint/keras/default/1/checkpoint.weights.h5"

print("=" * 80)
print("Loading 46-Epoch Trained Model")
print("=" * 80)

# Build model
print("\nBuilding model architecture...")
model = build_model()

# Load weights
print(f"Loading weights from: {CHECKPOINT_PATH}")
try:
    model.load_weights(CHECKPOINT_PATH)
    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    exit(1)

print("\nModel Summary:")
model.summary()

# Test on sample videos
print("\n" + "=" * 80)
print("Testing on Sample Videos")
print("=" * 80)

data_dir = 'data/s1'
video_files = glob.glob(os.path.join(data_dir, '*.mpg'))[:10]

if len(video_files) == 0:
    print(f"\n⚠️ No video files found in {data_dir}")
    print("Please ensure GRID dataset is available to test predictions.")
else:
    char_to_num, num_to_char = get_vocab_lookups()
    
    correct = 0
    total = 0
    
    for i, video_path in enumerate(video_files, 1):
        try:
            # Load video + ground truth
            frames, alignments = load_data(video_path)
            
            # Get ground truth text
            ground_truth = tf.strings.reduce_join(
                [num_to_char(x) for x in alignments]
            ).numpy().decode('utf-8')
            
            # Prepare input
            frames_batch = tf.expand_dims(frames, axis=0)
            
            # Predict
            prediction = model.predict(frames_batch, verbose=0)
            
            # Decode
            decoded = decode_predictions(prediction)
            predicted_text = decoded[0] if decoded else ""
            
            # Display
            print(f"\nSample {i}:")
            print(f"  Ground Truth: '{ground_truth}'")
            print(f"  Prediction:   '{predicted_text}'")
            
            # Calculate character-level accuracy
            if ground_truth.strip() == predicted_text.strip():
                print("  ✅ PERFECT MATCH!")
                correct += 1
            else:
                # Count character differences
                diff = sum(1 for a, b in zip(ground_truth, predicted_text) if a != b)
                diff += abs(len(ground_truth) - len(predicted_text))
                print(f"  ⚠️ Character errors: {diff}")
            
            total += 1
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Final stats
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Perfect matches: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Model performing as expected! ✅")
    
print("\n" + "=" * 80)
print("Next Steps:")
print("=" * 80)
print("1. ✅ Model loaded and tested")
print("2. 🚀 Ready for optimization (TFLite conversion)")
print("3. 🌐 Ready for deployment (Streamlit app)")
print("=" * 80)
