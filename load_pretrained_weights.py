"""
Load Pretrained LipNet Weights and Test Accuracy
This script loads the pretrained weights from rizkiarm/LipNet
and tests them on the GRID corpus to verify ~95% accuracy.
"""

import os
import sys
import tensorflow as tf
import numpy as np
from src.model import build_model
from src.data_loader import load_data
from src.utils import decode_predictions, get_vocab_lookups
from src.config import FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS

def load_pretrained_model(weights_path):
    """
    Load the model and pretrained weights
    """
    print("Building model architecture...")
    model = build_model(input_shape=(FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))
    
    print(f"Loading pretrained weights from {weights_path}...")
    try:
        model.load_weights(weights_path)
        print("✅ Weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print("\nTrying to load with by_name=True, skip_mismatch=True...")
        try:
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print("✅ Weights partially loaded (some layers skipped)")
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
            return None
    
    return model

def calculate_metrics(y_true_list, y_pred_list):
    """
    Calculate Word Error Rate (WER) and Character Error Rate (CER)
    """
    total_wer = 0
    total_cer = 0
    total_samples = len(y_true_list)
    
    for true_text, pred_text in zip(y_true_list, y_pred_list):
        # Word Error Rate
        true_words = true_text.split()
        pred_words = pred_text.split()
        
        # Simple WER calculation (Levenshtein distance would be better)
        wer = sum(1 for tw, pw in zip(true_words, pred_words) if tw != pw)
        wer += abs(len(true_words) - len(pred_words))
        wer = wer / max(len(true_words), 1)
        total_wer += wer
        
        # Character Error Rate
        cer = sum(1 for tc, pc in zip(true_text, pred_text) if tc != pc)
        cer += abs(len(true_text) - len(pred_text))
        cer = cer / max(len(true_text), 1)
        total_cer += cer
    
    avg_wer = (total_wer / total_samples) * 100
    avg_cer = (total_cer / total_samples) * 100
    
    return avg_wer, avg_cer

def test_on_samples(model, data_dir='data/s1', num_samples=10):
    """
    Test the model on sample videos from GRID corpus
    """
    import glob
    
    video_files = glob.glob(os.path.join(data_dir, '*.mpg'))[:num_samples]
    
    if len(video_files) == 0:
        print(f"❌ No video files found in {data_dir}")
        print("Please ensure the GRID dataset is available.")
        return
    
    print(f"\nTesting on {len(video_files)} samples...")
    print("=" * 80)
    
    char_to_num, num_to_char = get_vocab_lookups()
    
    y_true_list = []
    y_pred_list = []
    
    for i, video_path in enumerate(video_files, 1):
        try:
            # Load video + alignment
            frames, alignments = load_data(video_path)
            
            # Get ground truth text
            ground_truth = tf.strings.reduce_join(
                [num_to_char(x) for x in alignments]
            ).numpy().decode('utf-8')
            
            # Prepare input for model
            frames_batch = tf.expand_dims(frames, axis=0)  # Add batch dimension
            
            # Predict
            prediction = model.predict(frames_batch, verbose=0)
            
            # Decode prediction
            decoded = decode_predictions(prediction)
            predicted_text = decoded[0] if decoded else ""
            
            # Store for metrics
            y_true_list.append(ground_truth)
            y_pred_list.append(predicted_text)
            
            # Print result
            print(f"\nSample {i}:")
            print(f"  Ground Truth: '{ground_truth}'")
            print(f"  Prediction:   '{predicted_text}'")
            
            # Check if correct
            if ground_truth.strip() == predicted_text.strip():
                print("  ✅ CORRECT!")
            else:
                print("  ❌ INCORRECT")
            
        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")
            continue
    
    # Calculate overall metrics
    print("\n" + "=" * 80)
    print("OVERALL RESULTS:")
    print("=" * 80)
    
    if y_true_list and y_pred_list:
        avg_wer, avg_cer = calculate_metrics(y_true_list, y_pred_list)
        accuracy = 100 - avg_wer
        
        print(f"Word Error Rate (WER):     {avg_wer:.2f}%")
        print(f"Character Error Rate (CER): {avg_cer:.2f}%")
        print(f"Word Accuracy:             {accuracy:.2f}%")
        print("=" * 80)
        
        # Compare with SOTA
        print("\n📊 Comparison with SOTA LipNet:")
        print(f"   SOTA WER:    4.8%")
        print(f"   Your WER:    {avg_wer:.2f}%")
        print(f"   Difference:  {avg_wer - 4.8:+.2f}%")
        
        if avg_wer < 10:
            print("\n🎉 Excellent! You're very close to SOTA performance!")
        elif avg_wer < 20:
            print("\n✅ Good performance! Some fine-tuning could improve results.")
        else:
            print("\n⚠️ Performance gap detected. May need architecture adjustments.")

def main():
    """
    Main function to load weights and test
    """
    print("=" * 80)
    print("LipNet Pretrained Weights Testing")
    print("=" * 80)
    
    weights_path = 'pretrained_weights/lipnet_weights.h5'
    
    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found: {weights_path}")
        print("Please ensure you've downloaded the weights first.")
        return
    
    # Load model with pretrained weights
    model = load_pretrained_model(weights_path)
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Display model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Test on samples
    print("\n" + "=" * 80)
    test_on_samples(model, num_samples=20)

if __name__ == '__main__':
    main()
