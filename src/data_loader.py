import tensorflow as tf
import cv2
import os
from typing import List, Tuple
from .utils import get_vocab_lookups

def load_video(path: str) -> tf.Tensor:
    """
    Loads video, converts to grayscale, crops to mouth region, and normalizes.
    Matches notebook logic: frame[190:236, 80:220, :]
    """
    # If used in tf.data.Dataset map, path is a tensor
    if isinstance(path, tf.Tensor):
        path = bytes.decode(path.numpy())
        
    cap = cv2.VideoCapture(path)
    frames = []
    # Using int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) from notebook loop
    # Note: Sometimes frame count property is unreliable, but sticking to notebook logic
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame = tf.image.rgb_to_grayscale(frame)
        # Crop: Height 190:236 (46px), Width 80:220 (140px)
        frames.append(frame[190:236, 80:220, :])
    cap.release()
    
    # Handle case where video load fails or empty
    if len(frames) == 0:
        # Return zeros matching shape (75, 46, 140, 1) to prevent crash? 
        return tf.zeros((75, 46, 140, 1), dtype=tf.float32)

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    
    # Standardize
    # Add epsilon to std to avoid div by zero? Notebook didn't, but safer.
    # Notebook: tf.cast((frames - mean), tf.float32) / std
    if std == 0:
        return tf.cast(frames, tf.float32)
        
    return tf.cast((frames - mean), tf.float32) / std

def load_alignments(path: str) -> tf.Tensor:
    """
    Parses .align file to get tokenized text converted to numbers.
    """
    if isinstance(path, tf.Tensor):
        path = bytes.decode(path.numpy())
        
    with open(path, 'r') as f:
        lines = f.readlines()
        
    tokens = []
    for line in lines:
        parts = line.split()
        if parts[2] != 'sil': # Skip silence
            tokens = [*tokens, ' ', parts[2]]
            
    char_to_num, _ = get_vocab_lookups()
    # Convert to list of chars then lookup
    # Notebook: char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]
    
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Given a video path, infers the alignment path and loads both.
    """
    if isinstance(path, tf.Tensor):
        path = bytes.decode(path.numpy())
    
    # Path extraction logic from notebook:
    # file_name = path.split('/')[-1].split('.')[0]
    # In pure python we use os.path.split
    file_name = os.path.basename(path).split('.')[0]
    
    # Construct paths
    # Assuming 'data' at root if running from repo root
    # Note: Notebook used: os.path.join('data', 's1', f'{file_name}.mpg')
    # Since we pass 'path' in, we use that for video.
    # We infer alignment path.
    # Notebook: 'data/alignments/s1/bbal6n.align'
    
    # We need to find the root 'data' dir relative to where we assume we are.
    # If path is absolute, we might need to adjust.
    # Let's assume standard structure: data/s1/*.mpg and data/alignments/s1/*.align
    
    alignment_dir = os.path.join('data', 'alignments', 's1')
    alignment_path = os.path.join(alignment_dir, f'{file_name}.align')
    
    frames = load_video(path)
    alignments = load_alignments(alignment_path)
    
    return frames, alignments
