# Research: Logic of VLMs & LLMs for Lip Reading

## 1. Can we use Gemini / GPT-4o directly?
**Short Answer:** Generally, **No** (not for high-accuracy lip reading), but they are getting there.

### The Problem: Temporal Granularity
*   **How VLMs see video:** Models like Gemini 1.5 Pro or GPT-4 Video often process video by sampling **keyframes** (e.g., 1 frame per second).
*   **What Lip Reading needs:** Human speech produces phonemes at a rate of 10-15 per second. Visemes (visual phonemes) are extremely fast. To capture the difference between a "B" and a "P" or "M", the model needs **25 to 50 frames per second (FPS)** without skipping.
*   **Result:** A general VLM might understand "The person is talking" or "The person looks angry", but will likely miss the specific nuances of lip shape changes required to transcribe the exact sentence "Bin blue at l two now".

## 2. "LLM-like" Models for Lip Reading (State-of-the-Art)
You are correct that the industry has moved away from simple CNN+LSTMs (like your `MouthMap` project). The modern "LLMs of Lip Reading" are **Transformers**.

### A. AV-HuBERT (Audio-Visual Hidden Unit BERT)
*   **Concept**: Just like BERT learns language by hiding words and guessing them, AV-HuBERT learns lip-reading by hiding parts of the video/audio and guessing the missing features.
*   **Why it's better**: It is pre-trained on thousands of hours of video (LRS3, VoxCeleb) and then fine-tuned. It beats human performance.

### B. VATLM (Visual-Audio-Text Language Model)
*   **Concept**: This attempts to unify everything into one "LLM". It takes pixels (video), waves (audio), and text as inputs into a single Transformer.
*   **Relevance**: This is exactly "using an LLM" logic. It shows that massive pre-training on video+text works better than training a small model from scratch on GRID.

### C. Raven / Conformer
*   **Architecture**: Replaces the LSTM in your project with a **Transformer Encoder** (Self-Attention).
*   **Benefit**: LSTMs forget the start of a long sentence by the time they reach the end. Transformers look at the whole sentence at once (Global Attention).

## 3. Comparison: Your Project vs. Modern VLM Approach

| Feature | Your Current Method (CNN + LSTM) | Modern VLM / Transformer Method |
| :--- | :--- | :--- |
| **Architecture** | **Sequential**: Frame $\rightarrow$ CNN $\rightarrow$ RNN $\rightarrow$ Text | **Attention**: All Frames $\leftrightarrow$ All Text |
| **Data Requirement** | Needs aligned labels (Supervised) | Can unsupervised pre-train on YouTube videos (Self-Supervised) |
| **Accuracy (LRS3)** | ~40-50% WER (Word Error Rate) | **< 30% WER** (Super-human level) |
| **Compute Cost** | Low (Trainable on 1 GPU) | **Very High** (Requires clusters for pre-training) |
| **Flexibility** | Rigid (Fixed crop, fixed speaker) | Robust (Handles head pose, lighting, multiple speakers) |

## 4. Recommendation for MouthMap
If you want to modernize this project without needing Google-scale compute:

1.  **Swap LSTM for Transformer**: Replace the `Bidirectional(LSTM)` layers with `MultiHeadAttention` layers (Transformer Block). The Keras `TransformerEncoder` is a drop-in upgrade.
2.  **Use Pre-trained Embeddings**: Instead of raw pixels, feed the video frames into a pre-trained face wrapper (like **MediaPipe** or **Dlib**) to get consistent visual landmarks.
3.  **Don't use Gemini API**: It's too slow/expensive for real-time lip reading. Build a small "Gemini-style" transformer locally.
