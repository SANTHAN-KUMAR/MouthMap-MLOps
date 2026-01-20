# MouthMap Project Audit & Flaws Analysis

## 1. Project Overview (Pin-to-Pin Details)
**Project Name**: MouthMap  
**Objective**: End-to-end Lip Reading (Video-to-Text) using Deep Learning.  
**Core Logic**: 3D CNNs + Bi-LSTMs + CTC Loss.

### A. Technology Stack
*   **Framework**: TensorFlow / Keras (2.x)
*   **Data Processing**: OpenCV (`cv2`), `imageio`, `matplotlib`
*   **Input Data Pipeline**: `tf.data` API (optimization via prefetch/autotune)
*   **Source Management**: `gdown` (Google Drive extraction)

### B. Data Source & Preprocessing
1.  **Dataset**:
    *   **Corpus**: GRID Audiovisual Corpus.
    *   **Subject**: **Speaker 1 (`s1`) ONLY**.
    *   **Files**: `.mpg` (Video) and `.align` (Word Alignments).
2.  **Preprocessing Pipeline (`load_video`)**:
    *   **Frame Capture**: Iterates through video frames sequentially.
    *   **Grayscale Conversion**: `tf.image.rgb_to_grayscale`.
    *   **Region of Interest (ROI)**: **Hardcoded Cropping**.
        *   `frame[190:236, 80:220, :]`
        *   Extracts a fixed window of **Height: 46px**, **Width: 140px**.
    *   **Normalization**: Z-Score Normalization (Standardization).
        *   Formula: `(frames - mean) / std`
        *   **Scope**: Calculated per individual video instance.
3.  **Vocabulary**:
    *   **Type**: Character-level tokens.
    *   **Dictionary**: `[a-z], ', ?, !, [0-9], space`.
    *   **Size**: 40 chars + 1 OOV/Blank = **41 classes**.

### C. Model Architecture (Layer-by-Layer)
The model accepts inputs of shape `(Batch, 75, 46, 140, 1)`:
1.  **Spatiotemporal Feature Extraction (3D CNNs)**
    *   `Conv3D` (128 filters, 3x3x3 kernel) → `ReLU` → `MaxPool3D` (1, 2, 2)
    *   `Conv3D` (256 filters, 3x3x3 kernel) → `ReLU` → `MaxPool3D` (1, 2, 2)
    *   `Conv3D` (75 filters, 3x3x3 kernel) → `ReLU` → `MaxPool3D` (1, 2, 2)
    *   *Purpose*: Extracts visual features of lip movements across time and space.
2.  **Dimensionality Reduction**
    *   `Reshape`: Flattens spatial dimensions.
    *   *Transformation*: `(75, 5, 17, 75)` → `(75, 6375)`.
3.  **Sequence Modeling (RNNs)**
    *   `Bidirectional LSTM` (128 units) → `Dropout` (0.5)
    *   `Bidirectional LSTM` (128 units) → `Dropout` (0.5)
    *   *Purpose*: Models the temporal dependency of the character sequence.
4.  **Classification Head**
    *   `Dense` (41 units, `he_normal` init).
    *   `Softmax` Activation.
    *   *Purpose*: Probability distribution over the vocabulary for each timestep.

---

## 2. Extensive Flaws Analysis

### 🚨 Critical Architecture & Methodology Flaws

#### 1. Static, Hardcoded Face Cropping
*   **The Flaw**: The code uses a fixed slice `frame[190:236, 80:220, :]` to identify the mouth.
*   **Impact**:
    *   **Zero Robustness**: If the camera position shifts by even 10 pixels, the model sees a chin or a nose instead of lips.
    *   **No Head Movement Tolerance**: If the speaker tilts their head or leans forward/backward, the data becomes garbage.
    *   **Scale Invariance**: Assumes every video has the exact same zoom level and face size.
*   **Production Fix**: Implementing a Face Detection (Haar Cascades/MTCNN) and Facial Landmark Detection (Dlib/MediaPipe) pipeline is mandatory to dynamically locate and warp/align the mouth region.

#### 2. Single Speaker Overfitting ("The S1 Bias")
*   **The Flaw**: The dataset loader explicitly globs only `data/s1/*.mpg`.
*   **Impact**:
    *   **Biometric Overfit**: The model is likely learning the specific biometric features of Speaker 1 (mustache, teeth shape, jawline) rather than generalized phoneme-viseme mappings.
    *   **Generalization Failure**: The model will almost certainly fail (0% accuracy) on any other human being.
*   **Production Fix**: Train on multiple speakers (GRID has 34 speakers) to encourage learning general lip motions.

#### 3. Data Leaks & Splitting Strategy
*   **The Flaw**: `data.take(450)` for Train and `data.skip(450)` for Test.
*   **Impact**:
    *   **Representative Bias**: Depending on the shuffle buffer size (currently 500), this simple split might cluster similar sentences together.
    *   **No Validation Set**: There is no "Hold-out" set. Hyperparameters are likely tuned to maximize Test performance specifically, leading to inflated confidence scores that won't match real-world usage.

### ⚠️ Technical & Implementation Limitations

#### 4. Per-Instance Normalization Instability
*   **The Flaw**: `mean` and `std` are calculated on `frames` inside the `load_video` function (local normalization).
*   **Impact**:
    *   **Lighting Sensitivity**: A video with poor lighting will have a different numerical distribution than a bright one.
    *   **Noise Amplification**: In very dark videos (low std dev), dividing by a small number amplifies sensor noise.
*   **Production Fix**: Compute global dataset statistics or use Batch Normalization layers dynamically.

#### 5. Limited "Toy" Vocabulary
*   **The Flaw**: The current vocabulary assumes only lowercase chars and digits.
*   **Impact**:
    *   **Context Blindness**: It has no concept of capital letters, complex punctuation, or phonetic nuances.
    *   **Grammar Constraints**: The GRID corpus follows a strict, unnatural grammar ("Command Color Preposition Letter Digit Adverb"). The model is likely memorizing this structure rather than learning English.

#### 6. Missing Quantitative Metrics
*   **The Flaw**: The notebook relies on visual inspection (`plt.imshow`) and raw text decoding for validation.
*   **Impact**:
    *   **No Scientific Benchmark**: There is no implementation of **WER (Word Error Rate)** or **CER (Character Error Rate)** to objectively measure performance.
    *   **Greedy Decoding**: The prediction uses simple `argmax` (Greedy Search), which is suboptimal compared to Beam Search with a Language Model (which corrects spelling errors using a dictionary).

### 📉 Performance Bottlenecks

#### 7. Inefficient 3D CNN Scaling
*   **The Flaw**: Using `Conv3D` is computationally expensive compared to modern "2.5D" approaches (ResNet for spatial features -> temporal mixing).
*   **Impact**: Slower inference times and massive VRAM usage during training, limiting batch size.

#### 8. Data Input IO
*   **The Flaw**: Using `gdown` inside the notebook for every session.
*   **Impact**: Wastes bandwidth and startup time. Data should be persisted to a local volume or artifact storage in a real MLOps workflow.
