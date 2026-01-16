# MouthMap: Deep Learning Lip Reading System

![Project Status](https://img.shields.io/badge/Status-Research_Prototype-yellow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

**MouthMap** is an end-to-end deep learning model designed to recognize phrases from silent video of lip movements. It implements the **LipNet** architecture, achieving sentence-level sequence prediction using 3D Convolutional Neural Networks (STCNN) and Bidirectional LSTMs with CTC Loss.

## 📂 Project Structure

The codebase has been reorganized for clarity and reproducibility:

```text
MouthMap/
├── src/                 # Core Implementation
│   ├── config.py        # Configuration (Dimensions, Vocab)
│   ├── data_loader.py   # Video processing pipeline
│   ├── model.py         # 3D CNN + BiLSTM Architecture
│   └── utils.py         # CTC Decoding & Vocabulary helpers
├── notebooks/           # Jupyter Notebooks for experimentation
├── archive/             # Archived experiments (Approach-1, etc.)
├── requirements.txt     # Python Dependencies
├── PROJECT_REPORT.md    # Detailed performance & technique analysis
└── README.md            # This file
```

## 🧠 Model Architecture

The system processes video sequences (75 frames) through the following pipeline:

1.  **Spatio-Temporal Feature Extraction**:
    - 3 layers of **3D Convolutions** to capture motion and shape.
    - **3D Max Pooling** for spatial downsampling.
2.  **Sequence Modeling**:
    - **Bidirectional LSTMs** (2x 128 units) capture temporal context forward and backward.
    - **Dropout** (0.5) prevents overfitting.
3.  **Alignment-Free Training**:
    - **CTC Loss** (Connectionist Temporal Classification) allows training without frame-to-phoneme alignment labels.

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data
The project uses the **GRID Corpus**.
- Place video files (`.mpg`) in `data/s1/`
- Place alignment files (`.align`) in `data/alignments/s1/`

### Training (Notebook)
Currently, training is driven via the notebook in `Current--Approach/MouthMap.ipynb`.
*Future work will move the training loop to `src/train.py`.*

## 📊 Performance
- **Dataset**: GRID Corpus (34 speakers, 1000 sentences each).
- **Metric**: CTC Loss.
- **Status**: The model converges from initial loss of ~100 to ~50 within 10 epochs. Pre-trained checkpoints (epoch 40+) demonstrate recognizable sentence predictions.

## 📄 Documentation
For a deep dive into the code quality, state-of-the-art comparisons, and implementation details, please read the **[PROJECT_REPORT.md](PROJECT_REPORT.md)**.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors
- **Santhan Kumar** - *Lead Developer*
