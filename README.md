# Binary Change Detection on EO-SAR Image Pairs

Siamese U-Net++ based multimodal change detection pipeline for disaster assessment using Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery.

This project was developed as part of the **GalaxEye Space AI Research Intern Technical Assignment**.

---

# Project Overview

The objective of this project is to perform **binary pixel-level change detection** between paired pre-event and post-event EO-SAR imagery.

The model predicts:

* `1` → Change
* `0` → No Change

The implemented approach uses:

* **Siamese U-Net++**
* **ResNet34 encoder backbone**
* **EO-SAR multimodal feature fusion**
* **BCE + Dice Loss**
* **PyTorch framework**

---

# Features

* Multimodal EO-SAR change detection
* Siamese encoder architecture
* U-Net++ decoder with dense skip connections
* Binary segmentation masks
* Validation and test evaluation
* Automatic checkpoint saving
* Prediction visualization support
* Threshold-based inference

---

# Repository Structure

```text
GalaxEye-Disaster-Assessment/
│
├── configs/
│   └── config.yaml
│
├── datasets/
│   └── dataset.py
│
├── losses/
│   └── losses.py
│
├── models/
│   └── siamese_unetpp.py
│
├── outputs/
│
├── checkpoints/
│   └── best_model.pth
│
├── train.py
├── eval.py
├── infer.py
├── requirements.txt
├── README.md
└── report.pdf
```

---

# Dataset Structure

Place the dataset in the following structure:

```text
data/
│
├── train/
│   ├── pre/
│   ├── post/
│   └── mask/
│
├── val/
│   ├── pre/
│   ├── post/
│   └── mask/
│
└── test/
    ├── pre/
    ├── post/
    └── mask/
```

---

# Label Remapping

The original dataset contains four classes:

| Original Class | Original Value | Remapped Value |
| -------------- | -------------- | -------------- |
| Background     | 0              | 0              |
| Intact         | 1              | 0              |
| Damaged        | 2              | 1              |
| Destroyed      | 3              | 1              |

This remapping is applied inside `datasets/dataset.py`.

---

# Requirements

## Python Version

```text
Python 3.10
```

---

# Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

# Main Libraries

* PyTorch
* segmentation_models_pytorch
* rasterio
* albumentations
* OpenCV
* NumPy
* tqdm
* scikit-learn

---

# Environment Setup

## Create Conda Environment

```bash
conda create -n galaxeye python=3.10 -y
```

---

## Activate Environment

```bash
conda activate galaxeye
```

---

## Install PyTorch

### CUDA Version

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Run training using:

```bash
python train.py
```

The best model checkpoint will automatically be saved to:

```text
checkpoints/best_model.pth
```

---

# Evaluation

Evaluate the trained model on the test dataset using:

```bash
python eval.py
```

The evaluation script computes:

* IoU
* Precision
* Recall
* F1 Score

Prediction masks will be saved inside:

```text
outputs/
```

---

# Inference

Run inference on individual samples using:

```bash
python infer.py
```

---

# Configuration

All hyperparameters are stored inside:

```text
configs/config.yaml
```

Example configuration:

```yaml
seed: 42

image_size: 256

batch_size: 2

encoder_name: resnet34

learning_rate: 0.00001

weight_decay: 0.0001

epochs: 20

threshold: 0.35
```

---

# Training Strategy

## Architecture

* Siamese U-Net++
* Shared ResNet34 encoder
* EO-SAR feature fusion

---

## Loss Function

Combined:

* BCEWithLogitsLoss
* Dice Loss

Class imbalance handled using weighted BCE loss.

---

## Optimizer

```text
AdamW
```

---

## Scheduler

```text
CosineAnnealingLR
```

---

# Results

## Best Validation Results

| Metric    | Score  |
| --------- | ------ |
| IoU       | 0.2736 |
| Precision | 0.4702 |
| Recall    | 0.3956 |
| F1 Score  | 0.4297 |

---

## Test Results

| Metric    | Score  |
| --------- | ------ |
| IoU       | 0.0331 |
| Precision | 0.0374 |
| Recall    | 0.2237 |
| F1 Score  | 0.0641 |

---

# Challenges Faced

* Severe class imbalance
* SAR speckle noise
* EO-SAR modality differences
* Dataset distribution shift
* Threshold sensitivity

---

# Future Improvements

Possible future directions include:

* Transformer-based change detection
* Better SAR preprocessing
* Attention-based fusion
* Multi-scale feature aggregation
* Self-supervised pretraining
* Balanced patch sampling

---

# Model Weights

Download trained weights from:

```text
[Add Google Drive / HuggingFace Link Here]
```

---

# References

1. Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, 2015.

2. Zhou et al., UNet++: A Nested U-Net Architecture for Medical Image Segmentation, 2018.

3. SNUNet-CD: Dense Siamese Nested U-Net for Change Detection.

4. ChangeFormer: A Transformer-Based Siamese Network for Change Detection.

5. PyTorch Documentation

6. segmentation_models_pytorch Documentation

7. Albumentations Documentation

---

# Author

Lochan R Reddy

BE Artificial Intelligence and Machine Learning
CMR Institute of Technology
Visvesvaraya Technological University (VTU)
