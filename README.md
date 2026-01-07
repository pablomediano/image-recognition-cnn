<img src="assets/vilnius.png" width="260" align="left"/>

<br clear="left"/>

# Image Recognition with Convolutional Neural Networks

Deep Learning project for **image classification** using a custom dataset  
(cats, dogs and horses) and **Convolutional Neural Networks (CNNs)**.

Developed during my **Erasmus exchange** as part of an Artificial Intelligence course.

---

## Overview

The project implements a complete CNN pipeline:
- Loading a custom image dataset from folders
- Data preprocessing and augmentation
- Model training and validation
- Performance evaluation and visualization of results

---

## Dataset

Images are organized in class-based folders:

```text
dataset/
├── cats/
├── dogs/
└── horses/
```

The dataset is intentionally small and used to demonstrate the full training workflow.

---

## Model

- Convolutional Neural Network built with **TensorFlow / Keras**
- Data augmentation to reduce overfitting
- Early stopping and learning rate reduction
- Multi-class classification (softmax output)

---

## Results

- Training history (accuracy & loss)
- Sample predictions on test images
- Per-class accuracy analysis

All outputs are saved in the `results/` directory. (The trained model is not included due to size constraints.)

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV  
- scikit-learn  
- matplotlib  

---

## Run

```bash
pip install -r requirements.txt
python src/image_custom.py
