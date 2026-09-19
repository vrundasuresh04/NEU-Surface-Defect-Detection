# NEU Surface Defect Detection & Explainable AI

An AI-based industrial surface defect classification system
using Deep Learning and Explainable AI.

## Overview

This project uses a ResNet18 convolutional neural network
to classify industrial steel surface defects into six categories:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

The system also integrates Grad-CAM to explain which regions
of an image influenced the model's prediction.

## Technologies

- Python
- PyTorch
- ResNet18
- OpenCV
- Grad-CAM
- Scikit-learn
- Streamlit
- NumPy
- Pandas
- Matplotlib

## Features

- Six-class defect classification
- Confidence score analysis
- Class probability visualization
- Batch inference
- Robustness testing
- Grad-CAM explainability
- Streamlit-based inference application