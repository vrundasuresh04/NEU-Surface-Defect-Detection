# NEU Surface Defect Detection & Explainable AI

An AI-based industrial surface defect classification system that uses Deep Learning and Explainable AI to identify and interpret surface defects in steel images.

## Overview

Industrial surface inspection is an important quality-control task in manufacturing. Manual inspection can be time-consuming and may produce inconsistent results.

This project develops an automated image classification system using a ResNet18 deep learning model trained on the NEU Surface Defect Database.

The system can:

- Classify steel surface defects into six categories
- Predict the defect class with confidence
- Provide class probability analysis
- Generate Grad-CAM visualizations for model explainability
- Perform single-image inference
- Perform batch inference on multiple images
- Analyze model performance
- Evaluate model robustness under noisy image conditions
- Provide a Streamlit-based web interface

---

## Defect Classes

The model classifies images into six surface defect categories:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV
- Pillow
- Streamlit
- Grad-CAM
- Jupyter Notebook

---

## System Architecture

```text
                NEU Surface Defect Dataset
                           |
                           v
                  Dataset Exploration
                           |
                           v
                  Data Preprocessing
                           |
                           v
                     ResNet18 Model
                           |
                           v
                    Model Training
                           |
                           v
                    Model Evaluation
                           |
              +------------+------------+
              |                         |
              v                         v
       Image Inference          Grad-CAM Explainability
              |                         |
              v                         v
       Defect Prediction        Visual Explanation
              |
              v
       Performance Analysis
              |
              v
         Batch Inference
              |
              v
         Streamlit Web App
```

---

## Project Workflow

1. Dataset exploration
2. Data preprocessing
3. Model training using ResNet18
4. Model evaluation
5. Single-image inference
6. Grad-CAM explainability
7. Performance analysis
8. Batch inference
9. Streamlit deployment

---

## Model Performance

The trained ResNet18 model was evaluated using standard classification metrics.

| Metric | Score |
|---|---:|
| Accuracy | 100.00% |
| Precision | 100.00% |
| Recall | 100.00% |
| F1 Score | 100.00% |

### Batch Inference

The model was also evaluated on a batch of 300 images.

| Metric | Result |
|---|---:|
| Total Images | 300 |
| Correct Predictions | 300 |
| Incorrect Predictions | 0 |
| Accuracy | 100.00% |

> Note: These results are based on the evaluation and batch-inference experiments performed in this project. They should not be interpreted as guaranteed real-world performance on unseen industrial data.

---

## Explainable AI

Grad-CAM is used to visualize the image regions that contributed most strongly to the model's prediction.

This helps provide visual insight into why the deep learning model classified an image as a particular surface defect.

---

## Application Screenshots

### Streamlit Application

![Streamlit Home](screenshots/streamlit_home.png)

### Prediction Result

![Prediction Result](screenshots/prediction_result.png)

### Grad-CAM Explainability

![Grad-CAM](screenshots/gradcam.png)

### Detection Summary

![Detection Summary](screenshots/detection_summary.png)

### Model Performance

![Model Performance](screenshots/performance.png)

### Batch Inference

![Batch Inference](screenshots/batch_inference.png)

---

## Project Structure

```text
NEU-Surface-Defect-Detection/
|
├── app/
│   └── app.py
|
├── models/
│   └── best_resnet18_neu_det.pth
|
├── notebooks/
│   ├── 01_Dataset_Exploration.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Model_Evaluation.ipynb
│   ├── 05_Explainability_GradCAM.ipynb
│   ├── 06_Inference.ipynb
│   ├── 07_Performance_Analysis.ipynb
│   └── 08_Batch_Inference.ipynb
|
├── screenshots/
│   ├── streamlit_home.png
│   ├── prediction_result.png
│   ├── gradcam.png
│   ├── detection_summary.png
│   ├── performance.png
│   └── batch_inference.png
|
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vrundasuresh04/NEU-Surface-Defect-Detection.git
```

Navigate to the project directory:

```bash
cd NEU-Surface-Defect-Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit Application

Run the application using:

```bash
streamlit run app/app.py
```

The application will open in your browser.

Upload a steel surface image and use the detection interface to obtain:

- Predicted defect class
- Prediction confidence
- Class probabilities
- Grad-CAM visualization
- Detection summary

---

## Supported Defect Categories

The application supports the following six defect categories:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

---

## Key Features

### Deep Learning Classification

Uses a ResNet18 convolutional neural network for automated surface defect classification.

### Confidence Analysis

Displays the predicted class and confidence score along with probabilities for all defect classes.

### Grad-CAM Explainability

Provides visual explanations showing the regions that influenced the model's prediction.

### Batch Inference

Supports prediction on multiple images and reports batch-level accuracy.

### Performance Analysis

Evaluates the model using accuracy, precision, recall, and F1 score.

### Streamlit Interface

Provides an interactive web application for uploading images and viewing prediction results.

---

## Future Improvements

- Test the model on larger and completely unseen datasets
- Improve generalization to different industrial imaging conditions
- Add model monitoring and experiment tracking
- Deploy the application to a cloud platform
- Add REST API support
- Add additional explainability methods
- Optimize inference for production environments

---

## Conclusion

This project demonstrates an end-to-end deep learning workflow for industrial surface defect classification, combining ResNet18-based image classification with Grad-CAM explainability, performance analysis, batch inference, and a Streamlit web interface.

The project provides both automated defect prediction and visual explanations to make the model's decisions easier to interpret.

---

## Author

**Vrunda S**

MCA - Artificial Intelligence & Machine Learning

Bangalore, India