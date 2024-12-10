# Road Rage Detection System - Design Documentation

## Overview
The **Road Rage Detection System** is a machine learning-based solution that identifies road rage incidents using video data. The system uses deep learning techniques, specifically a **3D Convolutional Neural Network (3D CNN)**, for analyzing temporal video frames. The project consists of two main models:

1. **Violence Detection Model**: A pretrained model trained on generic violence detection datasets.
2. **Road Rage Detection Model**: A fine-tuned model using **transfer learning** from the Violence Detection Model, tailored to detect road rage behaviors.

The system processes video data in real-time and classifies incidents as either "Normal" or "Road Rage".

## Architecture

The architecture is divided into two major components:
1. **Pretrained Violence Detection Model**
2. **Road Rage Detection Model (Fine-tuned)**

### 1. **Pretrained Violence Detection Model**
   - **Purpose**: Classifies video frames into "violent" or "non-violent" categories.
   - **Dataset**: A generic violence detection dataset (e.g., Hockey Fight, Crowd Violence dataset).
   - **Model Type**: **3D Convolutional Neural Network (3D CNN)**
     - **Input**: Video frames of shape `(frames, height, width, channels)`.
     - **Output**: Binary classification (Violent vs Non-Violent).

### 2. **Road Rage Detection Model**
   - **Purpose**: Fine-tunes the pretrained violence detection model to detect road rage behavior.
   - **Dataset**: A custom dataset consisting of road rage video data (e.g., captured from traffic cameras, YouTube).
   - **Model Type**: Transfer learning from the **Violence Detection Model**, where the final layers are replaced for road rage classification.
     - **Input**: Video frames processed into batches.
     - **Output**: Binary classification (Normal vs Road Rage).

### Workflow

1. **Data Preprocessing**:
   - Raw video data is processed and converted into frames, and frames are resized to a uniform size (e.g., `30 x 150 x 150` pixels).
   - The frames are stored as `.npy` files for efficient loading during training and inference.

2. **Training**:
   - **Violence Detection**: Train a 3D CNN model on a violence dataset, saving the model for later use in road rage detection.
   - **Road Rage Detection**: Fine-tune the pretrained violence model by adding a classification head suited for road rage classification. Training occurs on the road rage dataset.

3. **Evaluation**:
   - Evaluate both models using a validation and test dataset. Compute metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
   
4. **Real-Time Detection**:
   - The fine-tuned Road Rage Detection Model is deployed for real-time video analysis. It processes video streams (from cameras or pre-recorded files) and classifies incidents as "Normal" or "Road Rage".

---

## Model Design

### **1. Violence Detection Model (Base Model)**
   - **Input Layer**: 3D ConvNet that accepts video data as `(frames, height, width, channels)`.
   - **Hidden Layers**: 
     - Convolutional layers followed by MaxPooling to extract spatial and temporal features.
     - BatchNormalization to improve convergence.
     - Dropout for regularization.
   - **Output Layer**: Dense layer with 2 neurons (binary classification: "Violent" vs "Non-Violent").

### **2. Road Rage Detection Model (Fine-tuned Model)**
   - **Base Model**: The pretrained Violence Detection Model (frozen layers).
   - **Added Layers**:
     - Dense layers with ReLU activation for non-linearity.
     - Dropout to avoid overfitting.
     - Output layer with 2 neurons for binary classification ("Normal" vs "Road Rage").

---

## Data Flow

1. **Input**:
   - Video input is provided either as a file (e.g., `.mp4`) or live from a camera feed.
   - Preprocessing steps include frame extraction, resizing, and normalization of pixel values.

2. **Model Processing**:
   - The video frames are passed through the **Road Rage Detection Model**, which outputs the classification label for each frame or video segment.

3. **Output**:
   - A prediction is made for each frame, and the system labels the video as either "Normal" or "Road Rage."
   - The output can be displayed in real-time or saved to disk for later review.

---

## Evaluation Metrics

- **Accuracy**: The proportion of correctly classified instances (either "Normal" or "Road Rage").
- **Precision**: The proportion of true positives (road rage incidents) among all predicted positive instances.
- **Recall**: The proportion of true positives identified among all actual road rage incidents.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.
- **Confusion Matrix**: A matrix showing the performance of the classification model in terms of true positives, false positives, true negatives, and false negatives.

---

## Training and Fine-tuning Process

### **Violence Detection Model**
   - Train the model on a violence detection dataset for binary classification.
   - Save the model after training and use it for transfer learning in the road rage detection model.

### **Road Rage Detection Model**
   - Load the pretrained **Violence Detection Model**.
   - Fine-tune the model by adding road rage-specific classification layers.
   - Train the modified model on the road rage dataset.
   - Evaluate the model's performance and save it for future use.

---

## Future Work

- **Data Augmentation**: Increase the robustness of the models by augmenting the training dataset (e.g., rotations, flipping, color jitter).
- **Real-time Streaming**: Improve the system’s ability to handle real-time video streams with low latency.
- **Deployment**: Explore deployment options for traffic monitoring, including cloud services and edge devices.

---

## Conclusion

The Road Rage Detection System is a powerful tool that combines deep learning with computer vision to detect aggressive driving behavior in real-time. By leveraging transfer learning from a pretrained violence detection model, the system is both efficient and effective in identifying road rage incidents, enhancing road safety, and potentially supporting law enforcement efforts.

