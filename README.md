# **Road Rage Detection System**

## **Project Overview**
This project consists of two stages:  
1. **Stage 1 - Violence Detection Model**: A pretrained model trained on generic violence detection datasets to classify whether a video contains violence or not.  
2. **Stage 2 - Road Rage Detection Model**: A model created using **transfer learning**, which fine-tunes the pretrained **Violence Detection Model** for the specific task of detecting road rage from videos.  

The final **Road Rage Detection System** detects aggressive driving behavior in real-time or from recorded videos, helping improve traffic safety and law enforcement.

---

## **Table of Contents**
1. [Project Description](#project-description)  
2. [Folder Structure](#folder-structure)  
3. [Setup and Installation](#setup-and-installation)  
4. [Models Overview](#models-overview)  
5. [Usage](#usage)  
6. [Results](#results)  
7. [Visualisation](#visualization)
8. [Output Videos](#output-videos)  
9. [Datasets](#datasets)  
10. [License](#license)  
11. [Contact](#contact)

---

## **Project Description**
The project implements a two-stage model-based approach:  
1. **Violence Detection**: A generic violence detection model was trained to classify videos as violent or non-violent.  
2. **Road Rage Detection**: The pretrained violence detection model was used as the base for transfer learning. A new dataset was curated for road rage detection, and the model was fine-tuned for binary classification (Normal vs. Road Rage).  

---

## **Folder Structure**

```
road-rage-detection/
│
├── README.md              # Project overview, setup, and usage instructions
├── requirements.txt       # List of dependencies
├── .gitignore             # Specifies files to ignore in the repository
│
├── notebooks/             # Jupyter notebooks for model training and analysis
│   ├── LiveDetection(RoadRage).ipynb   # Live detection for road rage incidents
│   ├── LiveDetection(Violence).ipynb   # Live detection for violence
│   ├── RoadRage.ipynb                  # Model training and fine-tuning for road rage detection
│   └── Violence.ipynb                   # Model training for violence detection
│
├── output/                # Stores the output videos and frames
│   ├── detected_videos/   # Output videos with detected incidents
|   ├── violence
|   ├── roadrage
│   ├── confusion_matrices/ # Confusion matrix images
├── detected_frames/   # Frames extracted from detected incidents
│   ├── violence/      # Extracted frames from violence detection
│   ├── road_rage/     # Extracted frames from road rage detection
```

---

## **Setup and Installation**

### **Clone the Repository**
```bash
git clone https://github.com/yourusername/road-rage-detection.git
cd road-rage-detection
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **Models Overview**

### **Stage 1: Violence Detection Model**
- **Purpose**: Trained on generic violence datasets to classify violent vs. non-violent videos.
- **Architecture**: 3D CNN (3D Convolutional Neural Network).  
- **Dataset**: Publicly available violence detection dataset: [RWF-2000 Dataset](https://www.kaggle.com/datasets/vulamnguyen/rwf2000)  

### **Stage 2: Road Rage Detection Model**
- **Purpose**: Fine-tuned the pretrained violence detection model to classify road rage incidents.  
- **Transfer Learning**:  
  - Pretrained layers from the **violence detection model** were frozen.
  - Final layers were updated for binary classification (Normal vs. Road Rage).  
- **Dataset**: Road Rage dataset: [Road Rage Dataset](https://www.kaggle.com/datasets/shaiktanveer7/road-rage-dataset)  

---

## **Usage**

### **Train the Models**
1. **Train Violence Detection Model**:  
   ```bash
   python notebooks/Violence.ipynb
   ```
   This trains the **violence detection model** and saves it.

2. **Train Road Rage Detection Model**:  
   ```bash
   python notebooks/RoadRage.ipynb
   ```
   This fine-tunes the **violence detection model** for road rage detection.

### **Real-Time Detection**
To detect road rage incidents in real-time from a video stream:
```bash
python notebooks/LiveDetection(RoadRage).ipynb
```
To detect general violence in real-time:
```bash
python notebooks/LiveDetection(Violence).ipynb
```

---

## **Results**

### **Violence Detection Model**
- **Accuracy**: 98%  
- **Precision (Normal)**: 0.99 
- **Precision (Violent)**: 0.96
- **Recall (Normal)**: 0.96
- **Recall (Violent)**: 0.99  
- **F1-Score (Normal)**: 0.99
- **F1-Score (Violent)**: 0.98



### **Road Rage Detection Model**
- **Accuracy**: 94%  
- **Precision (Normal)**: 0.99 
- **Precision (Road Rage)**: 0.90 
- **Recall (Normal)**: 0.88 
- **Recall (Road Rage)**: 0.99 
- **F1-Score (Normal)**: 0.94
- **F1-Score (Road Rage)**: 0.94 


### **Visualization**
Performance graphs and confusion matrices can be found in the `output/confusion_matrices` folder.

---

## **Output Videos**
### **Detected Video Samples**
Output videos of detected road rage incidents can be found in the `output/detected_videos/road-rage_detection` folder.
![Live Road Rage Detection](https://github.com/tanveer744/road-rage-detection/releases/download/v1/road-rage.mp4)

Output videos of detected Violence incidents can be found in the `output/detected_videos/violence_detection` folder.
![Live Violence Detection](https://github.com/tanveer744/road-rage-detection/releases/download/v2/violence.mp4)

---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Contact**
For more information, feel free to reach out:
- **Shaik Tanveer Lohare**: shaiktanveer07404@gmail.com
- **Mohammed Maaz**: maazkhaleel17@gmail.com
- **Mahammad Razi**: mohdrazi4408@gmail.com
- **Mohammed Nehal**: mohammednehal486@gmail.com

