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
7. [Output Videos & Frames](#output-videos--frames)  
8. [Datasets](#datasets)  
9. [License](#license)  
10. [Contact](#contact)

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
│   ├── frames/            # Extracted frames highlighting incidents
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
- **Precision**: 0.93  
- **Recall**: 0.90  

### **Road Rage Detection Model**
- **Accuracy**: 94%  
- **Precision (Normal)**: 0.95  
- **Recall (Road Rage)**: 0.94  

### **Visualization**
Performance graphs and confusion matrices can be found in the notebooks.

---

## **Output Videos & Frames**
### **Detected Video Samples**
Sample output videos of detected road rage incidents can be found in the `output/detected_videos/` folder.

### **Extracted Frames**
Keyframes highlighting aggressive behavior are stored in the `output/frames/` folder. These frames provide clear visual evidence of detected incidents.

To generate and store output frames, run:
```bash
python scripts/extract_frames.py
```
This script extracts and saves frames whenever a road rage incident is detected.

---

## **Datasets**
- **Road Rage Dataset**: [Road Rage Dataset](https://www.kaggle.com/datasets/shaiktanveer7/road-rage-dataset)
- **Violence Detection Dataset**: [RWF-2000 Dataset](https://www.kaggle.com/datasets/vulamnguyen/rwf2000)

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

