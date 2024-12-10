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
7. [License](#license)  
8. [Contact](#contact)

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
├── README.md                           # Project overview, setup, and usage instructions
├── requirements.txt                    # List of dependencies
├── .gitignore                          # Specifies files to ignore in the repository
│
├── src/                                # Source code for the project
│   ├── violence_detection.py           # Code for training and testing the Violence Detection Model
│   ├── road_rage_detection.py          # Code for fine-tuning and testing the Road Rage Detection Model
│   ├── live_detection.py               # Real-time detection using the Road Rage Detection Model
│   └── utils.py                        # Utility functions (data preprocessing, metrics, etc.)
│
├── models/                             # Folder for storing model files
│   ├── violence_detection_model.h5     # Pretrained Violence Detection Model
│   └── road_rage_model.h5              # Fine-tuned Road Rage Detection Model
│
├── data/                               # Dataset folder (not included in the repo due to size constraints)
│   ├── raw/                            # Raw video data (original dataset)
│   │   └── video_data/                 # Directory for storing raw video files
│   └── processed/                      # Preprocessed dataset (e.g., converted to .npy format)
│
├── notebooks/                          # Jupyter notebooks for experiments and analysis
│   ├── violence_training.ipynb         # Notebook for training the Violence Detection Model
│   └── road_rage_training.ipynb        # Notebook for fine-tuning the Road Rage Detection Model
│
├── results/                            # Folder to store results and outputs
│   ├── detection_samples/              # Example frames or videos with detected incidents
│   └── performance_plots/              # Training and evaluation plots (accuracy, loss, etc.)
│
└── docs/                               # Documentation and design-related files
    ├── design_documentation.md         # Project design and architecture description
    └── algorithm_details.md            # Algorithm details for both models

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

### **Download Models and Datasets**
- **Violence Detection Model**: Download the pretrained model [here](link-to-violence-model).  
- **Road Rage Detection Model**: Download the fine-tuned model [here](link-to-road-rage-model).  
- Place these files in the `models/` directory.  
- **Dataset**: Due to the large size, datasets are not included. Download the dataset [here](link-to-dataset).  

---

## **Models Overview**

### **Stage 1: Violence Detection Model**
- **Purpose**: Trained on generic violence datasets to classify violent vs. non-violent videos.
- **Architecture**: 3D CNN (3D Convolutional Neural Network).  
- **Dataset**: Publicly available violence detection datasets (e.g., Hockey Fight dataset, Crowd Violence dataset).  

### **Stage 2: Road Rage Detection Model**
- **Purpose**: Fine-tuned the pretrained violence detection model to classify road rage incidents.  
- **Transfer Learning**:  
  - Pretrained layers from the **violence detection model** were frozen.
  - Final layers were updated for binary classification (Normal vs. Road Rage).  
- **Dataset**: Custom road rage dataset collected from traffic cameras and online sources.  

---

## **Usage**

### **Train the Models**
1. **Train Violence Detection Model**:  
   ```bash
   python src/main.py --train-violence
   ```
   This trains the **violence detection model** and saves it as `violence_detection_model.h5`.

2. **Train Road Rage Detection Model**:  
   ```bash
   python src/main.py --train-road-rage
   ```
   This fine-tunes the **violence detection model** for road rage detection and saves it as `road_rage_model.h5`.

### **Evaluate the Models**
1. **Evaluate Violence Detection Model**:  
   ```bash
   python src/main.py --test-violence
   ```

2. **Evaluate Road Rage Detection Model**:  
   ```bash
   python src/main.py --test-road-rage
   ```

### **Real-Time Detection**
To detect road rage incidents in real-time from a video stream:
```bash
python src/detection_module.py --video_path path_to_video --output_dir path_to_output_directory
```

---

## **Results**

### **Violence Detection Model**
- **Accuracy**: 92%  
- **Precision**: 0.93  
- **Recall**: 0.90  

### **Road Rage Detection Model**
- **Accuracy**: 94%  
- **Precision (Normal)**: 0.95  
- **Recall (Road Rage)**: 0.94  

### **Visualization**
Confusion matrices and performance graphs can be found in the `results/` folder.

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
