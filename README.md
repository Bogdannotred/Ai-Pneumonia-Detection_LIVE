# Pneumonia Detection System

**Version: 0.4** ⚠️ **Work in Progress - Not Production Ready**

![Alt text](https://github.com/Bogdannotred/Ai-Pneumonia-Detection/blob/main/project_photos/projectphoto.png)

## Overview

## “The model accuracy is 70%. Work in progress…”

This project is an AI-powered pneumonia detection system that uses deep learning to analyze chest X-ray images and predict whether a patient has pneumonia or not. The system consists of a machine learning model trained on chest X-ray images, a FastAPI backend for inference, and a Streamlit frontend for user interaction.

## What It Does

The system can:
- Analyze chest X-ray images (JPEG/PNG format)
- Classify images as either **NORMAL** or **PNEUMONIA**
- Provide prediction confidence scores
- Generate Grad-CAM heatmaps to visualize which areas of the image the model focuses on

## How It Works

### Architecture

1. **Training Script** (`train_ai.py`)
   - Uses DenseNet121 (pre-trained on ImageNet) as the base model
   - Fine-tuned on chest X-ray images from the training dataset
   - Implements data augmentation (rotation, shifts, flips, zoom)
   - Saves the best model as `best_model.h5`

2. **Backend** (`backend.py`)
   - FastAPI server that loads the trained model
   - Accepts image uploads via POST request
   - Preprocesses images (resize to 224x224, normalize)
   - Runs inference and generates Grad-CAM heatmaps
   - Returns prediction results

3. **Frontend** (`frontend.py`)
   - Streamlit web interface
   - Allows users to upload chest X-ray images
   - Displays prediction results from the backend

### Dataset Structure

```
for_train_dataset/
├── train/
│   ├── NORMAL/ (1341 images)
│   └── PNEUMONIA/ (3875 images)
├── test/
│   ├── NORMAL/ (234 images)
│   └── PNEUMONIA/ (390 images)
└── val/
    ├── NORMAL/ (5 images)
    └── PNEUMONIA/ (4 images)
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Pneumonia Detection"
```

2. Create a virtual environment (recommended):
```bash
python -m venv tfenv
source tfenv/bin/activate  # On Windows: tfenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Update the dataset paths in `train_ai.py` if needed
2. Run the training script:
```bash
python train_ai.py
```

The model will be saved as `best_model.h5` in the project root.

### Running the Application

1. **Start the Backend Server:**
```bash
python backend.py
# Or with uvicorn:
uvicorn backend:app --reload
```
The backend will run on `http://127.0.0.1:8000`

2. **Start the Frontend (in a new terminal):**
```bash
streamlit run frontend.py
```
The frontend will be available at `http://localhost:8501`

3. **Use the Application:**
   - Open the Streamlit interface in your browser
   - Upload a chest X-ray image (JPG or PNG)
   - Click "Process" to get the prediction

## Requirements

- Python 3.11+
- TensorFlow 2.20.0
- FastAPI
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Pillow

See `requirements.txt` for the complete list of dependencies.

## Current Status (v0.4)

⚠️ **This is an early development version. The following features are incomplete or need improvement:**

- [ ] Error handling and validation
- [ ] Model performance optimization
- [ ] Frontend UI/UX improvements
- [ ] Grad-CAM visualization display in frontend
- [ ] API documentation
- [ ] Testing suite
- [ ] Docker containerization
- [ ] Deployment configuration

## Notes

- The model expects images to be 224x224 pixels
- Currently supports JPEG and PNG formats
- The backend and frontend need to run simultaneously
- Model file (`best_model.h5`) must be present in the project root for the backend to work

## License

[Add your license here]

## Contributing

This project is currently in active development. Contributions and feedback are welcome!
