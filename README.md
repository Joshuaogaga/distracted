# Comparative Analysis of Pre-trained CNN Architectures for Driver Distraction Detection

## Project Overview

### Project Title
Comparative Analysis of Pre-trained CNN Architectures for Driver Distraction Detection

### Introduction
Driver distraction is a major concern for road safety, contributing significantly to traffic accidents worldwide. Traditional monitoring methods are often inadequate for real-time detection of distracted driving behaviors. This project leverages advanced deep learning techniques, specifically comparing five state-of-the-art pre-trained models (ResNet, EfficientNet, VGG, MobileNetV2, and DenseNet), to create a robust system for detecting distracted driving behaviors. The project includes both a comparative analysis of model performance and a practical implementation through a Streamlit application that supports image, video, and webcam-based detection.

## Models Implemented
- **ResNet**: Deep residual learning framework
- **EfficientNet**: Advanced architecture with balanced depth, width, and resolution
- **VGG**: Classic architecture known for its simplicity and effectiveness
- **MobileNetV2**: Lightweight architecture optimized for mobile devices
- **DenseNet**: Dense Convolutional Network with enhanced feature reuse

## Dataset
This project uses the State Farm Distracted Driver Detection dataset from Kaggle. The dataset contains images of drivers captured in a car while performing various actions, from safe driving to different types of distractions.

### Dataset Structure
Total Images: 102,150
- Training Images: 17,939
- Validation Images: 4,485
- Test Images: 79,726

### Classes
The dataset is categorized into 10 classes:
1. c0: Safe driving
2. c1: Texting - right
3. c2: Talking on the phone - right
4. c3: Texting - left
5. c4: Talking on the phone - left
6. c5: Operating the radio
7. c6: Drinking
8. c7: Reaching behind
9. c8: Hair and makeup
10. c9: Talking to passenger

### Data Source
The dataset is available through the State Farm Distracted Driver Detection competition on Kaggle.

## How to Access and Run the Project

### Prerequisites
- Python 3.8+
- Required libraries:
```bash
torch
torchvision
streamlit
opencv-python
numpy
pandas
scikit-learn
pillow
```

### Installation Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/driver-distraction-detection.git
   cd driver-distraction-detection
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Libraries**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
1. **For Model Training and Comparison**
   ```bash
   python train_models.py
   ```

2. **Launch the Streamlit Application**
   ```bash
   streamlit run app.py
   ```

## Model Evaluation and Comparison
The project implements comprehensive evaluation metrics to compare model performance:

### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy across all classes
- **Precision**: Measure of correctly identified positive predictions
- **Recall**: Measure of correctly identified actual positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **Inference Time**: Processing speed for real-time applications
- **Model Size**: Storage requirements and deployment considerations

### Streamlit Application Features
The interactive web application provides:
- Real-time distraction detection through webcam feed
- Support for image upload and analysis
- Video file processing capabilities
- Model selection option to compare different architectures
- Visualization of detection results
- Performance metrics display

## Project Structure
```
driver-distraction-detection/
├── data/
│   ├── train/
│   └── test/
├── src/
│   ├── app.py
│   └── utils.py
├── models/
│   ├── resnet_model.h5
│   ├── efficientnet_model.h5
│   ├── vgg_model.h5
│   ├── mobilenet_model.h5
│   └── densenet_model.h5
├── notebooks/
│   └── train.ipynb
├── requirements.txt
└── README.md
```

## Future Improvements
- Ensemble method implementation
- Real-time performance optimization
- Mobile device deployment
- Additional distraction categories
- Integration with vehicle systems