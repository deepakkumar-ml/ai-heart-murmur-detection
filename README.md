# Heart Murmur Detection Using Deep Learning

## Overview

Heart Murmur Detection is a deep learning-based system for classifying heart sound recordings into three categories: **Normal**, **Murmur**, and **Artifact**. The project utilizes audio signal processing techniques and a hybrid CNN-LSTM architecture to analyze phonocardiogram (PCG) recordings and identify abnormal heart sounds.

The trained model is deployed as a Streamlit web application, enabling users to upload heart sound recordings and receive predictions in real time.

---

## Problem Statement

Heart murmurs are abnormal sounds produced during blood flow through the heart and may indicate underlying cardiovascular conditions. Traditional auscultation relies heavily on clinical expertise and can be subjective.

This project aims to automate heart sound classification using deep learning techniques, providing a scalable and accessible solution for preliminary heart sound analysis.

---

## Dataset

The dataset contains heart sound recordings belonging to the following classes:

| Class    | Description                     |
| -------- | ------------------------------- |
| Artifact | Noise or recording interference |
| Murmur   | Abnormal heart sound            |
| Normal   | Healthy heart sound             |

---

## Methodology

### Audio Preprocessing

* Audio loading and normalization
* Noise handling
* Fixed-length signal preparation

### Feature Extraction

Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from each audio recording to capture important acoustic characteristics of heart sounds.

### Model Training

The extracted MFCC features are used to train a hybrid CNN-LSTM model capable of learning both spatial and temporal patterns present in heart sound recordings.

---

## Model Architecture

The model consists of:

* Conv1D (2048 filters)

* MaxPooling1D

* Batch Normalization

* Conv1D (1024 filters)

* MaxPooling1D

* Batch Normalization

* Conv1D (512 filters)

* MaxPooling1D

* Batch Normalization

* LSTM (256 units)

* LSTM (128 units)

* Dense (64 units)

* Dropout (0.5)

* Dense (32 units)

* Dropout (0.5)

* Output Layer (Softmax)

### Training Configuration

* Optimizer: Adam
* Learning Rate: 0.0001
* Loss Function: Categorical Crossentropy

---

## Results

### Test Accuracy

**94.87%**

### Classification Performance

| Class    | Precision | Recall | F1-Score |
| -------- | --------- | ------ | -------- |
| Artifact | 0.93      | 1.00   | 0.97     |
| Murmur   | 0.94      | 0.82   | 0.87     |
| Normal   | 0.92      | 0.97   | 0.95     |

### Overall Accuracy

**93%**

---

## Technologies Used

* Python
* TensorFlow
* Keras
* Librosa
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit

---

## Project Structure

```text
Heart-Murmur-Detection/
│
├── app.py
├── requirements.txt
├── lstm_model.keras
├── lstm_model.h5
├── Heart_Murmur_Detection_Project.ipynb
├── README.md
└── assets/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/heart-murmur-detection.git
cd heart-murmur-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Usage

1. Launch the Streamlit application.
2. Upload a heart sound recording (.wav file).
3. The model processes the audio and extracts MFCC features.
4. The trained CNN-LSTM model generates a prediction.
5. The predicted class is displayed to the user.

---

## Streamlit Deployment

Live Application:

```text
https://your-streamlit-app-url.streamlit.app
```

---

## Future Improvements

* Integration with larger heart sound datasets
* Real-time heart sound monitoring
* Explainable AI for prediction interpretation
* Mobile application deployment
* Support for additional cardiac abnormalities

---

## Author

Deepak Kumar

Data Science | Machine Learning | Deep Learning

---

## Disclaimer

This project is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or clinical decision-making.
