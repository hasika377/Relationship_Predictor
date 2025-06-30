# 💖 Relationship Compatibility Predictor

Predict relationship compatibility using survey responses with the help of machine learning.  
This project combines *unsupervised clustering* to generate labels and a *supervised Artificial Neural Network (ANN)* to perform the final classification.

---

## 📌 Project Overview

The *Relationship Compatibility Predictor* analyzes answers from a relationship-based questionnaire to predict whether two individuals would make a good match.

### 🔗 Pipeline Includes:
- 📊 Survey Data Collection via Google Forms  
- 🏷️ Label Generation using *KMeans Clustering* (Unsupervised Learning)  
- 🧠 Model Training using an *Artificial Neural Network (ANN)* (Supervised Learning)  

This hybrid approach allows the model to:
- Discover natural clusters in respondent preferences (unsupervised),
- Learn complex patterns for accurate classification (supervised).

The final model is deployed through a *Streamlit app*, allowing users to input their answers and receive real-time compatibility predictions.

---

## 📊 1. Data Collection
- Responses gathered via a Google Form with *12 questions*.
- Questions were designed to capture preferences and personality traits.
- *149 responses* were collected and stored in CSV format.

---

## 🏷️ 2. Data Labeling
- Applied *KMeans Clustering* with k=2 to classify responses as either:
  - ✅ "Match"
  - ❌ "No Match"
- Used clustering to avoid rule-based bias and reveal natural groupings.

---

## 🔠 3. Preprocessing
- Applied *Label Encoding* to transform categorical answers into numerical format.
- Chosen for its compactness and order preservation.

---

## 🧠 4. Model Architecture
Developed an *Artificial Neural Network (ANN)*:
- 2 Hidden Layers with *ReLU*
- *Dropout* + *Batch Normalization* for regularization
- *Sigmoid Output Layer* for binary classification

---

## 💾 5. Model Saving
- Trained model saved as: best_model.keras using *Keras*
- *ModelCheckpoint* used to retain the best performing weights

---

## 🧪 6. Inference with Streamlit
The app (.py) is built using *Streamlit*:
- Accepts user inputs through a form
- Uses the trained model to predict compatibility
- Displays results in real-time

---

## ✅ How to Run

1. Clone the repository
https://github.com/hasika377/Relationship_Predictor.git
cd Relationship_Predictor

3. Install required packages  
pip install -r requirements.txt

4. Start the Streamlit app  
streamlit run Code_File.py

---

## 🚀 Features
- Real-time compatibility prediction  
- Easy-to-use form interface  
- Model built using real survey data  
- Compact and efficient model inference  

---

## 📁 Repository Structure

Relationship_Predictor/

├── Code_File.py                         # Streamlit app for predictions  
├── best_model.keras                     # Trained Keras model  
├── Assessment - Form Responses.csv      # Survey response dataset  
├── requirements.txt                     # Python dependencies  
└── README.md                            # Project overview

## 👥 Author

*Developed by:* Cholleti Hasika
- 🔗 https://github.com/hasika377
