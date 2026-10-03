# Meddy_Medical_Chatbot
A conversational AI medical chatbot that processes everyday symptom text using NLP and deep learning, then accurately predicts diseases using a multi-model Stacking Classifier with Flask web deployment.
# Meddy: AI Medical Chatbot & Symptom-Checking System

Meddy is an interactive, conversational AI symptom-checking chatbot that bridges the gap between everyday user language and structured medical diagnostic models. By combining Natural Language Processing (NLP), PyTorch deep learning for intent recognition, and a multi-model Scikit-learn Stacking Classifier, Meddy parses unstructured symptom descriptions, conducts intelligent follow-up inquiries, and outputs predicted conditions alongside descriptions, precautions, and emergency warnings.

---

## Architecture & System Workflow

Meddy operates via a dual-layer AI decision framework integrated into a Flask web application:

```text
User Input (Natural Text) 
  └──> Tokenization & Spelling Correction (NLTK, SpellChecker)
        └──> Layer 1: NLP Symptom Recognition (PyTorch Neural Net)
              └──> Dynamic Conversation Engine (Follow-Up Inquiries & Yes/No Logic)
                    └──> Layer 2: Disease Prediction (Stacking Classifier)
                          └──> Comprehensive Output (Disease, Precautions & Safety Alerts)

Symptom Parsing: Transforms raw text into clean, standardized symptom tags.
Interactive Dialogue: Asks targeted follow-up questions based on co-occurring symptoms to improve input quality.
Multi-Model Inference: Converts confirmed symptoms into a binary one-hot vector and predicts the condition.
Enriched Medical Response: Delivers actionable context, four tailored precautions, and severity-weighted emergency alerts.
Machine Learning Models
Layer 1: Symptom Recognition (PyTorch Neural Network)
Objective: Maps informal user phrasing (e.g., "I threw up", "my head hurts") to standard dataset symptom tags.
Architecture: Multi-Layer Perceptron (MLP) with Linear layers, ReLU activation, and Bag-of-Words (BoW) input vectors.
Layer 2: Disease Prediction (Stacking Classifier)
Objective: Evaluates binary symptom vectors (1 = present, 0 = absent) across 131 unique symptoms to predict 41 target conditions.
Base Learners:
Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Support Vector Machine (SVM)
Meta-Estimator: Logistic Regression (trained using 5-fold cross-validation cv=5).
Key Features
Natural Language Understanding: Processes free-form user descriptions without requiring exact medical terminology.
Automated Spell Correction: Handles common user typos (e.g., "vomting" → "vomiting").
Smart Follow-Up Engine: Asks up to 3 context-aware questions to confirm or reject related co-symptoms.
Safety & Emergency System: Cross-references symptom severity weights and high-risk conditions to issue immediate medical urgency disclaimers.
Session Tracking: Persists chat history, selected symptoms, timestamps, and diagnostic outputs to chat_history.json.

.
├── app.py                     # Main Flask web application and UI routes
├── train_symptom_model.py     # PyTorch script for NLP intent classification
├── train_disease_model.py     # Scikit-learn script for Stacking Classifier training
├── data/
│   ├── dataset.csv            # Primary dataset (4,920 records, 41 diseases, 131 symptoms)
│   ├── intents.json           # 132 symptom tag patterns and training phrases
│   ├── symptom_severity.csv   # Symptom weighting table for safety warnings
│   ├── disease_description.csv# Clinical descriptions for predicted conditions
│   └── disease_precaution.csv # Actionable precautions per disease
├── models/
│   ├── symptom_model.pth      # Trained PyTorch neural network weights
│   └── fitted_model.pickle    # Serialized Stacking Classifier model
└── chat_history.json          # Persisted user conversation logs

Installation & Setup
1. Prerequisites
Python 3.8+ installed on your machine.
2. Clone Repository
git clone [https://github.com/YourUsername/Meddy-AI-Medical-Chatbot.git](https://github.com/YourUsername/Meddy-AI-Medical-Chatbot.git)
cd Meddy-AI-Medical-Chatbot
3. Install Dependencies
pip install flask pytorch torch torchvision scikit-learn pandas numpy nltk pyspellchecker
4. NLTK Data Download
Ensure NLTK tokenization packages are loaded by executing inside Python:
import nltk
nltk.download('punkt')
5.Usage
Training Models (Optional)
To retrain the symptom classification or disease prediction pipelines from raw data:
# Train PyTorch NLP Neural Network
python train_symptom_model.py

# Train Scikit-learn Stacking Classifier
python train_disease_model.py
Running the Web Application
Launch the Flask development server:
python app.py

Medical Disclaimer
IMPORTANT: Meddy is an educational research prototype designed to demonstrate conversational AI and machine learning workflows in healthcare. It does not provide professional medical diagnosis, advice, or treatment plans. Critical symptoms or medical emergencies should always be evaluated immediately by qualified medical professionals.
