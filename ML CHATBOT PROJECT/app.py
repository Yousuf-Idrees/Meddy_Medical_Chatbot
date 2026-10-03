import json
import torch
import nltk
import pickle
import random
import os
from datetime import datetime
import numpy as np
import pandas as pd

from nnet import NeuralNet
from nltk_utils import bag_of_words
from flask import Flask, render_template, url_for, request, jsonify

random.seed(datetime.now().timestamp())

device = torch.device('cpu')
FILE = "models/data.pth"
model_data = torch.load(FILE)

input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']
all_words = model_data['all_words']
tags = model_data['tags']
model_state = model_data['model_state']

nlp_model = NeuralNet(input_size, hidden_size, output_size).to(device)
nlp_model.load_state_dict(model_state)
nlp_model.eval()

diseases_description = pd.read_csv("data/symptom_Description.csv")
diseases_description['Disease'] = diseases_description['Disease'].apply(lambda x: x.lower().strip(" "))

disease_precaution = pd.read_csv("data/symptom_precaution.csv")
disease_precaution['Disease'] = disease_precaution['Disease'].apply(lambda x: x.lower().strip(" "))

symptom_severity = pd.read_csv("data/Symptom-severity.csv")
symptom_severity = symptom_severity.map(lambda s: s.lower().strip(" ").replace(" ", "") if type(s) == str else s)

disease_symptoms_df = pd.read_csv("data/dataset.csv")


with open('data/list_of_symptoms.pickle', 'rb') as data_file:
    symptoms_list = pickle.load(data_file)

from spellchecker import SpellChecker
spell = SpellChecker()
spell.word_frequency.load_words(all_words)
for sym in symptoms_list:
    for word in sym.split("_"):
        spell.word_frequency._dictionary[word] = 1000000

try:
    with open('models/fitted_model.pickle', 'rb') as modelFile:
        prediction_model = pickle.load(modelFile)
except Exception as e:
    print(f"Warning: Could not load prediction model: {e}")
    prediction_model = None

user_symptoms = set()
rejected_symptoms = set()
pending_suggestion = None
questions_asked_count = 0

app = Flask(__name__, template_folder='static/templates')

CHAT_HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(entry):
    hist = load_history()
    hist.append(entry)
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(hist, f)

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(load_history())

@app.route('/load_history', methods=['POST'])
def load_history_endpoint():
    global user_symptoms, rejected_symptoms, pending_suggestion, questions_asked_count
    data = request.json
    
    user_symptoms.clear()
    rejected_symptoms.clear()
    pending_suggestion = None
    questions_asked_count = 0
    
    for s in data.get('symptoms', []):
        user_symptoms.add(s)
        
    return jsonify({"status": "success"})

def get_symptom(sentence):
    sentence = nltk.word_tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = nlp_model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    prob = prob.item()

    return tag, prob

@app.route('/')
def index():
    global pending_suggestion, questions_asked_count
    data = []
    user_symptoms.clear()
    rejected_symptoms.clear()
    pending_suggestion = None
    questions_asked_count = 0
    file = open("static/assets/files/ds_symptoms.txt", "r")
    all_symptoms = file.readlines()
    for s in all_symptoms:
        data.append(s.replace("'", "").replace("_", " ").replace(",\n", ""))
    data = json.dumps(data)

    return render_template('index.html', data=data)


@app.route('/symptom', methods=['GET', 'POST'])
def predict_symptom():
    global pending_suggestion, questions_asked_count
    try:
        print("Request json:", request.json)
        sentence = request.json['sentence']
        if sentence.replace(".", "").replace("!","").lower().strip() in ["done", "force done"]:

            if not user_symptoms:
                response_sentence = random.choice(
                    ["I can't know what disease you may have if you don't enter any symptoms :)",
                    "Meddy can't know the disease if there are no symptoms...",
                    "You first have to enter some symptoms!"])
            else:
                if prediction_model is None:
                    response_sentence = "Disease prediction model is not available. Please check the server setup."
                else:
                    x_test = []

                    for each in symptoms_list:
                        if each in user_symptoms:
                            x_test.append(1)
                        else:
                            x_test.append(0)

                    x_test = np.asarray(x_test)
                    
                    if hasattr(prediction_model, "predict_proba"):
                        proba = prediction_model.predict_proba(x_test.reshape(1,-1))[0]
                        max_prob = np.max(proba)
                        
                        if max_prob < 0.8 and sentence.replace(".", "").replace("!","").lower().strip() != "force done":
                            response_sentence = f"My confidence in the diagnosis is currently quite low ({(max_prob*100):.1f}%). To ensure a safer and more accurate prediction, could you please tell me if you have any other symptoms? If you really don't have any others, type <b>'Force Done'</b> to see the prediction anyway."
                            return jsonify(response_sentence)

                    disease = prediction_model.predict(x_test.reshape(1,-1))[0]
                    print(disease)

                    clean_disease = disease.strip(" ").lower()
                    description = diseases_description.loc[diseases_description['Disease'] == clean_disease, 'Description'].iloc[0]
                    precaution = disease_precaution[disease_precaution['Disease'] == clean_disease]
                    precautions = 'Precautions: ' + precaution.Precaution_1.iloc[0] + ", " + precaution.Precaution_2.iloc[0] + ", " + precaution.Precaution_3.iloc[0] + ", " + precaution.Precaution_4.iloc[0]
                    response_sentence = "It looks to me like you have " + disease + ". <br><br> <i>Description: " + description + "</i>" + "<br><br><b>"+ precautions + "</b>"

                    critical_diseases = [
                        'heart attack', 
                        'paralysis (brain hemorrhage)', 
                        'aids', 
                        'dengue', 
                        'tuberculosis', 
                        'pneumonia',
                        'typhoid'
                    ]
                    
                    if clean_disease in critical_diseases:
                        response_sentence += "<br><br><span style='color:red; font-size:1.1em;'><b>🚨 EMERGENCY WARNING:</b> This is a critical condition. Please seek immediate medical attention or call emergency services right away!</span>"
                    else:
                        severity = []
                        for each in user_symptoms:
                            severity.append(symptom_severity.loc[symptom_severity['Symptom'] == each.lower().strip(" ").replace(" ", ""), 'weight'].iloc[0])

                        if np.mean(severity) > 4 or np.max(severity) > 5:
                            response_sentence = response_sentence + "<br><br>Considering your symptoms are severe, and Meddy isn't a real doctor, you should consider talking to one. :)"

                    save_history({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symptoms": list(user_symptoms),
                        "disease": clean_disease
                    })

                    user_symptoms.clear()
                    rejected_symptoms.clear()
                    pending_suggestion = None
                    questions_asked_count = 0

        else:
            import re
            
            words = nltk.word_tokenize(sentence.lower())
            misspelled = spell.unknown(words)
            corrected_words = []
            for word in words:
                if word in misspelled:
                    c = spell.correction(word)
                    corrected_words.append(c if c else word)
                else:
                    corrected_words.append(word)
            corrected_sentence = " ".join(corrected_words)
            
            phrases = re.split(r'[.,;!?]|\band\b|\bbut\b', corrected_sentence)
            phrases = [p.strip() for p in phrases if p.strip()]
            
            greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "hi meddy", "hello meddy", "good morning meddy"]
            affirmative = ["yes", "yeah", "yep", "sure", "y", "of course", "i do", "yes i do", "yes i feel", "yes i have"]
            negative = ["no", "nope", "nah", "not really", "n", "i dont", "i don't", "no i don't", "no i dont", "no i do not", "no i feel"]
            
            found_symptoms = []
            answered_suggestion = False
            
            for p in phrases:
                if len(p) == 0:
                    continue
                
                is_affirmative = any(p == a or p.startswith(a + " ") for a in affirmative)
                is_negative = any(p == a or p.startswith(a + " ") for a in negative)
                
                if is_affirmative and pending_suggestion:
                    user_symptoms.add(pending_suggestion)
                    answered_suggestion = True
                    pending_suggestion = None
                    continue
                if is_negative and pending_suggestion:
                    rejected_symptoms.add(pending_suggestion)
                    answered_suggestion = True
                    pending_suggestion = None
                    continue
                    
                if not any(p == g for g in greetings) and not is_affirmative and not is_negative:
                    exact_matches = []
                    for sym in symptoms_list:
                        clean_sym = sym.replace("_", " ")
                        if clean_sym in p:
                            exact_matches.append((sym, 1.0))
                    
                    if exact_matches:
                        found_symptoms.extend(exact_matches)
                    else:
                        symptom, prob = get_symptom(p)
                        if prob > .5:
                            found_symptoms.append((symptom, prob))
            
            if found_symptoms or answered_suggestion:
                if found_symptoms:
                    if len(found_symptoms) == 1:
                        sym, prob = found_symptoms[0]
                        user_symptoms.add(sym)
                        response_sentence = f"Hmm, I've noted {sym.replace('_', ' ')}."
                    else:
                        added = []
                        for sym, prob in found_symptoms:
                            user_symptoms.add(sym)
                            added.append(sym.replace('_', ' '))
                        response_sentence = f"I've noted the following symptoms: {', '.join(added)}."
                else:
                    response_sentence = "Got it."
                
                if prediction_model is not None and user_symptoms:
                    x_test = []
                    for each in symptoms_list:
                        x_test.append(1 if each in user_symptoms else 0)
                    
                    x_test = np.asarray(x_test)
                    if hasattr(prediction_model, "predict_proba"):
                        proba = prediction_model.predict_proba(x_test.reshape(1,-1))[0]
                        top_disease_idx = np.argmax(proba)
                        top_disease = prediction_model.classes_[top_disease_idx]
                        
                        disease_rows = disease_symptoms_df[disease_symptoms_df['Disease'].str.lower().str.strip() == top_disease.lower().strip()]
                        possible_symptoms = []
                        for col in disease_symptoms_df.columns[1:]:
                            possible_symptoms.extend(disease_rows[col].dropna().unique().tolist())
                        
                        possible_symptoms = [s.strip().lower() for s in possible_symptoms if type(s) == str and s.strip() != ""]
                        remaining_symptoms = [s for s in set(possible_symptoms) if s not in user_symptoms and s not in rejected_symptoms]
                        
                        if remaining_symptoms and questions_asked_count < 3:
                            pending_suggestion = remaining_symptoms[0]
                            response_sentence += f" To make sure my prediction will be correct, do you also feel {pending_suggestion.replace('_', ' ')}?"
                            questions_asked_count += 1
                        else:
                            pending_suggestion = None
                            if np.max(proba) >= 0.8:
                                response_sentence += " I have enough information now! I am ready to diagnose, so please type 'Done'."
                            else:
                                response_sentence += f" I've asked a few questions, but my confidence is still quite low ({(np.max(proba)*100):.1f}%). Could you please tell me if you have any other symptoms? If you really don't have any others, type <b>'Force Done'</b> to see the prediction anyway."
                    else:
                        pending_suggestion = None
                        response_sentence += " I have enough information now! I am ready to diagnose, so please type 'Done'."
            else:
                if any(g in sentence.lower() for g in greetings):
                    response_sentence = "Hello! Please tell me what symptoms you are experiencing."
                else:
                    response_sentence = "I'm sorry, I couldn't understand that. The text seems unreadable or doesn't contain a symptom I recognize. Could you please check your spelling and try describing it again?"

            print("User symptoms:", user_symptoms)

        return jsonify(response_sentence.replace("_", " "))
    except Exception as e:
        print(f"Error in predict_symptom: {str(e)}")
        error_response = "Sorry, something went wrong while processing your request. Please try again."
        return jsonify(error_response), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
