import app
import nltk
import re

sentence = "Hello Meddy! I am feeling terrible. I have cheest pain and vomting."

words = nltk.word_tokenize(sentence.lower())
misspelled = app.spell.unknown(words)
corrected_words = []
for word in words:
    if word in misspelled:
        c = app.spell.correction(word)
        corrected_words.append(c if c else word)
    else:
        corrected_words.append(word)
corrected_sentence = " ".join(corrected_words)
print("Corrected:", corrected_sentence)

phrases = re.split(r'[.,;!?]|\band\b|\bbut\b', corrected_sentence)
phrases = [p.strip() for p in phrases if p.strip()]
print("Phrases:", phrases)

found_symptoms = []
for p in phrases:
    is_affirmative = any(p == a or p.startswith(a + " ") for a in app.affirmative)
    is_negative = any(p == a or p.startswith(a + " ") for a in app.negative)
    if not any(p == g for g in app.greetings) and not is_affirmative and not is_negative:
        symptom, prob = app.get_symptom(p)
        print(f"Phrase '{p}' -> {symptom} ({prob})")
        if prob > .5:
            found_symptoms.append((symptom, prob))

print("Found:", found_symptoms)
