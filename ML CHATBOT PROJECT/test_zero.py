import app

sentence = "hello"
symptom, prob = app.get_symptom(sentence)
print(f"Phrase: '{sentence}', Symptom: {symptom}, Prob: {prob}")

sentence = "good morning"
symptom, prob = app.get_symptom(sentence)
print(f"Phrase: '{sentence}', Symptom: {symptom}, Prob: {prob}")

sentence = "unknown_word_test"
symptom, prob = app.get_symptom(sentence)
print(f"Phrase: '{sentence}', Symptom: {symptom}, Prob: {prob}")
