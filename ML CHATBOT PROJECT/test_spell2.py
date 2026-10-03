from spellchecker import SpellChecker
import pickle

with open('data/list_of_symptoms.pickle', 'rb') as data_file:
    symptoms_list = pickle.load(data_file)

spell = SpellChecker()
for sym in symptoms_list:
    for word in sym.split("_"):
        spell.word_frequency._dictionary[word] = 1000000

print("vomting ->", spell.correction("vomting"))
