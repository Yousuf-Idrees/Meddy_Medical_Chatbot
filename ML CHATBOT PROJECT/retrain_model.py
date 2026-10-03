import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle

print("Loading dataset...")
df = pd.read_csv("data/dataset.csv")
df = df.drop_duplicates()

print("Preparing features...")
# Get all unique symptoms
symptoms = np.concatenate((df.Symptom_1.unique(), df.Symptom_2.unique(), df.Symptom_3.unique(), df.Symptom_4.unique(),
                          df.Symptom_5.unique(), df.Symptom_6.unique(), df.Symptom_7.unique(), df.Symptom_8.unique(),
                          df.Symptom_9.unique(), df.Symptom_10.unique(), df.Symptom_11.unique(), df.Symptom_12.unique(),
                          df.Symptom_13.unique(), df.Symptom_14.unique(), df.Symptom_15.unique(), df.Symptom_16.unique(),
                          df.Symptom_17.unique()))

symptoms_unique = list((set(symptoms)))

# Create one-hot encoding
i = 18
for each in symptoms_unique:
    df.insert(i, each, 0)
    i = i + 1

df = df.fillna(0)

for index, row in df.iterrows():
    disease_symptoms = [symptom for symptom in list(row)[1:] if symptom != 0]
    for each in disease_symptoms:
        df.at[index, each] = 1

df = df.drop(columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8',
             'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15', 'Symptom_16'
             , 'Symptom_17'])

df = df.loc[:, df.columns.notnull()]
df.columns = df.columns.str.replace(' ', '')
df = df.reindex(sorted(df.columns), axis=1)

labels = df.to_numpy()[:, :1]
examples = df.to_numpy()[:, 1:]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(examples, labels.ravel(), test_size=0.2, random_state=42)

print("Building ensemble model...")
level0 = list()
level0.append(('lr', LogisticRegression(solver='liblinear', C=0.03, max_iter=1000)))
level0.append(('knn', KNeighborsClassifier(n_neighbors=6, metric='cosine')))
level0.append(('dctree', DecisionTreeClassifier(splitter='random', max_depth=34, random_state=42)))
level0.append(('svm', SVC(C=0.1)))

level1 = LogisticRegression(max_iter=1000)
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

print("Training model...")
model.fit(examples, labels.ravel())

print("Testing model...")
print(f"Test Score: {model.score(X_test, y_test):.4f}")

print("Saving model...")
with open('models/fitted_model.pickle', 'wb') as modelFile:
    pickle.dump(model, modelFile)

print("Model saved successfully!")
