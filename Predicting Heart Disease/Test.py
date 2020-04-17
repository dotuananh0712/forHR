
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import time
test = pd.read_csv('E://Python/Predicting Heart Disease/test_values.csv')
train = pd.read_csv('E://Python/Predicting Heart Disease/train_values.csv')
labels = pd.read_csv('E://Python/Predicting Heart Disease/train_labels.csv')
labels.drop('patient_id', axis=1, inplace=True)
df = pd.concat([train, labels], axis=1)
df.drop(['patient_id'], axis=1, inplace=True)
df['thal'] = df['thal'].map({'normal': 1, 'reversible_defect': 2, 'fixed_defect': 3})
y = np.asarray(df['heart_disease_present'])
df.drop(['heart_disease_present'], axis=1, inplace=True)
x = np.asarray(df)
print(df.shape)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=4)
ts = time.time()

parameter = [{'max_features': [2, 3, 4],
              'min_samples_leaf':[1, 2]}]

RF = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=1024, oob_score=False)

grid = GridSearchCV(RF, parameter, cv=10)

grid.fit(xtrain, ytrain)

print("Best score of Ridge: ", grid.best_score_)

print("\nIt took", np.round((time.time()-ts)/60, 3), "minutes to run.")
grid.best_estimator_

predict1 = grid.predict_proba(xtrain)
print("\nOut-Sample logloss: ", np.round(log_loss(ytrain, predict1), 4))

predict2 = grid.predict_proba(xtest)
print("\nOut-Sample Logloss: ", np.round(log_loss(ytest, predict2), 4))
idd = test['patient_id']
test['thal'] = test['thal'].map({'normal': 1, 'reversible_defect': 2, 'fixed_defect': 3})
test.drop(['patient_id'], axis=1, inplace=True)
xx = np.asarray(test)
predictions = grid.predict_proba(xx)
pred = predictions[:, 1]
data = pd.DataFrame({'patient_id': idd, 'heart_disease_present': pred})
print(test.shape, xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
