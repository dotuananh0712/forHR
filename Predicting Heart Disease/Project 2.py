from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np

# EDA
test = pd.read_csv('E://Python/Predicting Heart Disease/test_values.csv')
train = pd.read_csv('E://Python/Predicting Heart Disease/train_values.csv')
label = pd.read_csv('E://Python/Predicting Heart Disease/train_labels.csv')
label.drop('patient_id', axis=1, inplace=True)
df = pd.concat([train, label], axis=1)
patient_id = test['patient_id']
df.drop(['patient_id'], axis=1, inplace=True)
df_thal = pd.get_dummies(df, drop_first=True)

# Modeling
y = np.asarray(df_thal['heart_disease_present'])
df_thal.drop(['heart_disease_present'], axis=1, inplace=True)
X = np.asarray(df_thal)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=7)

# Logistic Regression CV
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test))
logreg = LogisticRegressionCV(cv=2)
logreg.fit(X_train_scaled, y_train)
y_pred2a = logreg.predict(X_test_scaled)
y_pred2b = logreg.predict_proba(X_test_scaled)[:, 1]
print("""[Logistic Regression CV]\n Entropy Lost: {}""".format(log_loss(y_test, y_pred2b)))
print(classification_report(y_test, y_pred2a))

# Random Forest Classifier
ranfor = RandomForestClassifier(n_jobs=-1, oob_score=False, random_state=7)
params = {'n_estimators': [20, 30, 40],
          'max_features': [2, 3, 4]
          }
rf = GridSearchCV(ranfor, params, cv=3, scoring='neg_log_loss')
rf.fit(X_train, y_train)
y_pred3a = rf.predict(X_test)
y_pred3b = rf.predict_proba(X_test)[:, 1]
print("""[Random Forest Classifier]\n Entropy Lost: {}""".format(log_loss(y_test, y_pred3b)))
print(classification_report(y_test, y_pred3a))
# Prediction
rf.fit(X, y.ravel())
test.drop(['patient_id'], axis=1, inplace=True)
test_thal = pd.get_dummies(test, drop_first=True)
test_thal = np.asarray(test_thal)
final_prediction = rf.predict_proba(test_thal)[:, 1]
Submission = pd.DataFrame({'patient_id': patient_id, 'heart_disease_present': final_prediction})
Submission.to_csv('test_submission.csv')
