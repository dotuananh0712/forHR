import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# Read in .data file
df = pd.read_csv(
    'Predicting Credit Card Approval Rates/cc_approvals.data', header=None)
columns = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel', 'Ethnicity', 'YearsEmployed',
           'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense', 'Citizen', 'ZipCode', 'Income', 'ApprovalStatus']
df.columns = columns
# Replace ? with NaN
df = df.replace('?', np.nan)
df.fillna(df.mean(), inplace=True)


# Because some columns do not contain numeric values different approach is needed to handle NaN
for col in df.columns.values:
    # Check if the column is of object type
    if df[col].dtypes == 'object':
        # fill with the most frequent value
        df = df.fillna(df[col].value_counts().index[0])

# Data Preprocessing
le = LabelEncoder()
for col in df.columns:
    # Compare if the dtype is object
    if df[col].dtypes == 'object':
        # Use LabelEncoder to do the numeric transformation
        df[col] = le.fit_transform(df[col])
df = df.drop(['DriversLicense', 'ZipCode'], axis=1)  # Driver license and zipcode are irrelevent
X, y = df.iloc[:, 0:13], df.iloc[:, 13]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predicting
y_pred = logreg.predict(rescaledX_test)
print("Accuracy of logistic regression classifier: ",
      logreg.score(rescaledX_test, y_test))
print(confusion_matrix(y_pred, y_test))

# Parameters tuning
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]
param_grid = dict(tol=tol, max_iter=max_iter)
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)
rescaledX = scaler.fit_transform(X)
grid_model_result = grid_model.fit(rescaledX, y)
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))
