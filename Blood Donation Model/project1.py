# EDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('E://Python/Blood Donation Model/transfusion.csv')
columns_name = ('Months since last Donation', 'Number of Donations',
                'Total Volume Donated(c.c)', 'Months since first Donation', 'Made Donation in 2007')
df.columns = columns_name


# Overview of Data
plt.figure(figsize=(15, 7))
plt.style.use('ggplot')

plt.subplot(2, 2, 1)
sns.distplot(df[df['Made Donation in 2007'].values == 0]
             ['Months since last Donation'], color='Blue')
sns.distplot(df[df['Made Donation in 2007'].values == 1]['Months since last Donation'], color='Red')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
sns.distplot(df[df['Made Donation in 2007'].values == 0]['Number of Donations'], color='Blue')
sns.distplot(df[df['Made Donation in 2007'].values == 1]['Number of Donations'], color='Red')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
sns.distplot(df[df['Made Donation in 2007'].values == 0]['Total Volume Donated(c.c)'], color='Blue')
sns.distplot(df[df['Made Donation in 2007'].values == 1]['Total Volume Donated(c.c)'], color='Red')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
sns.distplot(df[df['Made Donation in 2007'].values == 0]
             ['Months since first Donation'], color='Blue')
sns.distplot(df[df['Made Donation in 2007'].values == 1]
             ['Months since first Donation'], color='Red')
plt.ylabel('Frequency')
plt.tight_layout()

# Relationship of variables
sns.pairplot(df.iloc[:, 0:4], diag_kind='kde')
plt.tight_layout()

# Perfect corr Total Number of Donations and Total Volume Donated or not ?
df_corr = df.iloc[:, 0:4].corr()
plt.figure(figsize=(30, 15))
corr = sns.heatmap(df_corr, annot=True)
plt.tight_layout()


# Removing Perfect corr
del df['Total Volume Donated(c.c)']
df['Waiting Time'] = (df['Months since first Donation'] -
                      df['Months since last Donation'])/df['Number of Donations']
df_corr2 = df.iloc[:, [0, 1, 2, 4]].corr()
corr2 = sns.heatmap(df_corr2, annot=True)


# Making train_test_split
X = df.iloc[:, [0, 1, 2, 4]]
y = df[['Made Donation in 2007']]
print(type(X), type(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(df.head())
print(X.head())
print(y.head())
print(X_train.isnull().sum(),
      y_train.isnull().sum(),
      X_test.isnull().sum(),
      y_test.isnull().sum())

# Scaling numeric features
scaler = StandardScaler()
numericfeatures = ['Months since last Donation', 'Number of Donations',
                   'Waiting Time', 'Months since first Donation']
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numericfeatures]))
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test[numericfeatures]))

# Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred1 = linreg.predict(X_test)
print("""[LinearRegression]\n Entropy loss: {}""".format(log_loss(y_test, y_pred1)))

# Logistic Regression CV
logreg = LogisticRegressionCV(cv=3, random_state=42, scoring='neg_log_loss')
logreg.fit(X_train, y_train.values.ravel())
y_pred2a = logreg.predict(X_test)
y_pred2b = logreg.predict_proba(X_test)
print("""[LogisticRegression]\n Test accuracy: \n{}""".format(
    classification_report(y_test, y_pred2a)))
print("""[LogisticRegression]\n Entropy loss: {}""".format(log_loss(y_test, y_pred2b)))

# Random Forest
ranfor = RandomForestClassifier(random_state=42)
params = {'n_estimators': [30, 60, 90],
          'max_features': [1, 2, 3]}
rf = GridSearchCV(estimator=ranfor, param_grid=params, cv=3, scoring='neg_log_loss')
rf.fit(X_train_scaled, y_train.values.ravel())
y_pred3a = rf.predict(X_test)
y_pred3b = rf.predict_proba(X_test)
print("""[RandomForestClassifier]\n Test accuracy: \n{}""".format(
    classification_report(y_test, y_pred3a)))
print("""[RandomForestClassifier]\n Entropy loss: {}""".format(log_loss(y_test, y_pred3b)))

# Prediction
X_total = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)])
y_total = pd.concat([y_train, y_test]).values
final_model = logreg.fit(X_total, y_total.ravel())
y_prediction = final_model.predict_proba(X_test)[:, 1]
print(X_total.shape, y_total.shape, y_prediction.shape, X_test.shape)
print(y_prediction.mean(), df['Made Donation in 2007'].values.mean())
print("""[Final Model]\n Entropy loss:{}""".format(log_loss(y_test, y_prediction)))

# Submission
y_prediction = y_prediction[:-25]
Submission = pd.read_csv(
    'E://Python/Blood Donation Model/Warm_Up_Predict_Blood_Donations_-_Submission_Format.csv', index_col=0)
Submission['Made Donation in March 2007'] = y_prediction
Submission.to_csv('test_submission.csv')
