"""
Requirements:
- stacked model
	- neural net
	- something else
- try scaling
- kfold
- load binaries into folder
- knnimputer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('train.csv', skiprows=1, names=(
	'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Applicant_Income',
	'Coapplicant_Income', 'Loan_amount', 'Term', 'Credit_History', 'Area', 'Status'
))

df['Status'].replace(['Y', 'N'], [1, 0], inplace=True)  # want 0 and 1 for classification model
df['Credit_History'].replace([0, 1], ['Y', 'N'], inplace=True)  # so get_dummies catches it

df = pd.get_dummies(df)

X = df.copy()
del X['Status']

y = df[['Status']]

imputer = KNNImputer(n_neighbors=5)

columns = X.columns

X = imputer.fit_transform(X)

X = pd.DataFrame(X, columns=columns)

""" Optimized for logistic model. """
# X = X[[
# 	'Applicant_Income',
# 	'Coapplicant_Income',
# 	'Loan_amount',
# 	'Married_No',
# 	'Dependents_2',
# 	'Education_Graduate',
# 	'Self_Employed_Yes',
# 	'Credit_History_N',
# 	'Credit_History_Y',
# 	'Area_Semiurban'
# ]]
#
# columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=columns)

"""Logistic model"""
logistic = LogisticRegression(fit_intercept=True, solver='liblinear')

# rfe = RFE(logistic)
#
# rfe = rfe.fit(X_train_scaled, y_train)

# for i in range(len(X_train_scaled.keys())):
# 	if rfe.support_[i]:
# 		print(X.keys()[i])

logistic.fit(X_train_scaled, y_train)

logistic_pred = logistic.predict(X_test_scaled)

cm = pd.crosstab(y_test['Status'], logistic_pred, rownames=['Actual'], colnames=['Predicted'])
print(cm)
print(classification_report(y_test, logistic_pred))

"""EDA graphs"""
# default = df[df['Status'] == 1]
# no_default = df[df['Status'] == 0]

# plt.figure(figsize=(50, 8))
#
# for i in range(len(df.columns)):
# 	plt.subplot(2, 11, i + 1)
# 	col = df.columns[i]
# 	plt.hist([default[col], no_default[col]], rwidth=0.8)
# 	plt.legend(['Default', 'No default'])
# 	plt.title(col + ' vs Default Status')
#
# plt.show()
