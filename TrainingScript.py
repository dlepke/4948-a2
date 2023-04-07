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
from sklearn.model_selection import KFold

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

print(df.head())
print(df.dtypes)

X = df.copy()
del X['Status']

y = df[['Status']]

imputer = KNNImputer(n_neighbors=5)

columns = X.columns

X = imputer.fit_transform(X)

X = pd.DataFrame(X, columns=columns)

# print(X.describe())

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
