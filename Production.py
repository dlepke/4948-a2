import pandas as pd
from pickle import load
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('og_train.csv', skiprows=1, names=(
	'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Applicant_Income',
	'Coapplicant_Income', 'Loan_amount', 'Term', 'Credit_History', 'Area', 'Status'
))

df['Status'].replace(['Y', 'N'], [1, 0], inplace=True)  # want 0 and 1 for classification model
df['Credit_History'].replace([0, 1], ['Y', 'N'], inplace=True)  # so get_dummies catches it

df = pd.get_dummies(df)

required_columns = [
	'Applicant_Income', 'Coapplicant_Income', 'Loan_amount',
	'Term', 'Married_No', 'Education_Graduate', 'Self_Employed_Yes',
	'Credit_History_N', 'Area_Semiurban', 'Area_Rural', 'Area_Urban'
]

columns = list(df.keys())

# ensure no missing dummy columns
for i in range(0, len(required_columns)):
	column_found = False
	for j in range(0, len(columns)):
		if columns[j] == required_columns[i]:
			column_found = True
	if not column_found:
		df[required_columns[i]] = 0

X = df[[
	'Applicant_Income', 'Coapplicant_Income', 'Loan_amount',
	'Term', 'Married_No', 'Education_Graduate', 'Self_Employed_Yes',
	'Credit_History_N', 'Area_Semiurban', 'Area_Rural', 'Area_Urban'
]]

y = df[['Status']]

imputer = KNNImputer()

X = imputer.fit_transform(X)

# X = pd.DataFrame(X, columns=required_columns)

with open('BinaryFolder/scaler.pkl', 'rb') as sf:
	scaler = load(sf)
	
with open('BinaryFolder/logistic.pkl', 'rb') as lf:
	logistic = load(lf)
	
with open('BinaryFolder/ann.pkl', 'rb') as af:
	ann = load(af)
	
with open('BinaryFolder/stacked.pkl', 'rb') as stf:
	stacked = load(stf)
	
X_scaled = scaler.transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=required_columns)

# X_scaled = X_scaled[[
# 	'Applicant_Income', 'Coapplicant_Income', 'Loan_amount',
# 	'Term', 'Married_No', 'Education_Graduate', 'Self_Employed_Yes',
# 	'Credit_History_N', 'Area_Semiurban', 'Area_Rural', 'Area_Urban'
# ]]

logistic_pred = logistic.predict(X_scaled)
ann_pred = ann.predict(X_scaled)

ann_predictions = []

# Convert continuous predictions to 0 or 1.
for i in range(0, len(ann_pred)):
	if ann_pred[i] > 0.5:
		ann_predictions.append(1)
	else:
		ann_predictions.append(0)

pred_df = pd.DataFrame()
pred_df['ann'] = ann_predictions
pred_df['logistic'] = logistic_pred

stacked_pred = stacked.predict(pred_df)

""" Not sure if by "must not reference the target column" you meant not assessing results,
	but this is here if you'd like to assess performance. """
# print(classification_report(y['Status'], stacked_pred))
# cm = pd.crosstab(y['Status'], stacked_pred, rownames=['Actual'], colnames=['Predicted'])
