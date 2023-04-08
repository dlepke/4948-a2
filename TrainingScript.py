"""
- load binaries into folder
"""

import pandas as pd
import numpy as np
from pickle import dump
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics

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
# print(columns)

X = imputer.fit_transform(X)

X = pd.DataFrame(X, columns=columns)

""" Optimized for logistic model. """
X = X[[
	'Applicant_Income',
	'Coapplicant_Income',
	'Loan_amount',
	'Term',
	'Married_No',
	# 'Dependents_0',
	# 'Dependents_1',
	# 'Dependents_2',
	# 'Dependents_3+',
	'Education_Graduate',
	'Self_Employed_Yes',
	'Credit_History_N',
	'Area_Semiurban',
	'Area_Rural',
	'Area_Urban'
]]
NUM_COLS = len(X.columns)

columns = X.columns

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
#
# print(X_train.head(), X_test.head(), X_val.head())

# X_train = X[:400]
# X_test = X[400:500]
# X_train = X[:500]
# X_val = X[500:]

# y_train = y[:400]
# y_test = y[400:500]
# y_train = y[:500]
# y_val = y[500:]

# print(X_train.head(), X_test.head(), X_val.head())

stacked_accuracies = []
stacked_precisions = []

logistic_accuracies = []
logistic_precisions = []

ann_accuracies = []
ann_precisions = []

kfold = KFold(n_splits=2)

""" train trains base models,
	test produces training data for stacked model,
	val tests stacked model
	test.csv contains separate test data for production testing """
for train_index, test_index in kfold.split(X):
	X_train = X.iloc[train_index, :]
	y_train = y.iloc[train_index, :]
	X_test = X.iloc[test_index, :]
	for test_index, val_index in kfold.split(X_test):
		X_test = X.iloc[test_index, :]
		y_test = y.iloc[test_index, :]
		X_val = X.iloc[val_index, :]
		y_val = y.iloc[val_index, :]
		
		scaler = MinMaxScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_test_scaled = scaler.transform(X_test)
		X_val_scaled = scaler.transform(X_val)
		
		dump(scaler, open('BinaryFolder/scaler.pkl', 'wb'))
		
		X_train_scaled = pd.DataFrame(X_train_scaled, columns=columns)
		X_test_scaled = pd.DataFrame(X_test_scaled, columns=columns)
		X_val_scaled = pd.DataFrame(X_val_scaled, columns=columns)
		
		""" Logistic model """
		logistic = LogisticRegression(fit_intercept=True, solver='liblinear')
		
		""" RFE for feature selection - used results in combination with EDA observations """
		# rfe = RFE(logistic)
		#
		# rfe = rfe.fit(X_train_scaled, y_train)
		#
		# for i in range(len(X_train_scaled.keys())):
		# 	if rfe.support_[i]:
		# 		print(X.keys()[i])
		
		logistic.fit(X_train_scaled, y_train)
		
		logistic_pred = logistic.predict(X_test_scaled)
		
		cm = pd.crosstab(y_test['Status'], logistic_pred, rownames=['Actual'], colnames=['Predicted'])
		print(cm)
		print(classification_report(y_test, logistic_pred))

		""" ANN with RandomizedSearchCV """
		def create_ann(num_neurons=10, initializer='uniform', activation='sigmoid', num_hidden_layers=3, learning_rate=0.005):
			model = Sequential()
			
			model.add(Dense(10, activation='sigmoid', input_shape=(NUM_COLS,)))
			
			for j in range(num_hidden_layers):
				model.add(Dense(num_neurons, activation=activation, kernel_initializer=initializer))
			
			model.add(Dense(1, activation='sigmoid'))
			
			opt = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)
			
			model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
			
			return model

		""" Optimizing parameters for ANN - selected options shown below """
		# params = {
		# 	'activation': ['relu', 'sigmoid', 'softmax', 'softplus'],
		# 	'num_neurons': [10, 25, 50, 100],
		# 	'num_hidden_layers': [3, 4, 5],
		# 	'initializer': ['normal', 'uniform', 'zero', 'he_normal'],
		# 	'learning_rate': [0.0001, 0.0005, 0.001, 0.005]
		# }
		
		params = {
			'activation': ['softplus'],
			'num_neurons': [50],
			'num_hidden_layers': [4],
			'initializer': ['he_normal'],
			'learning_rate': [0.0001]
		}
		
		""" Also tried randomized grid search. """
		# ann = KerasClassifier(build_fn=create_ann, epochs=500, batch_size=10, verbose=1)
		
		# grid = RandomizedSearchCV(ann, param_distributions=params, cv=3)
		#
		# grid_result = grid.fit(X_train_scaled, y_train)
		#
		# # summarize results
		# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
		# means = grid_result.cv_results_['mean_test_score']
		# stds = grid_result.cv_results_['std_test_score']
		# params = grid_result.cv_results_['params']
		#
		# for mean, stdev, param in zip(means, stds, params):
		# 	print("%f (%f) with: %r" % (mean, stdev, param))
		#
		# predictions = grid.best_estimator_.predict(X_test)

		# recording the best accuracy etc. for manual grid search
		best_acc = [{
			'accuracy': 0,
			'conditions': 'start'
		}]

		for activator in params['activation']:
			for num_neurons in params['num_neurons']:
				for hidden_layers in params['num_hidden_layers']:
					for initializer in params['initializer']:
						for learning_rate in params['learning_rate']:
							print("current conditions: ", activator, num_neurons, hidden_layers, initializer, learning_rate)
							ann = create_ann(
								num_neurons=num_neurons, activation=activator, num_hidden_layers=hidden_layers,
								initializer=initializer, learning_rate=learning_rate)
							
							es = EarlyStopping(monitor='loss', patience=20)
							mc = ModelCheckpoint('best_model.h5', monitor="val_accuracy", mode='max', verbose=1)
							
							ann.fit(X_train_scaled, y_train, epochs=500, verbose=1, callbacks=[es, mc])
							
							loss, acc = ann.evaluate(X_test_scaled, y_test, verbose=0)
							print('Test accuracy: %.3f' % acc)
							
							if acc > best_acc[-1]['accuracy']:
								best_acc.append({
									'accuracy': acc,
									'conditions': {
										'activator': activator,
										'num_neurons': num_neurons,
										'hidden_layers': hidden_layers,
										'initializer': initializer,
										'learning_rate': learning_rate
									}
								})
								# print("new best accuracy: ", acc)
								# print("current conditions: ", activator, num_neurons, hidden_layers, initializer, learning_rate)
							
							ann_preds = ann.predict(X_test_scaled)
							
							ann_predictions = []
							
							# Convert continuous predictions to 0 or 1.
							for i in range(0, len(ann_preds)):
								if ann_preds[i] > 0.5:
									ann_predictions.append(1)
								else:
									ann_predictions.append(0)
							
							print(classification_report(y_test, ann_predictions))
					
		for item in best_acc:
			print("accuracy: ", item['accuracy'])
			print("conditions: ", item['conditions'])
		
		""" Stacked model feat. logistic model and ANN """
		
		stacked = LogisticRegression(fit_intercept=True, solver='liblinear')
		
		test_df = pd.DataFrame()
		
		test_df['ann'] = ann_predictions
		test_df['logistic'] = logistic_pred
		target = np.array(y_test['Status'])
		
		# print(test_df.head(30))
		
		stacked.fit(test_df, target)
		
		logistic_val = logistic.predict(X_val_scaled)
		ann_val = ann.predict(X_val_scaled)
		
		ann_val_predictions = []

		# Convert continuous predictions to 0 or 1.
		for i in range(0, len(ann_val)):
			if ann_val[i] > 0.5:
				ann_val_predictions.append(1)
			else:
				ann_val_predictions.append(0)
		
		val_df = pd.DataFrame()
		
		val_df['ann'] = ann_val_predictions
		val_df['logistic'] = logistic_val
		
		# print(val_df.head(30))
		
		stacked_predictions = stacked.predict(val_df)
		
		stacked_accuracies.append(metrics.accuracy_score(y_val['Status'], stacked_predictions))
		stacked_precisions.append(metrics.precision_score(y_val['Status'], stacked_predictions))
		
		ann_accuracies.append(metrics.accuracy_score(y_val['Status'], ann_val_predictions))
		ann_precisions.append(metrics.precision_score(y_val['Status'], ann_val_predictions))
		
		logistic_accuracies.append(metrics.accuracy_score(y_val['Status'], logistic_val))
		logistic_precisions.append(metrics.precision_score(y_val['Status'], logistic_val))
		
		cm = pd.crosstab(y_val['Status'], stacked_predictions, rownames=['Actual'], colnames=['Predicted'])
		print(cm)
		print(classification_report(y_val['Status'], stacked_predictions))
		

dump(logistic, open('BinaryFolder/logistic.pkl', 'wb'))
dump(ann, open('BinaryFolder/ann.pkl', 'wb'))
dump(stacked, open('BinaryFolder/stacked.pkl', 'wb'))

print("logistic model:")
print(np.mean(logistic_accuracies))
print(np.std(logistic_accuracies))
print(np.mean(logistic_precisions))
print(np.std(logistic_precisions))

print("ann model:")
print(np.mean(ann_accuracies))
print(np.std(ann_accuracies))
print(np.mean(ann_precisions))
print(np.std(ann_precisions))

print("stacked model:")
print(np.mean(stacked_accuracies))
print(np.std(stacked_accuracies))
print(np.mean(stacked_precisions))
print(np.std(stacked_precisions))


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
