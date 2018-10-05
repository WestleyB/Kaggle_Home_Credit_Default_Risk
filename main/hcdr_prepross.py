import numpy as np
import pandas as pd
# import datetime as dt
import gc
import time
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import skew, kurtosis, iqr

from utils import memory_reduce


def hcdr_credit_card_balance():
	credit_card_balance = pd.read_csv('../data/credit_card_balance.csv')
	print('\t- credit_card_balance data shape: ', credit_card_balance.shape)
	return credit_card_balance


def hcdr_pos_cash_balance():
	pos_cash_balance = pd.read_csv('../data/POS_CASH_balance.csv')
	print('\t- POS_CASH_balance data shape: ', pos_cash_balance.shape)
	return 	pos_cash_balance


def hcdr_previous_application():
	previous_application = pd.read_csv('../data/previous_application.csv')
	print('\t- previous_application data shape: ', previous_application.shape)
	return previous_application


def hcdr_installments_payments():
	installments_payments = pd.read_csv('../data/installments_payments.csv')
	print('\t- installments_payments data shape: ', installments_payments.shape)
	return installments_payments


def hcdr_step_one(deep=3, strategy=None):
	if deep < 1:
		return pd.DataFrame()
	else:
		print('\n> Launch Step 1 Data Enrichment')
		df_1 = hcdr_pos_cash_balance()
		df_1 = memory_reduce(df_1)
		df_2 = hcdr_credit_card_balance()
		df_2 = memory_reduce(df_2)

		keys = ['MONTHS_BALANCE', 'NAME_CONTRACT_STATUS', 'SK_DPD', 'SK_DPD_DEF', 'SK_ID_CURR', 'SK_ID_PREV']
		dataframe = df_2.merge(df_1, on=keys, how='outer')
		print('\t- POS_CASH_balance + credit_card_balance data shape: ', dataframe.shape)

		# Clean up memory
		gc.enable()
		del df_2, df_1
		gc.collect()

		# Fill NaN values
		if strategy:
			dataframe['AMT_PAYMENT_CURRENT'].fillna(strategy, inplace=True)
			dataframe['CNT_DRAWINGS_POS_CURRENT'].fillna(strategy, inplace=True)
			dataframe['CNT_DRAWINGS_OTHER_CURRENT'].fillna(strategy, inplace=True)
			dataframe['CNT_DRAWINGS_ATM_CURRENT'].fillna(strategy, inplace=True)
			dataframe['AMT_DRAWINGS_POS_CURRENT'].fillna(strategy, inplace=True)
			dataframe['AMT_DRAWINGS_OTHER_CURRENT'].fillna(strategy, inplace=True)
			dataframe['AMT_DRAWINGS_ATM_CURRENT'].fillna(strategy, inplace=True)
			dataframe['CNT_INSTALMENT_MATURE_CUM'].fillna(strategy, inplace=True)
			dataframe['AMT_INST_MIN_REGULARITY'].fillna(strategy, inplace=True)
			dataframe['CNT_INSTALMENT'].fillna(strategy, inplace=True)
			dataframe['CNT_INSTALMENT_FUTURE'].fillna(strategy, inplace=True)

		dataframe = memory_reduce(dataframe)

		print('\n> End of Step 1 Data Enrichment')
		return dataframe


def hcdr_step_two(deep=3):
	dataframe = hcdr_step_one(deep=deep, strategy=0)
	if deep < 2:
		return dataframe
	else:
		print('\n> Launch Step 2 Data Enrichment')
		df = hcdr_previous_application()
		df = memory_reduce(df)

		keys = ['NAME_CONTRACT_STATUS', 'SK_ID_CURR', 'SK_ID_PREV']
		dataframe = dataframe.merge(df, on=keys, how='outer')
		print('\t- POS_CASH_balance + credit_card_balance + previous_application data shape: ', dataframe.shape)

		# Clean up memory
		gc.enable()
		del df
		gc.collect()

		dataframe = memory_reduce(dataframe)
		print('\n> End of Step 2 Data Enrichment')
		return dataframe


def hcdr_step_three(deep=3):
	dataframe = hcdr_step_two(deep=deep)
	if deep < 3:
		return dataframe
	else:
		print('\n> Launch Step 3 Data Enrichment')
		df = hcdr_installments_payments()
		df = memory_reduce(df)

		keys = ['SK_ID_CURR', 'SK_ID_PREV']
		dataframe = dataframe.merge(df, on=keys, how='outer')

		# Clean up memory
		gc.enable()
		del df
		gc.collect()

		dataframe = memory_reduce(dataframe)
		print('\n> End of Step 3 Data Enrichment')
		return dataframe


def hcdr_step_final(train, test, deep=3):
	if deep > 0:
		to_enrich = hcdr_step_three(deep=deep)
		print(to_enrich.shape)
	else:
		print('\n> No Enrichment')
		return train, test

	print('\n> Launch Final Step Data Enrichment')

	# Merge Bureau features into training dataframe
	train = train.merge(to_enrich, on='SK_ID_CURR', how='left')

	# Merge Bureau features into testing dataframe
	test = test.merge(to_enrich, on='SK_ID_CURR', how='left')

	# Clean up memory
	gc.enable()
	del to_enrich
	gc.collect()

	# Print out the new shapes
	print('\t- Training data enriched shape: ', train.shape)
	print('\t- Testing data enriched shape:  ', test.shape)

	print('\n> End Data Final Enrichment')
	return train, test


def hcdr_add_bureau_features(train, test, strategy='median'):
	print('\n> Launch Data Enrichment with Bureau')

	bureau_balance = pd.read_csv('../data/bureau_balance.csv')
	print('\t- bureau_balance data shape: ', bureau_balance.shape)
	bureau = pd.read_csv('../data/bureau.csv')
	print('\t- bureau data shape: ', bureau.shape)

	print('\t- Enrichment Step 1')

	bureau_balance = pd.get_dummies(bureau_balance)
	print('\t- bureau_balance with dummies data shape: ', bureau_balance.shape)

	# Create features with the right balance for each status
	for col in bureau_balance.columns:
		if 'STATUS' in col:
			bureau_balance[col] = bureau_balance[col] * bureau_balance['MONTHS_BALANCE']

	# if strategy:
	# 	print('\t- Imputer for handling missing values')

	# 	# imputer for handling missing values
	# 	imputer = Imputer(strategy=strategy)

	# 	# Need to impute missing values
	# 	bureau_balance = imputer.fit_transform(bureau_balance)

	# fillna for handling missing values
	bureau_balance.fillna(bureau_balance.median(), inplace=True)

	"""
	# TODO: Old Strategy building
	# Create Step1 Aggregation Strategy
	aggregation_strategy_step1 = {
		'MONTHS_BALANCE': [np.min, np.max, np.mean, np.sum, 'median', 'count', 'var', np.std, skew, kurtosis, iqr], 
		'STATUS_0': [np.min, np.max, np.mean, np.sum, 'median', 'var', np.std, skew, kurtosis, iqr],
		'STATUS_1': [np.min, np.max, np.mean, np.sum, 'median', 'var', np.std, skew, kurtosis, iqr],
		'STATUS_2': [np.min, np.max, np.mean, np.sum, 'median', 'var', np.std, skew, kurtosis, iqr],
		'STATUS_3': [np.min, np.max, np.mean, np.sum, 'median', 'var', np.std, skew, kurtosis, iqr],
		'STATUS_4': [np.min, np.max, np.mean, np.sum, 'median', 'var', np.std, skew, kurtosis, iqr],
		'STATUS_5': [np.min, np.max, np.mean, np.sum, 'median', 'var', np.std, skew, kurtosis, iqr],
		'STATUS_C': [np.min, np.max, np.mean, np.sum, 'median', 'var', np.std, skew, kurtosis, iqr],
		'STATUS_X': [np.min, np.max, np.mean, np.sum, 'median', 'var', np.std, skew, kurtosis, iqr]
		}
	"""

	# Create Step1 Aggregation Strategy
	# Functions tested : [np.min, np.max, np.mean, np.sum, 'median', 'count', 'var', np.std, skew, kurtosis, iqr]
	aggregation_strategy_step1 = dict()
	for col in bureau_balance.columns:
		if 'SK_ID' not in col:
			if 'MONTHS' in col:
				aggregation_strategy_step1[col] = [np.min, np.max, np.mean, np.sum, 'median', 'count', 'var', np.std]
			elif 'STATUS' in col:
				aggregation_strategy_step1[col] = [np.min, np.max, np.mean, np.sum, 'median', 'var', np.std]
			else:
				aggregation_strategy_step1[col] = [np.min, np.max, np.mean, np.sum, 'median', 'var']

	# Apply the aggregation strategy
	bureau_balance = bureau_balance.groupby(['SK_ID_BUREAU'], as_index=False).agg(aggregation_strategy_step1)

	# Rename the columns
	bureau_balance.columns = [col[0] + "_" + col[1] for col in bureau_balance.columns]
	bureau_balance.rename(columns={'SK_ID_BUREAU_':'SK_ID_BUREAU'}, inplace=True)
	print('\t- bureau_balance aggregated data shape: ', bureau_balance.shape)

	bureau_balance = memory_reduce(bureau_balance)

	print('\t- Enrichment Step 2')
	bureau = bureau.merge(bureau_balance, on='SK_ID_BUREAU', how='left')
	print('\t- bureau data consolidated shape: ', bureau.shape)

	# Create Dummies with categorials features
	bureau = pd.get_dummies(bureau)
	print('\t- bureau data consolidated with dummies shape: ', bureau.shape)

	# Create features with the right Amount Credit Sum
	for col in bureau.columns:
		if 'CREDIT_CURRENCY' in col:
			bureau[col] = bureau[col] * bureau['AMT_CREDIT_SUM']
		if 'CREDIT_ACTIVE' in col:
			bureau[col] = bureau[col] * bureau['AMT_CREDIT_SUM']


	# if strategy:
	# 	print('\t- Imputer for handling missing values')

	# 	# imputer for handling missing values
	# 	imputer = Imputer(strategy=strategy)

	# 	# Need to impute missing values
	# 	bureau = imputer.fit_transform(bureau)

	# fillna for handling missing values
	bureau.fillna(bureau.median(), inplace=True)

	# Create Step2 Aggregation Strategy
	# Functions tested : [np.min, np.max, np.mean, np.sum, 'median', 'count', 'var', np.std, skew, kurtosis, iqr]
	aggregation_strategy_step2 = dict()
	for col in bureau.columns:
		if 'SK_ID' not in col:
			if 'MONTHS' in col:
				aggregation_strategy_step2[col] = [np.min, np.max, np.mean, np.sum, 'median', 'count', 'var', np.std]
			elif 'STATUS' in col:
				aggregation_strategy_step2[col] = [np.min, np.max, np.mean, np.sum, 'median', 'var']
			elif 'AMT_CREDIT' in col:
				aggregation_strategy_step2[col] = [np.min, np.max, np.mean, np.sum, 'median', 'count', 'var', np.std]
			elif 'CREDIT' in col:
				aggregation_strategy_step2[col] = [np.min, np.max, np.mean, np.sum, 'median', 'var']
			else:
				aggregation_strategy_step2[col] = [np.min, np.max, np.mean, np.sum, 'median', 'var']


	bureau = bureau.groupby(['SK_ID_CURR'], as_index=False).agg(aggregation_strategy_step2)
	print('\t- bureau consolided aggregated data shape: ', bureau.shape)

	# Rename the columns
	bureau.columns = [col[0] + "_" + col[1] for col in bureau.columns]
	bureau.rename(columns={'SK_ID_BUREAU_':'SK_ID_BUREAU', 'SK_ID_CURR_':'SK_ID_CURR'}, inplace=True)

	bureau = memory_reduce(bureau)

	# Merge Bureau features into training dataframe
	train = train.merge(bureau, on='SK_ID_CURR', how='left')

	# Merge Bureau features into testing dataframe
	test = test.merge(bureau, on='SK_ID_CURR', how='left')

	# Print out the new shapes
	print('\t- Training data merged with Bureau features shape: ', train.shape)
	print('\t- Testing data merged with Bureau features shape:  ', test.shape)

	print('\n> End Data Enrichment with Bureau')
	return train, test




def hcdr_add_pos_cash_features(train, test):
	print('\n> Launch Data Enrichment with POS_CASH_balance')
	pos_cash_balance = pd.read_csv('../data/POS_CASH_balance.csv')
	print('\t- POS_CASH_balance data shape: ', pos_cash_balance.shape)

	pos_cash_balance = pd.get_dummies(pos_cash_balance)
	print('\t- POS_CASH_balance with dummies data shape: ', pos_cash_balance.shape)

	# fillna for handling missing values
	pos_cash_balance.fillna(pos_cash_balance.median(), inplace=True)

	aggregation_strategy = {
		'MONTHS_BALANCE': [np.min, np.max, np.mean, np.sum, 'median', 'count', 'var', np.std, skew, kurtosis, iqr],
		'CNT_INSTALMENT': [np.min, np.max, np.mean, 'median', 'var', np.std, skew, kurtosis, iqr],
		'CNT_INSTALMENT_FUTURE': [np.min, np.max, np.mean, 'median', 'var', np.std, skew, kurtosis, iqr],
		'SK_DPD': [np.min, np.max, np.mean, 'median', 'var', np.std, skew, kurtosis, iqr],
		'SK_DPD_DEF': [np.min, np.max, np.mean, 'median', 'var', np.std, skew, kurtosis, iqr],
		'SK_ID_PREV': ['nunique'],
		'NAME_CONTRACT_STATUS_Active': [np.sum],
		'NAME_CONTRACT_STATUS_Amortized debt': [np.sum],
		'NAME_CONTRACT_STATUS_Approved': [np.sum],
		'NAME_CONTRACT_STATUS_Canceled': [np.sum],
		'NAME_CONTRACT_STATUS_Completed': [np.sum],
		'NAME_CONTRACT_STATUS_Demand': [np.sum],
		'NAME_CONTRACT_STATUS_Returned to the store': [np.sum],
		'NAME_CONTRACT_STATUS_Signed': [np.sum],
		'NAME_CONTRACT_STATUS_XNA': [np.sum]
	}

	pos_cash_balance = pos_cash_balance.groupby(['SK_ID_CURR'], as_index=False).agg(aggregation_strategy)

	pos_cash_balance.columns = [col[0] + "_" + col[1] for col in pos_cash_balance.columns]
	pos_cash_balance.rename(columns={'SK_ID_CURR_':'SK_ID_CURR'}, inplace=True)

	print('\t- POS_CASH_balance aggregated data shape: ', pos_cash_balance.shape)

	pos_cash_balance = memory_reduce(pos_cash_balance)

	# Merge POS_CASH_balance features into training dataframe
	train = train.merge(pos_cash_balance, on='SK_ID_CURR', how='left')

	# Merge POS_CASH_balance features into testing dataframe
	test = test.merge(pos_cash_balance, on='SK_ID_CURR', how='left')

	# Print out the new shapes
	print('\t- Training data merged with POS_CASH_balance features shape: ', train.shape)
	print('\t- Testing data merged with POS_CASH_balance features shape:  ', test.shape)

	print('\n> End Train Enrichment with POS_CASH_balance')
	return train, test


def hcdr_anomalous_corrections(train, test):
	print('\n> Launch Anomalous Corrections')

	# Record correction
	train['DAYS_BIRTH'] = abs(train['DAYS_BIRTH'])
	test['DAYS_BIRTH'] = abs(test['DAYS_BIRTH'])

	# Create an anomalous flag column
	train['DAYS_EMPLOYED_ANOM'] = train["DAYS_EMPLOYED"] == 365243
	test['DAYS_EMPLOYED_ANOM'] = test["DAYS_EMPLOYED"] == 365243

	# Replace the anomalous values with nan
	train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
	test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

	print('\t- There are {} anomalies in the train data out of {} entries'.format(train["DAYS_EMPLOYED_ANOM"].sum(), len(train)))
	print('\t- There are {} anomalies in the test data out of {} entries'.format(test["DAYS_EMPLOYED_ANOM"].sum(), len(test)))

	print('\n> End of Anomalous Corrections')
	return train, test


def hcdr_add_custom_features(train, test):
	print('\n> Launch Add Custom Features')

	train['YEARS_BIRTH'] = train['DAYS_BIRTH'] / 365
	test['YEARS_BIRTH'] = test['DAYS_BIRTH'] / 365

	# Bin the age data
	train['YEARS_BINNED'] = pd.cut(train['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
	test['YEARS_BINNED'] = pd.cut(test['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

	# Add Domain Knowledge Features to train
	train['CREDIT_INCOME_PERCENT'] = train['AMT_CREDIT'] / train['AMT_INCOME_TOTAL']
	train['ANNUITY_INCOME_PERCENT'] = train['AMT_ANNUITY'] / train['AMT_INCOME_TOTAL']
	train['CREDIT_TERM'] = train['AMT_ANNUITY'] / train['AMT_CREDIT']
	train['DAYS_EMPLOYED_PERCENT'] = train['DAYS_EMPLOYED'] / train['DAYS_BIRTH']

	# Add Domain Knowledge Features to test
	test['CREDIT_INCOME_PERCENT'] = test['AMT_CREDIT'] / test['AMT_INCOME_TOTAL']
	test['ANNUITY_INCOME_PERCENT'] = test['AMT_ANNUITY'] / test['AMT_INCOME_TOTAL']
	test['CREDIT_TERM'] = test['AMT_ANNUITY'] / test['AMT_CREDIT']
	test['DAYS_EMPLOYED_PERCENT'] = test['DAYS_EMPLOYED'] / test['DAYS_BIRTH']

	print('\n> Custom Features Added')
	return train, test


def hcdr_add_polynomial_features(train, test, columns, label_column, keyjoin, degree=3, strategy='median'):
	print('\n> Launch Add Polynomial Features')

	# Make a new dataframe for polynomial features
	_columns = columns.copy()
	_columns.append(label_column)

	poly_features = train[columns]
	poly_features_test = test[columns]
	# poly_target = poly_features[label_column]

	if strategy:
		print('\t- Imputer for handling missing values')

		# imputer for handling missing values
		imputer = Imputer(strategy=strategy)

		# poly_features = poly_features.drop(columns=[label_column])

		# Need to impute missing values
		poly_features = imputer.fit_transform(poly_features)
		poly_features_test = imputer.transform(poly_features_test)
	                 
	# Create the polynomial object with specified degree
	poly_transformer = PolynomialFeatures(degree=degree)

	# Train the polynomial features
	poly_transformer.fit(poly_features)

	# Transform the features
	poly_features = poly_transformer.transform(poly_features)
	poly_features_test = poly_transformer.transform(poly_features_test)

	print('\t- Polynomial Features train shape: ', poly_features.shape)
	print('\t- Polynomial Features test shape: ', poly_features_test.shape)

	# Create a dataframe of the features 
	poly_features = pd.DataFrame(poly_features, columns=poly_transformer.get_feature_names(columns))
	poly_features_test = pd.DataFrame(poly_features_test, columns=poly_transformer.get_feature_names(columns))

	# Add in the target
	# poly_features[label_column] = poly_target

	# Find the correlations with the target
	# poly_corrs = poly_features.corr()['TARGET'].sort_values()

	# Display most negative and most positive
	# print(poly_corrs.head(10))
	# print(poly_corrs.tail(5))


	# Merge polynomial features into training dataframe
	poly_features[keyjoin] = train[keyjoin]
	train_poly = train.merge(poly_features, on=keyjoin, how='left')

	# Merge polnomial features into testing dataframe
	poly_features_test[keyjoin] = test[keyjoin]
	test_poly = test.merge(poly_features_test, on=keyjoin, how='left')

	# Print out the new shapes
	print('\t- Training data with polynomial features shape: ', train_poly.shape)
	print('\t- Testing data with polynomial features shape:  ', test_poly.shape)

	return train_poly, test_poly


def hcdr_features_encoding(train, test):
	print('\n> Launch Features Encoding')
	# Create a label encoder object
	le = LabelEncoder()
	le_count = 0
	cat_features = []

	# Iterate through the columns
	for col in train:
	    if train[col].dtype == 'object':
	        # If 2 or fewer unique categories
	        if len(list(train[col].unique())) <= 2:
	            # Train on the training data
	            le.fit(train[col])
	            # Transform both training and testing data
	            train[col] = le.transform(train[col])
	            test[col] = le.transform(test[col])
	            cat_features.append(col)

	            # Keep track of how many columns were label encoded
	            le_count += 1
	            
	print('\t- %d columns were label encoded.' % le_count)

	# one-hot encoding of categorical variables
	train = pd.get_dummies(train)
	test = pd.get_dummies(test)

	print('\t- Training Features shape: ', train.shape)
	print('\t- Testing Features shape: ', test.shape)
	
	print('\n> End of Features Encoding')
	return train, test, cat_features
