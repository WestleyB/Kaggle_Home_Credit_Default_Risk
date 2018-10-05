import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from datetime import datetime
import time
import gc

from utils import memory_reduce
from hcdr_prepross import hcdr_add_custom_features, hcdr_add_polynomial_features, hcdr_anomalous_corrections,\
hcdr_features_encoding, hcdr_add_pos_cash_features, hcdr_add_bureau_features, hcdr_step_final
from hcdr_model import model
from hcdr_display import plot_features_importance, display_correlation
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hcdr_hyperparams import hyper_params


def data_loader():
	print('\n> Loading data...')
	app_train = pd.read_csv('../data/application_train.csv')
	print('\t- application_train data shape: ', app_train.shape)

	app_test = pd.read_csv('../data/application_test.csv')
	print('\t- application_test data shape: ', app_test.shape)

	template_submission = pd.read_csv('../data/sample_submission.csv')
	print('\t- sample_submission data shape: ', template_submission.shape)

	print('\n> Data loaded')
	return app_train, app_test, template_submission


if __name__ == "__main__":
	print('\n> Launch Preprocessing')
	start = time.time()

	app_train, app_test, template_submission = data_loader()

	app_train = memory_reduce(app_train)
	app_test = memory_reduce(app_test)

	app_train, app_test = hcdr_anomalous_corrections(app_train, app_test)

	app_train, app_test = hcdr_add_custom_features(app_train, app_test)

	# app_train, app_test = hcdr_add_pos_cash_features(app_train, app_test)

	# app_train, app_test = hcdr_add_bureau_features(app_train, app_test)

	app_train, app_test = hcdr_step_final(app_train, app_test)

	train_corrs = display_correlation(app_train, label='TARGET')

	print('\t- Training data shape: ', app_train.shape)
	print('\t- Testing data shape:  ', app_test.shape)

	columns_for_poly = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']
	app_train, app_test = hcdr_add_polynomial_features(app_train, app_test, columns=columns_for_poly, label_column='TARGET', keyjoin='SK_ID_CURR')

	app_train, app_test, cat_features = hcdr_features_encoding(app_train, app_test)

	app_train = memory_reduce(app_train)
	app_test = memory_reduce(app_test)

	app_train_corrs = display_correlation(app_train, label='TARGET')

	print('\n> Check Data Size')
	print('\t- Training data shape: ', app_train.shape)
	print('\t- Testing data shape:  ', app_test.shape)

	print('\n> Save Processed Data')
	app_train.to_csv('../data_processed/application_train_processed_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')
	app_test.to_csv('../data_processed/application_test_processed__{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')
	
	print('\n> End Preprocessing in {}'.format(time.time() - start))


	# Define spaces for hyperparameters
	# space4knn = {
	# 	'learning_rate': hp.uniform('learning_rate', 0.02, 0.03),
	# 	# 'n_estimators': hp.choice('n_estimators', range(8000, 20000, 1000)),
	# 	# 'subsample': hp.uniform('subsample', 0.5, 1.0),
	# 	# 'reg_alpha': hp.uniform('reg_alpha', 0.0, 2.0),
	# 	# 'reg_lambda': hp.uniform('reg_lambda', 0.0, 2.0),

	# 	# 'base_score': hp.choice('base_score', [0.0122520153048]),
	# 	# 'max_depth': hp.choice('max_depth', range(6, 13)),
	# 	# 'min_child_weight': hp.uniform('min_child_weight', 0, 2),
	# 	# 'gamma': hp.uniform('gamma', 0.0, 5.0),
	# 	# 'objective': hp.choice('objective', ['reg:linear']),
	# 	'seed': hp.choice('seed', [42])
	# 	}

	# best_params = hyper_params(app_train, space4knn)

	print('\n> Launch Model Training...')
	fit_start = time.time()
	submission, feature_importances, metrics = model(app_train, app_test)
	print('\n> End of Model Training in {}'.format(time.time() - fit_start))

	print('\n> Saving predictions...')
	assert submission.shape == template_submission.shape
	submission.to_csv('../submissions/hcdr_lightgbm_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')
	print('\nPrediction available !!!')
	
	print('\n> Saving feature_importances...')
	feature_importances.to_csv('../features_importances_graph/feature_importances_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')

	print('\n> Saving metrics...')
	metrics.to_csv('../features_importances_graph/metrics_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')

	print('\n> Display Features Importance')
	feature_importances_sorted = plot_features_importance(feature_importances, n=app_train.shape[1])
	feature_importances_sorted = plot_features_importance(feature_importances, n=100)

	print('\n> ...The End')
