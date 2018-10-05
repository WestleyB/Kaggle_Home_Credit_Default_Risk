import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import lightgbm as lgb


def hyperopt_train_test(model, x_train, y_train):
    return cross_val_score(model, x_train, y_train).mean()


def f(params):
	model = lgb.LGBMClassifier(params)
	# acc = hyperopt_train_test(model)
	acc = cross_val_score(model, x_train, y_train, scoring='roc_auc').mean()
	return {'loss': acc, 'status': STATUS_OK}


"""
space4knn = {
	'learning_rate': hp.uniform('learning_rate', 0.02, 0.04),
	'max_depth': hp.choice('max_depth', range(6, 13)),
	'min_child_weight': hp.uniform('min_child_weight', 0, 2),
	'gamma': hp.uniform('gamma', 0.0, 5.0),
	'subsample': hp.uniform('subsample', 0.5, 1.0),
	'objective': hp.choice('objective', ['reg:linear']),
	'n_estimators': hp.choice('n_estimators', range(500, 3000, 100)),
	'reg_alpha': hp.uniform('reg_alpha', 0.0, 2.0),
	'reg_lambda': hp.uniform('reg_lambda', 0.0, 2.0),
	'base_score': hp.choice('base_score', [0.0122520153048]),
	'seed': hp.choice('seed', [42])
}
"""

def hyper_params(x_train, space4knn):
	print('\n> Launching Tuning Hyperparameters...')
	# Remove the ids and target
	y_train = x_train['TARGET']
	x_train = x_train.drop(columns = ['SK_ID_CURR', 'TARGET'])

	trials = Trials()
	best = fmin(fn=f,  space=space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
	print('\t- best parameters :')
	print(best)
	print('\n> End of Tuning Hyperparameters')
	return best


def run_trials():

	trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
	max_trials = 5  # initial max_trials. put something small to not have to wait

	try:  # try to load an already saved trials object, and increase the max
		trials = pickle.load(open("lgbm_model.hyperopt", "rb"))
		print("\t- Found saved Trials! Loading...")
		max_trials = len(trials.trials) + trials_step
		print("\t- Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
	except:  # create a new trials object and start searching
		trials = Trials()

	best = fmin(fn=f, space=model_space, algo=tpe.suggest, max_evals=max_trials, trials=trials)
	print("\t- Best:", best)

	# save the trials object
	with open(_model + ".hyperopt", "wb") as f:
		pickle.dump(trials, f)


# loop indefinitely and stop whenever you like
# while True:
# 	run_trials()
