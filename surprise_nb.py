import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
import os
import surprise
'''
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.25)
algo = SVD(biased = False)
algo.fit(trainset)
predictions = algo.test(testset)

print("Accuracy without kfold, non-biased")
accuracy.rmse(predictions)

lgo = SVD(biased = True)
algo.fit(trainset)
predictions = algo.test(testset)

print("Accuracy without kfold, biased")
accuracy.rmse(predictions)


files_dir = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/')
reader = Reader('ml-100k')
train_file = files_dir + 'u%d.base'
test_file = files_dir + 'u%d.test'
folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

data = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()

print("K-FOLD validation")
for trainset, testset in pkf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)
'''
print("Can use custom parameters")
data = Dataset.load_builtin('ml-100k')

param_grid = {'n_epochs': [10], 'lr_all': [0.005],
              'reg_all': [0.4], 'n_factors':[10], 'biased':[True]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)


gs.fit(data)
results_df = pd.DataFrame.from_dict(gs.cv_results)
print(results_df)
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

algo = gs.best_estimator['rmse']
algo.fit(data.build_full_trainset())

