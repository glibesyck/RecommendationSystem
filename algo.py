import os
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from surprise import Reader, Dataset, SVD
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split, PredefinedKFold, GridSearchCV

def predict_using_surprise(data, biased, k_fold):
    trainset, testset = train_test_split(data, test_size=.25)
    algo = SVD(biased = biased)
    algo.fit(trainset)
    predictions = algo.test(testset)

    print("Accuracy without kfold, non-biased")
    accuracy.rmse(predictions)

    algo = SVD(biased = True)
    algo.fit(trainset)
    predictions = algo.test(testset)

    print("Accuracy without kfold, biased")
    accuracy.rmse(predictions)

    if (k_fold):
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
    return algo


def search_wgrid (data, epochs:list, lr:list, reg:list, factors:list, biased:list):
    param_grid = {'n_epochs': epochs, 'lr_all': lr,
              'reg_all': reg, 'n_factors':factors, 'biased':biased}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    return gs

def scipy_svd(ratings):
    
    Ratings = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
    Rr = Ratings.values

    U, sigma, Vt = svds(Rr, k = 50)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    preds = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)

    print(preds.head())
    return U, sigma, Vt

if __name__ == "__main__":
    data = Dataset.load_builtin('ml-100k')
    ratings = pd.read_csv('ml-25m/ratings.csv')

    print(len(ratings))
    n_users = ratings.userId.unique().shape[0]
    n_movies = ratings.movieId.unique().shape[0]
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies))
    sparsity = round(1.0 - len(ratings) / float(n_users * n_movies), 3)
    print('The sparsity level of MovieLens25M dataset is ' +  str(sparsity * 100) + '%')