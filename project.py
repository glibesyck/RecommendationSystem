import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from surprise import Reader, Dataset, SVD
import surprise
from surprise.model_selection import cross_validate


data = Dataset.load_builtin('ml-100k')
ratings = pd.read_csv('ml-25m/ratings.csv')

print(len(ratings))
n_users = ratings.userId.unique().shape[0]
n_movies = ratings.movieId.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies))
sparsity = round(1.0 - len(ratings) / float(n_users * n_movies), 3)
print('The sparsity level of MovieLens25M dataset is ' +  str(sparsity * 100) + '%')


# Ratings = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
# Rr = Ratings.values
# # user_ratings_mean = np.mean(Rr, axis = 1)
# # Ratings_demeaned = Rr - user_ratings_mean.reshape(-1, 1)

# U, sigma, Vt = svds(Rr, k = 50)
# sigma = np.diag(sigma)
# all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
# preds = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)

# print(preds.head())

# reader = Reader()

# # Load ratings dataset with Dataset library
# data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# # Compute the RMSE of the SVD algorithm.
# algo = SVD()

# # Run 5-fold cross-validation and then print results
# print(cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True))
