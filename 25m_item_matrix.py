"""
Module which implements the search of the values of latent factors for big dataset - 25M MovieLens
"""
from surprise import Dataset
from surprise import Reader
from surprise import SVD
import numpy as np
import pandas as pd

#preparing data
reader = Reader(line_format='user item rating timestamp', sep=',')
data = Dataset.load_from_file('ml-25m/ratings.csv', reader=reader)
trainset = data.build_full_trainset()
#preparing algo
algo = SVD(n_factors = 10, n_epochs = 10, biased = True, reg_all = 0.02, lr_all = 0.005)
algo.fit(trainset)
#writing item-factors matrix into .csv
np.savetxt('obtained_data/items_factors.csv', algo.qi, delimiter=",")
#writing item bias matrix into .csv
np.savetxt('obtained_data/items_bias.csv', algo.bi, delimiter=",")
#indexation is inner, we need to translate it to raw (which is in file)
df = pd.read_csv('ml-25m/movies.csv')
for index, row in df.iterrows():
    try:
        inner = trainset.to_inner_iid(str(row['movieId']))
        df.at[index, 'innerId'] = inner
    except ValueError:
        df.at[index, 'innerId'] = -1
df.to_csv('ml-25m/moviess.csv', index=False)