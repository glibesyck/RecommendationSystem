import numpy as np
import pandas as pd

n_iter = 100
regularization = 0.02
learning_rate = 0.005

user_ratings = {'1824' : 5, '12682': 5, '0': 4, '14253': 4.5,
 '9763' : 5, '817': 3, '2665': 4.5, '371': 4, '2427': 5, '29': 5} #inner_id : rating

df_items = pd.read_csv('obtained_data/items_factors.csv', header=None)
items = df_items.to_numpy()
df_bias = pd.read_csv('obtained_data/items_bias.csv', header=None)
bias = df_bias.to_numpy()

user_factors = np.random.normal(0, 1, (10))
for _ in range(n_iter):
    for id, rating in user_ratings.items():
        error = rating - user_factors @ items[int(id)]
        user_factors = user_factors + learning_rate*(error*items[int(id)] - regularization*user_factors)
all_ratings = items @ user_factors
for idx, elem in enumerate(all_ratings):
    elem += bias[idx]
all_ratings = sorted(enumerate(all_ratings), key=lambda i: i[1], reverse=True)
right = 0
recommendations = []
rec = []
for rating in all_ratings:
    if str(rating[0]) not in user_ratings.keys():
        recommendations.append(rating[0])
        right += 1
    if right == 5:
        break
df_films = pd.read_csv('ml-25m/movies.csv')
print(df_films[df_films['innerId'].isin(recommendations)])

