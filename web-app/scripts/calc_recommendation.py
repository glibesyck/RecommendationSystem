import numpy as np
import pandas as pd

DF_FILMS = pd.read_csv('ml-25m/movies.csv')
N_ITER = 100
REGULARIZATION = 0.02
LEARNING_RATE = 0.005

def recommend(user_ratings: dict):
    df_items = pd.read_csv('obtained_data/items_factors.csv', header=None)
    items = df_items.to_numpy()
    df_bias = pd.read_csv('obtained_data/items_bias.csv', header=None)
    bias = df_bias.to_numpy()

    user_factors = np.random.normal(0, 1, (10))
    for _ in range(N_ITER):
        for id, rating in user_ratings.items():
            error = rating - user_factors @ items[int(id)]
            user_factors = user_factors + LEARNING_RATE*(error*items[int(id)] - REGULARIZATION*user_factors)
    all_ratings = items @ user_factors
    for idx, elem in enumerate(all_ratings):
        elem += bias[idx]
    all_ratings = sorted(enumerate(all_ratings), key=lambda i: i[1], reverse=True)
    right = 0
    recommendations = []
    for rating in all_ratings:
        if str(rating[0]) not in user_ratings.keys():
            recommendations.append(rating[0])
            right += 1
        if right == 5:
            break
    
    df2 = DF_FILMS[DF_FILMS['innerId'].isin(recommendations)]
    df2=df2[['title', 'tmdbId']]
    return df2.values.tolist()


if __name__ == '__main__': 
    user_ratings = {'1824' : 5, '12682': 5, '0': 4, '14253': 4.5,
    '9763' : 5, '817': 3, '2665': 4.5, '371': 4, '2427': 5, '29': 5} #inner_id : rating

    print(recommend(user_ratings))
