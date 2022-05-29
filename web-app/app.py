from unittest import result
from flask import Flask, render_template, request, url_for, send_file
from matplotlib.style import use
import requests
import numpy as np
import pandas as pd


app = Flask(__name__)

chosen_films=[]
user_ratings={}
film_names = []
film_grades = []
BASE_URL = 'https://api.themoviedb.org/3/movie/'
API_KEY ='?api_key=523e924d1ef2f06415de79b602cc54d6'
BASE_IMG_URL = 'https://image.tmdb.org/t/p/w500'
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

def construct_ratings():
    global film_names
    global user_ratings
    global film_grades
    for index, movie_name in enumerate(film_names):
        user_ratings[DF_FILMS[DF_FILMS['title']==movie_name]['innerId'].values[0]] = film_grades[index]

@app.route('/', methods = ['GET', 'POST'])
def index():
    global chosen_films
    global film_names
    global film_grades
    global user_ratings
    button_available = False
    if request.method == 'POST':
        movie_name = request.form.get("movie")
        grade = float(request.form.get("grade"))
        if(not movie_name in film_names and movie_name != ''):
            chosen_films.append(movie_name+" - "+str(grade))
            film_names.append(movie_name)
            film_grades.append(grade)
    else:
        user_ratings = {}
        chosen_films=[]
    if len(chosen_films) >= 10:
        button_available = True
    return render_template("index.html", content=chosen_films, button_av = button_available, movies=DF_FILMS['title'])

@app.route('/recommendation', methods = ['GET', 'POST'])
def recommendation():
    construct_ratings()
    global user_ratings
    global chosen_films
    result = recommend(user_ratings)
    for i in range(len(result)):
        
        response = requests.request(
            "GET", BASE_URL+str(result[i][1])+API_KEY)
        if(response.status_code == 200):
            result[i][1] = BASE_IMG_URL + response.json()["poster_path"]
        else:
            result[i][1] ="https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"
    
    user_ratings = {}
    chosen_films = []
    return render_template("recommendation.html", content=result)

if __name__ == "__main__":
    app.run(debug=True)