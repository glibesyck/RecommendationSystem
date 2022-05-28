from unittest import result
from flask import Flask, render_template, request, url_for, send_file
from matplotlib.style import use
from scripts.calc_recommendation import recommend
import requests


app = Flask(__name__)

chosen_films=[]
user_ratings={}
BASE_URL = 'https://api.themoviedb.org/3/movie/'
API_KEY ='?api_key=523e924d1ef2f06415de79b602cc54d6'
BASE_IMG_URL = 'https://image.tmdb.org/t/p/w500'

@app.route('/', methods = ['GET', 'POST'])
def index():
    button_available = False
    if request.method == 'POST':
        movie_name = request.form.get("movie")
        grade = float(request.form.get("grade"))
        if(not movie_name in user_ratings):
            chosen_films.append(movie_name+" with grade "+str(grade))
            user_ratings[movie_name] = grade
    if len(chosen_films) >= 10:
        button_available = True
    return render_template("index.html", content=chosen_films, button_av = button_available)

@app.route('/recommendation', methods = ['GET', 'POST'])
def recommendation():

    global user_ratings
    result = recommend(user_ratings)
    for i in range(len(result)):
        
        response = requests.request(
            "GET", BASE_URL+str(result[i][1])+API_KEY)
        if(response.status_code == 200):
            result[i][1] = BASE_IMG_URL + response.json()["poster_path"]
        else:
            result[i][1] ="https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"
    
    user_ratings = {}
    return render_template("recommendation.html", content=result)

if __name__ == "__main__":
    app.run(debug=True)