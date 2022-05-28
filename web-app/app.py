from flask import Flask, render_template, request, url_for, send_file

app = Flask(__name__)

chosen_films=[]

@app.route('/', methods = ['GET', 'POST'])
def index():
    button_available = False
    if request.method == 'POST':
        movie_name = request.form.get("movie")
        grade = request.form.get("grade")
        chosen_films.append(movie_name+" with grade "+grade)
    if len(chosen_films) >= 10:
        button_available = True
    return render_template("index.html", content=chosen_films, button_av = button_available)

@app.route('/recommendation', methods = ['GET', 'POST'])
def recommendation():
    pass

if __name__ == "__main__":
    app.run(debug=True)