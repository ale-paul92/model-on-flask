#importing libraries
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#loaded_model = pickle.load(open("C:/Alessandro/Documenti/DATA SCIENCE/progetti flask/model on flask/model.pkl","rb"))
loaded_model = pickle.load(open("model.pkl","rb"))

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
    #return "Hello World"


#prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    #loaded_model = pickle.load(open("C:/Alessandro/Documenti/DATA SCIENCE/progetti flask/model on flask/model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':

        feature_order = [
            "age",
            "w_class",
            "edu",
            "martial_stat",
            "occup",
            "relation",
            "race",
            "gender",
            "c_gain",
            "c_loss",
            "hours_per_week",
            "native-country"
        ]

        to_predict_list = [request.form[x] for x in feature_order]
        to_predict_list = list(map(int, to_predict_list))

        result = ValuePredictor(to_predict_list)

        if int(result) == 1:
            prediction = 'Income more than 50K'
        else:
            prediction = 'Income less than 50K'

        return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000)
     
# if __name__ == "__main__":
# 	app.run(host='0.0.0.0', port=5000)