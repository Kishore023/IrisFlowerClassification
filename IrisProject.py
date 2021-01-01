from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

model = joblib.load('model.pkl')
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


@app.route('/')
def inputlabels():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        inputvariables = request.form
        inputs = [x for x in inputvariables.values()]
        final = [np.array(inputs)]
        prediction = model.predict(final)

        return render_template('predict.html', prediction=classes[prediction[0]])


if __name__ == '__main__':
    app.run(debug=True)
