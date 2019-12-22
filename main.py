import os
import requests
import json
from flask import Flask, request, render_template, make_response, send_file
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from fastai import *
from fastai.vision import *
from flask_jsonpify import jsonify
import pickle

app = Flask(__name__, static_folder="../templates", static_url_path="")
api = Api(app)
model = pickle.load(open('export.pkl', 'rb'))
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def submit():
      return render_template('index.html')

@app.route('/employees', methods=['GET', 'POST'])
def get():
        return {'employees': 'kd'} 

""" @app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'GET':
       image = request.get('File')
    return model(image) """

@app.route('/image', methods=['GET', 'POST'])
def handler():
    defaults.device = torch.device('cpu')
    path = Path('.')
    learner = load_learner(path, 'export.pkl')
    image = request.files['files']
    img = open_image(image)
    pred_class,pred_idx,outputs = learner.predict(img)
    result = {
        "prediction": sorted(
            zip(learner.data.classes, map(float, outputs)),
            key=lambda p: p[1],
            reverse=True
        )
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))

 
