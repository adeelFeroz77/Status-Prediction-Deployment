from imp import load_module
from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)


model = pickle.load(open('ensemble.pkl','rb'))
cols = ['founded_at', 'funding_rounds', 'funding_total_usd', 'milestones', 'relationships']
status = {2:'ipo',1:'closed',0:'acquired',3:'operating'}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    # print(int_features)
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_unseen)
    print(prediction)
    return render_template('home.html',pred='Expected Status will be {}'.format(status[prediction[0]]))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.get_json(force=True)
#     data_unseen = pd.DataFrame([data])
#     prediction = model.predict(data_unseen)
#     output = prediction[0]
#     return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)