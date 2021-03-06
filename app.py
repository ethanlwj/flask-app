# -*- coding: utf-8 -*-

import pandas as pd
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import json
import pickle

# NN_model = load_model('NN_model.h5')
# KNN_model = pickle.load(open('KNN_model.pkl','rb'))
NN_model_v2 = load_model('NN_model_v2.h5',compile=False)
KNN_model_v2 = pickle.load(open('KNN_model_v2.pkl','rb'))
# Light_classification = pickle.load(open('Light_Classification.pkl','rb'))
Light_classification_v2 = pickle.load(open('Light_Classification_v2.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)
    
    # convert data into dataframe
    data_df = pd.DataFrame(data)
    
    max_value = data_df.max(axis=1)
    print(max_value)
    df_scaled = data_df.apply(lambda x: x/max_value)
    
    # predictions
    result_NN = NN_model_v2.predict(df_scaled)
    result_scaling_factor = KNN_model_v2.predict(data_df)
    
    result = result_NN*result_scaling_factor[0]
    
    result_light_classification = Light_classification_v2.predict(df_scaled)[0]

    
    result_dict = {
        'SPM' : result[0].tolist(),
        'Light' : result_light_classification
        }

    result = json.dumps(result_dict)
    
    # return data
    return result

@app.route('/LC', methods=['POST'])
def classify_light():
    data = request.get_json(force=True)
    
    # convert data into dataframe
    data_df = pd.DataFrame(data)
    
    max_value = data_df.max(axis=1)
    df_scaled = data_df.apply(lambda x: x/max_value)
    
    result_light_classification = Light_classification.predict(df_scaled)[0]
    print(result_light_classification)
    
    result = json.dumps(result_light_classification)
    
    return result;


if __name__ == '__main__':
    app.run(port = 5000, debug=True)
    
