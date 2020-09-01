# -*- coding: utf-8 -*-

import pandas as pd
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import json
import pickle

NN_model = load_model('NN_model.h5')
KNN_model = pickle.load(open('KNN_model.pkl','rb'))

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
    result_NN = NN_model.predict(df_scaled)
    result_scaling_factor = KNN_model.predict(data_df)
    
    result = result_NN*result_scaling_factor[0]

    # send back to browser
    result = json.dumps(result.tolist())
    
    # return data
    return result

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
    
