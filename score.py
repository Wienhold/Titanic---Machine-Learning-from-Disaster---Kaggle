import json
import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)
    print("Model loaded:\n", model)

def run(data):
    try:
        data = json.loads(data)['data']
        print("Data received of type", type(data), "\nContents:\n", data)
        data = pd.DataFrame.from_dict(data)
        print("Data was converted to type", type(data), "\nContents:\n", data)

        result = model.predict(data)
        print("Model produced result of type", type(result), "\nContents:\n", result)
        
        result_list = result.tolist()
        print("Result was converted to type", type(result_list), "\nContents:\n", result_list)

        # You can return any data type, as long as it is JSON serializable.
        return result_list
    except Exception as e:
        error = str(e)
        return error
    