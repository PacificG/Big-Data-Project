import pandas as pd
from fbprophet import Prophet
import json
from fbprophet.serialize import model_to_json, model_from_json

def pModel():
    m = Prophet()
    return m

def save_model(model, path):
    with open(f"{path}/model.json", 'w') as fout:
        json.dump(model_to_json(model), fout)
    return f'saved model at {path}/model.json'

def load_model(path):
    with open(f"{path}/model.json", 'r') as fin:
        m = model_from_json(json.load(fin))
    return m

def trainModel(model, df):
    model.fit(df)
    return model


    
