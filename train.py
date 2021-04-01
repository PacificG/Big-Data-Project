import pandas as pd
import datetime
import requests
import json
from fbprophet import Prophet
import model as fbmodel
from sklearn.metrics import mean_squared_error

api_key = ""

def dataset(df, date_col, pred):
    df_new = df.rename(columns={datecol:'ds', pred:'y'})[['ds', 'y']]
    return df_new

def train(df, model):
    model.fit(df)
    return model

def save_dataframe(df, path):
    df.to_csv(path+ "/df.csv")

def load_dataframe(path):
    df = pd.read_csv(path+ "/df.csv")
    return df

def train_online(path,api_key):
    model = fbmodel.load_model(path)
    df = load_dataframe(path)
    lasttime = list(df.iloc[df.shape[0]-1:, 0])[0]
    tp = int(datetime.datetime.timestamp(pd.Timestamp(lasttime))) + 3600

    now = datetime.datetime.now()
    e = datetime.datetime.timestamp(now)
    e = int(e - (e%3600))

    dfnew = get_data_from_api(api_key, 1277333, tp, e)
    
    future = dfnew[['ds']]
    forecast = model.predict(future)[['ds', 'yhat']]
    forecast['yactual'] = dfnew['y']
    print(forecast)
    
    print(mean_squared_error(forecast['yactual'], forecast['yhat']))
    params = stan_init(model)
    newmodel = Prophet(yearly_seasonality=True, weekly_seasonality=True).fit(dfnew, init=params)
    #fbmodel.save_model(path)
    dfsave = df.append(dfnew)
    #save_dataframe(dfsave, path)
    
def get_data_from_api(api_key, id, start, end):
    data = requests.get(f'http://history.openweathermap.org/data/2.5/history/city?id={id}&type=hour&start={start}&end={end}&appid={api_key}')
    dic = data.json()
    temp = {}
    for da in dic['list']:
        temp[datetime.datetime.fromtimestamp(da['dt']).__str__()] = da['main']['temp'] - 273.15
    dfnew = pd.DataFrame(list(temp.items()),columns = ['ds','y'])
    return dfnew
        
def stan_init(m):
    """Retrieve parameters from a trained model.
    
    Retrieve parameters from a trained model in the format
    used to initialize a new Stan model.
    
    Parameters
    ----------
    m: A trained model of the Prophet class.
    
    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res




    
