import pandas as pd
import helper as hp

def read(path):
    data = pd.read_csv(path)
    data = hp.data_preprocessing(data)
    return data

def write(data, path):
    df = pd.DataFrame( {'Market Share_total':data} , columns = ['Market Share_total']) 
    df.to_csv(path)
