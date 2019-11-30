import pandas as pd
import numpy as np
import pickle

def load(path):
    return pickle.load(open(path, "rb"))

def data_preprocessing(df):
    df.drop(['Unnamed: 0', 'Name of show', 'Temperature in Montreal during episode'], axis = 1, inplace = True) 
    binary= load('categories/binary.pkl') 
    season = load('categories/season.pkl') 
    channel_type = load('categories/channel_type.pkl')
    day_of_week = load('categories/day_of_week.pkl')
    genre = load('categories/genre.pkl')
    station = load('categories/station.pkl')
    episode = load('categories/episode.pkl')
    name_of_episode = load('categories/name_of_episode.pkl')
    end_time = load('categories/end_time.pkl')
    start_time = load('categories/start_time.pkl')
    date = load('categories/date.pkl')
    year = load('categories/year.pkl')
    df['Movie?'] = df['Movie?'].map(binary)
    df['Movie?'] = df['Movie?'].fillna(value=2)
    df['Channel Type'] = df['Channel Type'].map(channel_type)
    df['Channel Type'] = df['Channel Type'].fillna(value=2)
    df['Season'] = df['Season'].map(season)
    df['Day of week'] = df['Day of week'].map(day_of_week)
    df['First time or rerun'] = df['First time or rerun'].map(binary)
    df['First time or rerun'] = df['First time or rerun'].fillna(value=2)
    df['# of episode in the season'] = df['# of episode in the season'].map(binary)
    df['# of episode in the season'] = df['# of episode in the season'].fillna(value=2)
    df['Game of the Canadiens during episode?'] = df['Game of the Canadiens during episode?'].map(binary)
    df['Game of the Canadiens during episode?'] = df['Game of the Canadiens during episode?'].fillna(value=2)
    df['Station'] = df['Station'].map(station)
    df['Station'] = df['Station'].fillna(station['No value'])
    df['Genre'] = df['Genre'].map(genre)
    df['Genre'] = df['Genre'].fillna(genre['No value'])
    df['Year'] = df['Year'].map(year)
    df['Year'] = df['Year'].fillna(year['No value'])
    df['Episode'] = df['Episode'].map(episode)
    df['Episode'] = df['Episode'].fillna(episode['No value'])
    df['Name of episode'] = df['Name of episode'].map(name_of_episode)
    df['Name of episode'] = df['Name of episode'].fillna(name_of_episode['No value'])
    df['End_time'] = df['End_time'].map(end_time)
    df['End_time'] = df['End_time'].fillna(end_time['No value'])
    df['Start_time'] = df['Start_time'].map(start_time)
    df['Start_time'] = df['Start_time'].fillna(start_time['No value'])
    df['Date'] = df['Date'].map(date)
    df['Date'] = df['Date'].fillna(date['No value'])
    return df

def categorical_encoding(path):
    df = pd.read_csv(path)
    save( {"Yes":1, "No":0}, "binary.pkl")
    save(build_encoder(df['Season'].unique()) ,"season.pkl")
    save({ 'General Channel': 1,'Specialty Channel':0},"channel_type.pkl")
    save(build_encoder(df['Day of week'].unique()), "day_of_week.pkl")
    save(build_encoder(df['Genre'].unique()), "genre.pkl")
    save(build_encoder(df['Station'].unique()) ,"station.pkl")
    save(build_encoder(df['Year'].unique()), "year.pkl")
    save(build_encoder(df['Episode'].unique()),"episode.pkl")
    save(build_encoder(df['Name of episode'].unique()), "name_of_episode.pkl")
    save(build_encoder(df['End_time'].unique()), "end_time.pkl")
    save(build_encoder(df['Start_time'].unique()), "start_time.pkl")
    save(build_encoder(df['Date'].unique()), "date.pkl")

def build_encoder(lst):   
    labels = {value:key for key, value in enumerate(lst,1)}   
    labels['No value'] = len(lst) + 1
    return labels

def save(keys , path):
    pickle.dump( keys ,open(path,"wb"))
