from flask import Flask, request
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sportsreference.mlb.schedule import Schedule as MLB_Schedule
from sportsreference.nba.schedule import Schedule as NBA_Schedule 
from sportsreference.nfl.schedule import Schedule as NFL_Schedule
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

@app.route("/", methods=['GET'])
def home_view():
    return "<h1>Hi welcome to ML</h1>"

@app.route('/MLB/<team1>/<team2>', methods=['GET'])
def predict_mlb_game(team1, team2):

    # FIELDS_TO_DROP = ['away_points', 'home_points', 'date', 'location',
    #               'losing_abbr', 'losing_name', 'winner', 'winning_abbr',
    #               'winning_name', 'home_ranking', 'away_ranking']

    # dataset = pd.DataFrame()
    # auburn_schedule = Schedule('AUBURN')
    # dataset = pd.concat([dataset, auburn_schedule.dataframe_extended])
    # # for team in teams:
    # #     dataset = pd.concat([dataset, team.schedule.dataframe_extended])
    # X = dataset.drop(FIELDS_TO_DROP, 1).dropna().drop_duplicates()
    # y = dataset[['home_points', 'away_points']].values
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    # parameters = {'bootstrap': False,
    #             'min_samples_leaf': 3,
    #             'n_estimators': 50,
    #             'min_samples_split': 10,
    #             'max_features': 'sqrt',
    #             'max_depth': 6}
    # model = RandomForestRegressor(**parameters)
    # model.fit(X_train, y_train)
    
    # predicted_scores = model.predict(X_test).astype(int).tolist()
    # results = {"predicted": predicted_scores, "actual": y_test.tolist()}
    # json_results = json.dumps(results)
    # return json_results


    dataset = {}
    teams = [team1, team2]
    for num, team in enumerate(teams):
        df = MLB_Schedule(team, year=2019).dataframe
        df = df[['runs_scored']].head(147) #started at 130 on August 24th 2019 - now on 147 Sep 7 hasnt run yet

        forecast_out=int(1)
        print(df.shape)
        df['Prediction'] = df[['runs_scored']].shift(-forecast_out)

        X = np.array(df.drop(['Prediction'], 1))
        X = preprocessing.scale(X)

        X_forecast = X[-forecast_out:]
        X = X[:-forecast_out]

        y = np.array(df['Prediction'])
        y = y[:-forecast_out]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        clf = LinearRegression()
        clf.fit(X_train, y_train)

        confidence = clf.score(X_test, y_test)

        forecast_prediction = clf.predict(X_forecast)
        lists_of_forecast = forecast_prediction.tolist()
        if num == 0:
            dataset[team1] = {"confidence": confidence, "predicted_score": lists_of_forecast}
        else:
            dataset[team2] = {"confidence": confidence, "predicted_score": lists_of_forecast}

    json_forecast = json.dumps(dataset, default=str)

    return json_forecast


@app.route('/NBA/<team1>/<team2>', methods=['GET'])
def predict_nba_game(team1, team2):
    dataset = {}
    teams = [team1, team2]

    for num, team in enumerate(teams):
        df = NBA_Schedule(team, year=2019).dataframe
        df = df[['points_scored']]

        forecast_out=int(1)
        print(df.shape)
        df['Prediction'] = df[['points_scored']].shift(-forecast_out)

        X = np.array(df.drop(['Prediction'], 1))
        X = preprocessing.scale(X)

        X_forecast = X[-forecast_out:]
        X = X[:-forecast_out]

        y = np.array(df['Prediction'])
        y = y[:-forecast_out]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        clf = LinearRegression()
        clf.fit(X_train, y_train)

        confidence = clf.score(X_test, y_test)

        forecast_prediction = clf.predict(X_forecast)
        lists_of_forecast = forecast_prediction.tolist()
        if num == 0:
            dataset[team1] = {"confidence": confidence, "predicted_score": lists_of_forecast}
        else:
            dataset[team2] = {"confidence": confidence, "predicted_score": lists_of_forecast}

    json_forecast = json.dumps(dataset, default=str)

    return json_forecast


@app.route('/NFL/<team1>/<team2>', methods=['GET'])
def predict_nfl_game(team1, team2):
    dataset = {}
    teams = [team1, team2]
    for num, team in enumerate(teams):
        df = NFL_Schedule(team, year=2018).dataframe
        df = df[['points_scored']]

        forecast_out=int(1)
        print(df.shape)
        df['Prediction'] = df[['points_scored']].shift(-forecast_out)

        X = np.array(df.drop(['Prediction'], 1))
        X = preprocessing.scale(X)

        X_forecast = X[-forecast_out:]
        X = X[:-forecast_out]

        y = np.array(df['Prediction'])
        y = y[:-forecast_out]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        clf = LinearRegression()
        clf.fit(X_train, y_train)

        confidence = clf.score(X_test, y_test)

        forecast_prediction = clf.predict(X_forecast)
        lists_of_forecast = forecast_prediction.tolist()
        if num == 0:
            dataset[team1] = {"confidence": confidence, "predicted_score": lists_of_forecast}
        else:
            dataset[team2] = {"confidence": confidence, "predicted_score": lists_of_forecast}

    json_forecast = json.dumps(dataset, default=str)

    return json_forecast

app.run()