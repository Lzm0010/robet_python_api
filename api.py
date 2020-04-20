from flask import Flask, request
import json
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/sports/<sport>/events/<event_id>', methods=['GET'])
def predict_game(sport, event_id):
    if sport and event_id:
        sport = sport
        event = int(event_id)
    else:
        return "Error: No sport or event field provided. Please specify both." 

    return {"sport": sport, "event": event}

app.run()