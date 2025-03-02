import pandas as pd
from autogluon.tabular import TabularPredictor

from src.data_loader import DataLoader
from src.prediction_utils import *

#file_path = ["playground-series-s4e11/"]

data_path = "./data/"
file_path = ["exploring-mental-health-data-clone/","dir2/"]

print('Loading Files')
data_loader = DataLoader(data_path, file_path)
files = data_loader.load_csvs()

#print(files)

#breakpoint()

train = files['train.csv']
test = files['test.csv']

print('Making predictions...')
#predictor = TabularPredictor(label="Depression", eval_metric="log_loss").fit(train,presets='medium')
predictor = TabularPredictor.load("/Users/chiu/Documents/AIML/kaggle_depression/AutogluonModels/ag-20250302_180125")

y_pred = predictor.predict(test)
probs = predictor.predict_proba(test)

y_pred.index = y_pred.index + 140700
probs.index = probs.index + 140700

y_pred.to_csv("y_pred.csv",index=True,index_label='id')
probs.to_csv("probs.csv",index=True,index_label='id')



