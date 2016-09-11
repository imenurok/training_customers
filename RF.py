#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import data_writer
folder_name="RF"

data = pd.read_csv("train.csv")

training_label = data["y"]
data = data.drop("y",axis=1)

data2 = pd.read_csv("test.csv")
data = pd.concat((data,data2),axis=0)

def dummies(data,name):
    print name
    dummy = pd.get_dummies(data[name])
    data = pd.concat((data,dummy),axis=1)
    data = data.drop(name,axis=1)
    return data
names=["job","marital","education","default","housing","loan","contact","month","poutcome"]
for name in names:
    data=dummies(data,name)

training_data=data[:training_label.shape[0]]
predict_data=data[training_label.shape[0]:]

print training_data.shape
print training_label.shape
print predict_data.shape

names=training_data.columns

np.random.seed(0)

model = RandomForestRegressor(n_estimators=400)
model.fit(training_data,training_label)
training_output,predict_output = model.predict(training_data),model.predict(predict_data)
data_writer.data_write(folder_name,training_output,predict_output,silent=False,start1=True)

score = model.feature_importances_
mapped = {names[i]:score[i] for i in range(len(names))}
fig, ax = plt.subplots(1, 1, figsize=(7, 25))
xgb.plot_importance(mapped,ax=ax)
plt.savefig("graph.png")
