# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:58:41 2021

@author: apere
"""
import matplotlib.pyplot as plt
import numpy as np
import math as math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('data.csv',usecols=["suicide_rates","alcohol_consumption","gov_expenditure","homicide_rates","mortality_rates","unemployment"],nrows=182)
#print(data)

data = (data-data.min())/(data.max()-data.min())
X = data[['alcohol_consumption','gov_expenditure','homicide_rates','mortality_rates','unemployment']]
Y = data['suicide_rates']
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


mejori = 0
mejorj = 0
minErr = 1
for i in range(10):
    for j in range(10):
        modelo = MLPRegressor(random_state=1,hidden_layer_sizes=(i+1,j+1),
                      activation='tanh',
                      learning_rate_init=0.1,
                      max_iter=1000,
                      solver='sgd')
        modelo.fit(x_train,y_train)
        yH_test = modelo.predict(x_test)
        E_prueba = yH_test - y_test
        Error_prueba = 1/(2*int(x_test.shape[0]))*sum((E_prueba)**2)
        
        print("i="+str(i+1)+ " j= "+str(j+1)+" error de prueba = "+str(Error_prueba))
        if Error_prueba < minErr:
            minErr = Error_prueba
            mejori = i+1
            mejorj = j+1
            

            
modelo = MLPRegressor(random_state=1,hidden_layer_sizes=(mejori,mejorj),
                      activation='tanh',
                      learning_rate_init=0.1,
                      max_iter=1000,
                      solver='sgd')
modelo.fit(x_train,y_train)
yH_test = modelo.predict(x_test)
valoresx = np.linspace(1,37,37)
plt.plot(valoresx,yH_test,label="Tasa predecida")
plt.plot(valoresx,y_test,label="Tasa Real")
plt.title("Prediccion de tasas de suicidio")
plt.ylabel("tasa de suicidio")
plt.legend()
plt.show()
print(modelo.accuracy_score(x_train,y_train))
            
        

                      
