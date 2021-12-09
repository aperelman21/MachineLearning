# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:47:37 2021

@author: apere
"""


import matplotlib.pyplot as plt
import numpy as np
import math as math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd

#Importamos la base de datos que generamos previamente
#data = pd.read_csv('suiciderates.csv')
data = pd.read_csv('data.csv',usecols=["suicide_rates","alcohol_consumption","gov_expenditure","homicide_rates","mortality_rates","unemployment"],nrows=182)
#print(data)

data = (data-data.min())/(data.max()-data.min())

# plt.plot(data['suicide_rates'],data['alcohol_consumption'],'o')
# plt.xlabel("suicide rates")
# plt.ylabel("alcohol consumption")
# plt.show()
# plt.plot(data['suicide_rates'],data['gov_expenditure'],'o')
# plt.xlabel("suicide rates")
# plt.ylabel("gov_expenditure")
# plt.show()
# plt.plot(data['suicide_rates'],data['homicide_rates'],'o')
# plt.xlabel("suicide rates")
# plt.ylabel("homicide rates")
# plt.show()
# plt.plot(data['suicide_rates'],data['mortality_rates'],'o')
# plt.xlabel("suicide rates")
# plt.ylabel("infant mortality rates")
# plt.show()

X = data[['alcohol_consumption','gov_expenditure','homicide_rates','mortality_rates','unemployment']]
Y = data['suicide_rates']
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

def regresion_lineal(X,Y,alpha,maxIter,gamma):
    #Inicilaizar los parametros del modelo
    np.random.seed(1)
    theta0 = np.random.rand()
    theta1 = np.random.rand()
    theta2 = np.random.rand()
    theta3 = np.random.rand()
    theta4 = np.random.rand()
    theta5 = np.random.rand()
    
    #variables auxiliares
    i = 0
    converge = False
    m = int(X.shape[0])
    logErr = [0.1]
    #ciclo de aprendizaje
    while not converge and i <= maxIter:
        i += 1
        #Observar
        yH = theta0 + theta1*X.iloc[:,0]+ theta2*X.iloc[:,1]+ theta3*X.iloc[:,2]+ theta4*X.iloc[:,3]+ theta5*X.iloc[:,4]
        E = yH-Y
        
        #actualizar
        newt0 = theta0*(1-(alpha*gamma/m)) - alpha * (1/m) * sum(E)
        newt1 = theta1*(1-(alpha*gamma/m)) - alpha * (1/m) * sum(E*X.iloc[:,0])
        newt2 = theta2*(1-(alpha*gamma/m)) - alpha * (1/m) * sum(E*X.iloc[:,1])
        newt3 = theta3*(1-(alpha*gamma/m)) - alpha * (1/m) * sum(E*X.iloc[:,2])
        newt4 = theta4*(1-(alpha*gamma/m)) - alpha * (1/m) * sum(E*X.iloc[:,3])
        newt5 = theta5*(1-(alpha*gamma/m)) - alpha * (1/m) * sum(E*X.iloc[:,4])
    
        #Condicion de convergencia
        Error = 1/(2*m)*(sum((E)**2) + gamma*(theta1**2+theta2**2+theta3**2+theta4**2+theta5**2))
        logErr.append(Error)
        converge = (math.isclose(logErr[-1],logErr[-2],rel_tol=1e-9))
        
        theta0 = newt0
        theta1 = newt1
        theta2 = newt2
        theta3 = newt3
        theta4 = newt4
        theta5 = newt5
        resultados = [Error,theta0,theta1,theta2,theta3,theta4,theta5]
    return resultados

alpha = 0.1
gamma = 2
resp = regresion_lineal(x_train,y_train,alpha,10000,gamma)
print(resp)

yH_test = resp[1] + resp[2]*x_test.iloc[:,0] + resp[2]*x_test.iloc[:,0] + resp[3]*x_test.iloc[:,1] + resp[4]*x_test.iloc[:,2] + resp[5]*x_test.iloc[:,3]+ resp[6]*x_test.iloc[:,4]
E_prueba = yH_test - y_test
E_prueba = yH_test - y_test
Error_prueba = 1/(2*int(x_test.shape[0]))*sum((E_prueba)**2)
print(Error_prueba)

valoresx = np.linspace(1,37,37)
plt.plot(valoresx,yH_test,label="Tasa predecida")
plt.plot(valoresx,y_test,label="Tasa Real")
plt.title("Prediccion de tasas de suicidio")
plt.ylabel("tasa de suicidio")
plt.legend()
plt.show()







