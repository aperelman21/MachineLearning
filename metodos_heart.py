"ARBOLES DE DECISIÓN"

"lIBRERÍAS"
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


"__________________Cart__________________"

df = pd.read_csv("HeartDisease.csv")
Y = np.array(df["Enfermedad"])
X = df[["ChestPainClass","SexClass","MaxHeartRateClass","CholesterolClass","ClaseRestingBlood","ClaseEdad"]]
one_hot = pd.get_dummies(X)
X = np.array(one_hot)

modelo = tree.DecisionTreeClassifier() 
modelo.fit(X,Y)

tree.plot_tree(modelo)


"____________________Bagging_______________________"
#Note que no se pudo considerar todo el dataset pues algunas de sus 
#variables son categoricas.
 
#Importacion del DataSet
np.random.seed(2)
df=pd.read_csv("HeartNum.csv")

#Generación y Normalización de variables
Y = np.array(df["target"])
X = df[["age","trestbps","chol","thalach","oldpeak"]]
X = X.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Propuesta de vecinos más cercanos para weak classifier 
n_neighbors=5
modeloK=KNeighborsClassifier(n_neighbors)
modeloK.fit(X_train,y_train)

modeloB=BaggingClassifier(KNeighborsClassifier(n_neighbors),n_estimators=50)
modeloB.fit(X_train,y_train)

print(modeloK.score(X_test,y_test))
print(modeloB.score(X_test,y_test))

"____________________Random Forest_______________________"
#Note que no se pudo considerar todo el dataset pues algunas de sus 
#variables son categoricas.
 
#Importacion del DataSet
np.random.seed(2)
df=pd.read_csv("HeartNum.csv")

#Generación y Normalización de variables
Y = np.array(df["target"])
X = df[["age","trestbps","chol","thalach","oldpeak"]]
X = X.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Propuesta de vecinos más cercanos para weak classifier 
n_neighbors=5
modeloK=KNeighborsClassifier(n_neighbors)
modeloK.fit(X_train,y_train)

modeloB=BaggingClassifier(KNeighborsClassifier(n_neighbors),n_estimators=50)
modeloB.fit(X_train,y_train)

ModeloDT = tree.DecisionTreeClassifier(criterion = "gini")
ModeloDT.fit(X_train,y_train)

ModeloRF = RandomForestClassifier(n_estimators=100,max_depth=5)
ModeloRF.fit(X_train,y_train)

print("K-Neighbors: ",modeloK.score(X_test,y_test))
print("Decision tree: ",ModeloDT.score(X_test,y_test))
print("Random Forest: ",ModeloRF.score(X_test,y_test))

plt.plot(ModeloRF.feature_importances_)







