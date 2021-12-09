"ARBOLES DE DECISIÓN"

"lIBRERÍAS"
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("BaseClasificada_SR.csv")
df2 = df[['MortalityClass', 'HomicideClass', 'GobExpenditureClass', 'AlcoholConsumptionClass']]
y = np.array(df['SuicideClass'])
one_hot = pd.get_dummies(df2)
x = np.array(one_hot)


modelo=tree.DecisionTreeClassifier()
modelo.fit(x,y)



np.random.seed(2)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)



scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



n_neighbors = 5 
modelo2 = KNeighborsClassifier(n_neighbors)
modelo2.fit(x_train,y_train)



modelo2b=BaggingClassifier(KNeighborsClassifier(n_neighbors),n_estimators = 50)
modelo2b.fit(x_train,y_train)


print(modelo2.score(x_test, y_test))
print(modelo2b.score(x_test, y_test))


modelorf=RandomForestClassifier(n_estimators=100,max_depth=5)
modelorf.fit(x_train,y_train)

"____________________Random Forest_______________________"

 
#Importacion del DataSet

df=pd.read_csv("data.csv",nrows=182)

#Generación y Normalización de variables
Y = np.array(df["suicide_rates"])
Y = Y.astype(int)
X = df[["alcohol_consumption","alcohol_consumption","gov_expenditure","homicide_rates","mortality_rates","unemployment"]]
X = X.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Propuesta de vecinos más cercanos para weak classifier 
n_neighbors=5
modelo=KNeighborsClassifier(n_neighbors)
modelo.fit(X_train,y_train)

modeloB=BaggingClassifier(KNeighborsClassifier(n_neighbors),n_estimators=50)
modeloB.fit(X_train,y_train)

ModeloDT = tree.DecisionTreeClassifier(criterion = "gini")
ModeloDT.fit(X_train,y_train)

ModeloRF = RandomForestClassifier(n_estimators=100,max_depth=5)
ModeloRF.fit(X_train,y_train)

print("Random Forest: ",ModeloRF.score(X_test,y_test))

plt.plot(ModeloRF.feature_importances_)

yH_test = ModeloRF.predict(X_test)
valoresx = np.linspace(1,37,37)
plt.plot(valoresx,yH_test,label="Tasa predecida")
plt.plot(valoresx,y_test,label="Tasa Real")
plt.title("Prediccion de tasas de suicidio")
plt.ylabel("tasa de suicidio")
plt.legend()
plt.show()

ModeloRF2 = RandomForestRegressor(n_estimators=100,max_depth=5)
ModeloRF2.fit(X_train,y_train)
print("Random Forest: ",ModeloRF2.score(X_test,y_test))

plt.plot(ModeloRF2.feature_importances_)

yH_test = ModeloRF2.predict(X_test)
valoresx = np.linspace(1,37,37)
plt.plot(valoresx,yH_test,label="Tasa predecida")
plt.plot(valoresx,y_test,label="Tasa Real")
plt.title("Prediccion de tasas de suicidio")
plt.ylabel("tasa de suicidio")
plt.legend()
plt.show()


