
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


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





