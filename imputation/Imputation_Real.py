import pandas as pd 
import missingno as msno
from sklearn.experimental import enable_iterative_imputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
import numpy as np
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, SimpleFill


df_completo = pd.read_csv("C:/Users/jose_/Documents/TODO DOCTORADO/PROYECTOS/DS_SINALOA_FILTRO_IMPUTADO.csv", sep=',', encoding='latin1')
#df_completo = df_completo.dropna()

#scaler = MinMaxScaler()
#df_completo = pd.DataFrame(scaler.fit_transform(df_completo), columns=df_completo.columns)
parametro = "PROFUNDIDAD"
print(df_completo)
#df_incompleto = df_completo.copy(deep=True)
estimator = linear_model.LinearRegression()
df_filled_knn =IterativeImputer(estimator=estimator).fit_transform(X=df_completo) 
#df_filled_knn =KNNImputer(n_neighbors=6).fit_transform(X=df_completo)

pd.DataFrame(df_filled_knn).to_csv("C:/Users/jose_/Documents/TODO DOCTORADO/PROYECTOS/DS_SINALOA_FILTRO_IMPUTADO.csv")
