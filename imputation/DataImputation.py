import pandas as pd 
import missingno as msno
from sklearn.experimental import enable_iterative_imputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model

metals = pd.read_excel("/Users/posgrado02/Documents/PROYECTOS/IMPUTATION/Rio_fuerte_ITERATIVE_imputed1.xlsx", sheet_name="Sheet1")
print(metals.isnull().mean().sort_values(ascending=False))

#metals_knn_imputed = metals.copy(deep=True)
#mice_imputed = metals.filter(["COLI_FEC", "N_TOTK","TURBIEDAD", "ORTO_PO4", "COLOR_VER", "N_NH3", "N_NO2", "N_NO3","N_ORG", "N_TOT"], axis=1).copy()
metals_knn_imputed = metals.filter(["COLI_FEC", "N_TOTK","TURBIEDAD", "ORTO_PO4", "COLOR_VER", "N_NH3", "N_NO2", "N_NO3","N_ORG", "N_TOT"], axis=1).copy()

scaler = MinMaxScaler(feature_range=(0,1))
metals_knn_imputed = pd.DataFrame(scaler.fit_transform(metals_knn_imputed), columns = metals_knn_imputed.columns)
knn_imp = KNNImputer(n_neighbors=3, weights= "uniform", metric="nan_euclidean")
metals_knn_imputed = pd.DataFrame(knn_imp.fit_transform(metals_knn_imputed), columns=metals_knn_imputed.columns)
converted = scaler.inverse_transform(metals_knn_imputed)
metals_knn_imputed = pd.DataFrame(converted)
metals_knn_imputed.loc[:, :] = knn_imp.fit_transform(metals_knn_imputed)

metals_knn_imputed.to_excel("/Users/posgrado02/Documents/PROYECTOS/IMPUTATION/Rio_fuerte_KNN_imputed.xlsx")

#ii_imp = IterativeImputer(estimator= linear_model.BayesianRidge(), max_iter=10, n_nearest_features=None, imputation_order="ascending")

#df_mice_imputed = pd.DataFrame(ii_imp.fit_transform(mice_imputed), columns = mice_imputed.columns)
#metals_iterative_imputed.loc[:, :] = ii_imp.fit_transform(metals_iterative_imputed)

#df_mice_imputed.to_excel("/Users/posgrado02/Documents/PROYECTOS/IMPUTATION/Rio_fuerte_ITERATIVE_imputed.xlsx")



