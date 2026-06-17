import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

path = "C:/Users/jose_/Documents/TODO DOCTORADO/PROYECTOS/outliers_parametros.csv"
outliersParametros = pd.read_csv(path, sep=",", encoding='latin-1')
print(outliersParametros)
print(outliersParametros.isnull().sum())
outliersParametros=outliersParametros.dropna()
sns.boxplot(outliersParametros["ORTO_PO4"])

