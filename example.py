import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_log_error
from sklearn.model_selection import train_test_split
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from yellowbrick.regressor import ResidualsPlot
import numpy as np

# cargar el dataset 
car_df = pd.read_csv('/Users/ditsy/PythonProjects/MachineLearning/CarPrice.csv')

car_df.shape

# print(car_df.head())

# print(car_df.info())

car_df_numeric = car_df.select_dtypes(include=['float64', 'int64'])
# print(car_df_numeric.info())

# print(car_df.describe())

null_count = car_df.isnull().sum()
# print(null_count)

def simbol():
    # Analizar estadísticos de las variables symboling, price, curbweight
    var1 = 'symboling'
    print("Estadísticos de "+var1+":")
    print(car_df[var1].describe().round(2))

    # Crear histogramas de las variables symboling, price, curbweight
    plt.hist(car_df[var1], bins=20)
    plt.title("Distribución de "+var1)
    plt.show()


def price():
    # Analizar estadísticos de las variables symboling, price, curbweight

    var1 = 'price'
    print("Estadísticos de "+var1+":")
    print(car_df[var1].describe().round(2))

    # Crear histogramas de las variables symboling, price, curbweight
    plt.hist(car_df[var1], bins=20)
    plt.title("Distribución de "+var1)
    plt.show()

def curbweight():
    # Analizar estadísticos de las variables symboling, price, curbweight
    var1 = 'curbweight'
    print("Estadísticos de "+var1+":")
    print(car_df[var1].describe().round(2))

    # Crear histogramas de las variables symboling, price, curbweight
    plt.hist(car_df[var1], bins=20)
    plt.title("Distribución de "+var1)
    plt.show()

# simbol()
# price()
# curbweight()