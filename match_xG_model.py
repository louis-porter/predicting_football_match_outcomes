from data_loader import x,y
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


def predict_xG_for(x,y):
    x, y = np.array(x), np.array(y)

    model = LinearRegression().fit(x,y)

    r_sq = model.score(x,y)

    print(f"Coefficient of determination: {r_sq}")

    print(f"intercept: {model.intercept_}")

    print(f"coefficients: {model.coef_}")

    return x, y, model


def corr_matrix(x):
    corr_matrix = x.corr()
    print(corr_matrix)


    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()
    

    



