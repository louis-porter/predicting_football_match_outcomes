from data_loader import normalized_stats
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def predict_xG_for():

    x = normalized_stats[["Home?", "Days Rest", "Average xG For", "Average xG/D", "Rolling xG For", "Rolling xG/D", "Average G/D", "Salary", 
                          "Opponent Average xG Against", "Opponent Average xG/D", "Opponent Rolling xG Against", "Opponent Rolling xG/D", 
                          "Opponent Average G/D", "Opponent Salary"]]
    y = normalized_stats["xG For"]
    x, y = np.array(x), np.array(y)

    model = LinearRegression().fit(x,y)

    r_sq = model.score(x,y)

    print(f"Coefficient of determination: {r_sq}")

    print(f"intercept: {model.intercept_}")

    print(f"coefficients: {model.coef_}")

    return x, y, model



predict_xG_for()


