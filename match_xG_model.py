from feature_engineering import final_df
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


def test_training_split(df):
    X = df.drop(["division", "season", "team", "opponent_team", "xG"], axis=1)
    y = df["xG"]
    print(y.isna().sum())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def predict_xG_for(X_train,y_train, X_test, y_test):
    X_train, y_train = np.array(X_train), np.array(y_train)

    model = LinearRegression().fit(X_train,y_train)
    y_pred = model.predict(X_test)

    r_sq = model.score(X_test,y_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Coefficient of determination: {r_sq}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"intercept: {model.intercept_}")
    print(f"coefficients: {model.coef_}")

    return X_train, y_train, model


def corr_matrix(x):
    corr_matrix = x.corr()
    print(corr_matrix)


    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()
    
#TEST APPLYING LOG ONTO OUTCOME VARIABLES

X_train, X_test, y_train, y_test = test_training_split(final_df)
predict_xG_for(X_train, y_train, X_test, y_test)




