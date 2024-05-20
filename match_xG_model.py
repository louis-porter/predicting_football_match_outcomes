from feature_engineering import final_df
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso


def test_training_split(df):
    X_full = df[["home?", "rolling_shots", "rolling_goals", "rolling_xG", "rolling_deep",
            "opponent_rolling_shots_conceded", "opponent_rolling_xG_conceded","opponent_rolling_goals_conceded",
            "opponent_rolling_deep_conceded", "avg_market_value", "opponent_avg_market_value"]]
    
    X = df[["rolling_xG", "opponent_rolling_xG_conceded", "home?"]]

    y = df["xG"]
    X_train, X_temp, y_train, y_temp = train_test_split(X_full, y, test_size=0.4, random_state=42)
    X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_test, y_train, y_test, X_cv, y_cv


def predict_xG_for(X_train,y_train, X_test, y_test, X_cv, y_cv):
    X_train, y_train = np.array(X_train), np.array(y_train)

    model = LinearRegression().fit(X_train,y_train)
    model_ridge = Ridge(alpha=0.1).fit(X_train, y_train)
    model_lasso = Lasso(alpha=0.1).fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_cv = model.predict(X_cv)
    y_pred_test = model.predict(X_test)

    #Testing regulariazation
    y_pred_train_ridge = model_ridge.predict(X_train)
    y_pred_cv_ridge = model_ridge.predict(X_cv)

    y_pred_train_lasso = model_lasso.predict(X_train)
    y_pred_cv_lasso = model_lasso.predict(X_cv)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_cv = mean_squared_error(y_cv, y_pred_cv)

    mse_train_ridge = mean_squared_error(y_train, y_pred_train_ridge)
    mse_cv_ridge = mean_squared_error(y_cv, y_pred_cv_ridge)

    mse_train_lasso = mean_squared_error(y_train, y_pred_train_lasso)
    mse_cv_lasso = mean_squared_error(y_cv, y_pred_cv_lasso)


    print("Normal Model:")
    print(f"Training error: {mse_train}")
    print(f"CV error: {mse_cv}")

    print("\nRidge Model:")
    print(f"Training error: {mse_train_ridge}")
    print(f"CV error: {mse_cv_ridge}")

    print("\nLasso Model:")
    print(f"Training error: {mse_train_lasso}")
    print(f"CV error: {mse_cv_lasso}")

    r_sq = model.score(X_test,y_test)
    # mse = mean_squared_error(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)
    # rmse = np.sqrt(mse)

    print(f"Coefficient of determination: {r_sq}")
    # print(f"MSE: {mse}")
    # print(f"MAE: {mae}")
    # print(f"RMSE: {rmse}")
    # print(f"intercept: {model.intercept_}")
    # print(f"coefficients: {model.coef_}")

    return X_train, y_train, model


    
#TEST APPLYING LOG ONTO OUTCOME VARIABLES

X_train, X_test, y_train, y_test, X_cv, y_cv = test_training_split(final_df)
predict_xG_for(X_train, y_train, X_test, y_test, X_cv, y_cv)




