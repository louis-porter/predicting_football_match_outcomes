from preprocessing import df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalise_home_away(df):
    # Separates out the home vs away and allows every team to have an individual row of data. Allows us to feature engineer home effect easier.
    home_df = df[["division", "season", "match_date",
             "home_team", "home_avg_market_value", "home_goals", "home_shots", "home_xgoals", "home_deep", "home_ppda",
             "away_team", "away_avg_market_value", "away_goals", "away_shots", "away_xgoals", "away_deep", "away_ppda"]].copy()
    home_df["home?"] = 1
    home_df = home_df.rename(columns={"home_team":"team", "home_avg_market_value":"avg_market_value", "home_goals":"goals",
             "home_shots":"shots", "home_xgoals":"xG", "home_deep":"deep", "home_ppda":"ppda",
             "away_team":"opponent_team", "away_avg_market_value":"opponent_avg_market_value", "away_goals":"opponent_goals", 
             "away_shots": "opponent_shots", "away_xgoals": "opponent_xG", "away_deep":"opponent_deep", "away_ppda":"opponent_ppda"})

    away_df = df[["division", "season", "match_date",
                "away_team", "away_avg_market_value", "away_goals", "away_shots", "away_xgoals", "away_deep", "away_ppda",
                "home_team", "home_avg_market_value", "home_goals", "home_shots", "home_xgoals", "home_deep", "home_ppda"]].copy()
    away_df["home?"] = 0
    away_df = away_df.rename(columns={"away_team":"team", "away_avg_market_value":"avg_market_value", "away_goals":"goals",
             "away_shots":"shots", "away_xgoals":"xG", "away_deep":"deep", "away_ppda":"ppda",
             "home_team":"opponent_team", "home_avg_market_value":"opponent_avg_market_value", "home_goals":"opponent_goals", 
             "home_shots": "opponent_shots", "home_xgoals": "opponent_xG", "home_deep":"opponent_deep", "home_ppda":"opponent_ppda"})

    df = pd.concat([away_df, home_df])
    df.sort_values(["match_date", "division"], inplace=True)

    return df

def penalized_ema(group_df, column_name, span=35):
    # Creates an exponential moving average that applies a penalty for games within span that are in previous seaoson
    ema_values = pd.Series(index=group_df.index, dtype=float)
    for season in group_df['season'].unique():
        season_data = group_df[group_df['season'] == season]
        season_ema = season_data[column_name].astype(float).ewm(span=span, adjust=False, min_periods=1).mean()
        if not ema_values.dropna().empty: # if ema_values series already has data then use last value from prev season
            initial_value = ema_values.ffill().iloc[-1] * 0.75
            season_ema.iloc[0] = (season_ema.iloc[0] + initial_value) / 2
        ema_values.update(season_ema)
    return ema_values

def calculate_team_rolling_means(df):
    df.sort_values(by=["team", "match_date"], inplace=True)
    stat_columns = ['goals', 'xG', 'shots', 'deep', 'ppda']

    # Compute EMAs for team stats
    for col in stat_columns:
        df[f"rolling_{col}"] = df.groupby("team").apply(lambda x: penalized_ema(x, col)).droplevel(0)
        df[f"rolling_{col}_conceded"] = df.groupby("team").apply(lambda x: penalized_ema(x, f'opponent_{col}')).droplevel(0)

    df.reset_index(drop=True, inplace=True)

    return df

def calculate_opponent_rolling_means(df):
    df.sort_values(by=["opponent_team", "match_date"], inplace=True)
    stat_columns = ['goals', 'xG', 'shots', 'deep', 'ppda']

    # Compute EMAs for opponent stats
    for col in stat_columns:
        df[f"opponent_rolling_{col}"] = df.groupby("opponent_team").apply(lambda x: penalized_ema(x, f'opponent_{col}')).droplevel(0)
        df[f"opponent_rolling_{col}_conceded"] = df.groupby("opponent_team").apply(lambda x: penalized_ema(x, col)).droplevel(0)

    df.reset_index(drop=True, inplace=True)

    return df

def calculate_days_rest(df):
    df.sort_values(by=["match_date", "team"], inplace=True)
    df["days_rest"] = df.groupby(["season", "team"], observed=True)["match_date"].diff().dt.days
    df["days_rest"].fillna(0, inplace=True)

    return df


def create_model_df(df):
    df = normalise_home_away(df)
    df = calculate_team_rolling_means(df)
    df = calculate_opponent_rolling_means(df)
    df = calculate_days_rest(df)

    df = df[["division", "season", "team", "days_rest", "home?", "avg_market_value", "rolling_goals",
             "rolling_goals_conceded", "rolling_xG", "rolling_xG_conceded", "rolling_shots", "rolling_shots_conceded",
             "rolling_deep", "rolling_deep_conceded", "rolling_ppda", "rolling_ppda_conceded", 
             "opponent_team", "opponent_avg_market_value", "opponent_rolling_goals", "opponent_rolling_goals_conceded", "opponent_rolling_xG",
             "opponent_rolling_xG_conceded", "opponent_rolling_shots", "opponent_rolling_shots_conceded", "opponent_rolling_deep", "opponent_rolling_deep_conceded", 
             "opponent_rolling_ppda", "opponent_rolling_ppda_conceded", "xG"]]
    
    df = df[df["division"] == "Ligue 1"]
    
    return df

    

final_df = create_model_df(df)

