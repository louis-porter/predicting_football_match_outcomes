from preprocessing import df
import pandas as pd

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


# TODO: Need to build exponential moving average features for all team metrics (consider league and season in weighting?)

# TODO: Build a number of days rest feature (Make it be 0 when first game in new season)

df = normalise_home_away(df)

