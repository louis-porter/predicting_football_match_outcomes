import _sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data():
    # Loads in the data from the SQLite3 database
    conn = _sqlite3.connect("data/historic_matches.db")
    query = "SELECT DISTINCT * FROM match_data"
    df = pd.read_sql_query(query, conn)

    return df

def transform_columns(df):
    # Transforms the datatypes of the columns in the dataframe. Also scales the numeric columns
    df["match_date"] = pd.to_datetime(df["match_date"])

    numeric_cols = ["home_num_players", "home_market_value", "home_avg_market_value", "away_num_players", "away_market_value", "away_avg_market_value",
                "home_goals", "away_goals", "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target", "home_corners", "away_corners",
                "home_red", "away_red", "home_yellow", "away_yellow", "home_xgoals", "away_xgoals", "home_deep", "away_deep", "home_ppda", "away_ppda",
                "bet365_home_odds", "bet365_away_odds", "bet365_draw_odds", "bet365_u25_odds", "bet365_o25_odds"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols]) 

    return df

def remove_nulls(df):
    # Removing rows where there are null values, i.e. where data is not tracked (ppda for some leagues etc.)
    df = df.dropna(subset=["home_ppda"])
    df = df.dropna(subset=["home_shots"])
    df = df.dropna(subset=["home_yellow"])
    
    return df




df = load_data()
df = transform_columns(df)
df = remove_nulls(df)