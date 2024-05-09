import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#===========================LOAD AND PREPROCESSING===========================#
def load_and_prepare_match_data(filepath):
    dtype = {
        "Home Goals": "int32",
        "Away Goals": "int32",
        "Home Team": "category",
        "Away Team": "category",
        "GW": "int16",
    }
    parse_dates = ["Date"]
    df = pd.read_csv(filepath, dtype=dtype, parse_dates=parse_dates)
    df.sort_values("Date", inplace=True)
    df[["Home Goals", "Away Goals"]] = df["Score"].str.split("–", expand=True).astype("int32")
    return df

def load_and_prepare_salary_data(filepath):
    dtype = {
        "Salary P/W (K)" : "int32",
        "Season": "category",
        "Team": "category"
    }
    df2 = pd.read_csv(filepath, dtype=dtype)
    return df2


#===========================NORMALIZED STATISTICAL CALCULATIONS===========================#
def calculate_normalized_stats(df, salary):
    # Merge dataset with salary data
    salary["Salary P/W (K)"] = salary["Salary P/W (K)"]/1000 # Normalise to M units
    df = df.merge(salary, left_on=["Home Team", "Season"], right_on=["Team", "Season"], how="left")
    df.rename(columns={"Salary P/W (K)": "Home Team Salary"}, inplace=True)
    df = df.merge(salary, left_on=["Away Team", "Season"], right_on=["Team", "Season"], how="left")
    df.rename(columns={"Salary P/W (K)": "Away Team Salary"}, inplace=True)

    # Normalise the dataset so that we have Team and Opponent rather than Home and Away
    home_stats = df[["Season", "Date", "GW", "Home Team","Home Team Salary", "Away Team", "Home Goals",
                      "Home xG", "Away Goals", "Away xG", "Away Team Salary"]].rename(
        columns={"Home Team": "Team", "Away Team": "Opponent", "Home Goals": "Goals For", "Away Goals":
                 "Goals Against", "Home xG": "xG For", "Away xG": "xG Against", "Home Team Salary": "Salary",
                 "Away Team Salary": "Opponent Salary"}
    )
    home_stats["Home?"] = 1
    away_stats = df[["Season", "Date", "GW", "Home Team", "Away Team Salary", "Away Team", "Home Goals", 
                     "Home xG", "Away Goals", "Away xG", "Home Team Salary"]].rename(
        columns={"Away Team": "Team", "Home Team": "Opponent", "Away Goals": "Goals For", "Home Goals":
                 "Goals Against", "Away xG": "xG For", "Home xG": "xG Against", "Away Team Salary": "Salary",
                 "Home Team Salary": "Opponent Salary"}
    )
    away_stats["Home?"] = 0

    combined_stats = pd.concat([home_stats, away_stats])
    combined_stats.sort_values(["Date", "Team"], inplace=True)

    # Calculate average goal difference for home and away team.
    combined_stats["Match G/D"] = combined_stats["Goals For"] - combined_stats["Goals Against"]
    combined_stats['Average G/D'] = combined_stats.groupby(["Season", "Team"],
                                                            observed=True)["Match G/D"].transform("mean")
    combined_stats['Opponent Average G/D'] = (combined_stats.groupby(["Season", "Opponent"],                                                                                                                                  
                                                            observed=True)["Match G/D"].transform("mean")) * -1
    
    # Calculate average xG, xGA and xG/D for Team
    combined_stats['Average xG For'] = (combined_stats.groupby(["Season", "Team"],
                                                            observed=True)["xG For"].transform("mean"))
    combined_stats['Average xG Against'] = (combined_stats.groupby(["Season", "Team"],
                                                            observed=True)["xG Against"].transform("mean"))
    combined_stats['Average xG/D'] = combined_stats['Average xG For'] - combined_stats['Average xG Against']
    
    # Calculate average xG, xGA, and xG/D for Opponent
    combined_stats['Opponent Average xG For'] = (combined_stats.groupby(["Season", "Opponent"],
                                                            observed=True)["xG For"].transform("mean"))
    combined_stats['Opponent Average xG Against'] = (combined_stats.groupby(["Season", "Opponent"],
                                                            observed=True)["xG Against"].transform("mean"))
    combined_stats['Opponent Average xG/D'] = (combined_stats['Opponent Average xG For'] - combined_stats['Opponent Average xG Against']) * -1

    # Calculate Number of Days Rest
    combined_stats["Days Rest"] = combined_stats.groupby(["Season", "Team"], observed=True)["Date"].diff().dt.days
    combined_stats["Days Rest"] = combined_stats["Days Rest"].fillna(0)

    # Calculate exponential rolling xG averages
    combined_stats['Rolling xG For'] = combined_stats.groupby(['Team', "Season"], observed=True)['xG For'].transform(
        lambda x: x.ewm(span=10, adjust=False).mean().shift(1)).fillna(0)
    combined_stats['Rolling xG Against'] = combined_stats.groupby(['Team',"Season"],observed=True)['xG Against'].transform(
        lambda x: x.ewm(span=10, adjust=False).mean().shift(1)).fillna(0)

    combined_stats['Opponent Rolling xG For'] = combined_stats.groupby(['Opponent', "Season"], observed=True)['xG Against'].transform(
        lambda x: x.ewm(span=10, adjust=False).mean().shift(1)).fillna(0)
    combined_stats['Opponent Rolling xG Against'] = combined_stats.groupby(['Opponent',"Season"],observed=True)['xG For'].transform(
        lambda x: x.ewm(span=10, adjust=False).mean().shift(1)).fillna(0)

    combined_stats["Rolling xG/D"] = combined_stats["Rolling xG For"] - combined_stats["Rolling xG Against"]
    combined_stats["Opponent Rolling xG/D"] = combined_stats["Opponent Rolling xG For"] - combined_stats["Opponent Rolling xG Against"]


    return combined_stats


#===========================FEATURE ENGINEERING===========================#
def engineer_features(normalized_stats):
    x = normalized_stats[["Home?", "Days Rest", "Average xG For", "Average xG/D", "Rolling xG For", "Rolling xG/D", "Average G/D", "Salary", 
                          "Opponent Average xG Against", "Opponent Average xG/D", "Opponent Rolling xG Against", "Opponent Rolling xG/D", 
                          "Opponent Average G/D", "Opponent Salary"]]


    y = normalized_stats["xG For"]

    x = x.copy()

    x["Team Strength"] = x[["Average xG For", "Average xG/D", "Rolling xG For", "Rolling xG/D", "Average G/D", "Salary"]].apply(pd.to_numeric).mean(axis=1)
    x["Opponent Team Strength"] = x[[ "Opponent Average xG Against", "Opponent Average xG/D", "Opponent Rolling xG Against", "Opponent Rolling xG/D", 
                            "Opponent Average G/D", "Opponent Salary"]].apply(pd.to_numeric).mean(axis=1)

    x = x[["Home?", "Days Rest", "Team Strength", "Opponent Team Strength"]].apply(pd.to_numeric)

    return x,y




df = load_and_prepare_match_data(r"data\2024_match_stats.csv")
salaries = load_and_prepare_salary_data(r"data\prem_salaries.csv")
normalized_stats = calculate_normalized_stats(df,salaries)
x,y = engineer_features(normalized_stats)
