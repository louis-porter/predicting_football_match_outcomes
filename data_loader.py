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
    df = pd.read_csv(filepath, dtype=dtype)
    return df




#===========================STATISTICAL CALCULATIONS===========================#
def calculate_cumulative_goal_difference(df):
    # Create column that tracks home and away goal difference.
    df["Home G/D"] = df["Home Goals"] - df["Away Goals"]
    df["Away G/D"] = df["Away Goals"] - df["Home Goals"]

    home_stats = df[["Date", "GW", "Home Team", "Home G/D"]].rename(
        columns={"Home Team": "Team", "Home G/D": "G/D"}
    )

    away_stats = df[["Date", "GW", "Away Team", "Away G/D"]].rename(
        columns={"Away Team": "Team", "Away G/D": "G/D"}
    )

    goal_diff_data = pd.concat([home_stats, away_stats])
    goal_diff_data = goal_diff_data.sort_values("Date")
    goal_diff_data["Cumulative G/D"] = goal_diff_data.groupby("Team", observed=True)["G/D"].cumsum()
    goal_diff_data["Cumulative G/D"] = goal_diff_data["Cumulative G/D"] / goal_diff_data["GW"]


    df = df.merge(goal_diff_data[["Date", "Team", "Cumulative G/D"]], left_on=["Date", "Home Team"],
                right_on=["Date", "Team"], how="left")
    df.rename(columns={"Cumulative G/D": "Home Cumulative G/D"}, inplace=True)
    df.drop("Team", axis=1, inplace=True)

    df = df.merge(goal_diff_data[["Date", "Team", "Cumulative G/D"]], left_on=["Date", "Away Team"],
                right_on=["Date", "Team"], how="left")
    df.rename(columns={"Cumulative G/D": "Away Cumulative G/D"}, inplace=True)
    df.drop("Team", axis=1, inplace=True)
    return df

def calculate_average_gd(df):
    # Create column that tracks home and away goal difference.
    df["Home G/D"] = df["Home Goals"] - df["Away Goals"]
    df["Away G/D"] = df["Away Goals"] - df["Home Goals"]

    home_stats = df[["Season", "Date", "GW", "Home Team", "Home G/D"]].rename(
        columns={"Home Team": "Team", "Home G/D": "G/D"}
    )

    away_stats = df[["Season", "Date", "GW", "Away Team", "Away G/D"]].rename(
        columns={"Away Team": "Team", "Away G/D": "G/D"}
    )

    goal_diff_data = pd.concat([home_stats, away_stats])
    goal_diff_data = goal_diff_data.sort_values("Date")
    goal_diff_data.set_index(["Season", "Team", "Date"], inplace=True)
    
    group = goal_diff_data.groupby(["Season", "Team"], observed=True)
    goal_diff_data["Average G/D"] = group["G/D"].transform("mean")
    goal_diff_data_reset = group.mean().reset_index()

    
    df = df.merge(goal_diff_data_reset[["Season", "Team", "Average G/D"]], left_on=["Season", "Home Team"],
                right_on=["Season", "Team"], how="left")
    df.rename(columns={"Average G/D": "Home AvG G/D"}, inplace=True)
    df.drop("Team", axis=1, inplace=True)

    df = df.merge(goal_diff_data_reset[["Season", "Team", "Average G/D"]], left_on=["Season", "Away Team"],
                right_on=["Season","Team"], how="left")
    df.rename(columns={"Average G/D": "Away Avg G/D"}, inplace=True)
    df.drop("Team", axis=1, inplace=True)
    df.drop(["Home G/D", "Away G/D"], axis=1, inplace=True)

    return df

def calculate_average_xGD(df):
    # Create a column that tracks average xG for and xG against.
    home_xg_stats = df[["Season", "GW", "Date", "Home Team", "Home xG", "Away xG"]].rename(
        columns={"Home Team": "Team", "Home xG": "xG For", "Away xG": "xG Against"}
    )
    away_xg_stats = df[["Season", "GW", "Date", "Away Team", "Away xG", "Home xG"]].rename(
        columns={"Away Team": "Team", "Away xG": "xG For", "Home xG": "xG Against"}
    )
    xg_data = pd.concat([home_xg_stats, away_xg_stats])
    xg_data.sort_values(["Season", "Team", "Date"], inplace=True)
    xg_data.set_index(["Season", "Team", "Date"], inplace=True)

    group = xg_data.groupby(["Season", "Team"], observed=True)
    xg_data["Average xG For"] = group["xG For"].transform("mean")
    xg_data["Average xG Against"] = group["xG Against"].transform("mean")
    xg_data["Average xG/D"] = xg_data["Average xG For"] - xg_data["Average xG Against"]

    xg_data_reset = group.mean().reset_index()

    df = df.merge(xg_data_reset[["Season", "Team", "Average xG For", "Average xG Against", "Average xG/D"]], 
                left_on=["Home Team", "Season"], right_on=["Team", "Season"], how="left")
    df.rename(columns={"Average xG For": "Home Avg xG", "Average xG Against": "Home Avg xGA",
                       "Average xG/D": "Home avg xG/D"}, inplace=True)
    df.drop("Team", axis=1, inplace=True)

    df = df.merge(xg_data_reset[["Season", "Team", "Average xG For", "Average xG Against", "Average xG/D"]], 
                left_on=["Away Team", "Season"], right_on=["Team", "Season"], how="left")
    df.rename(columns={"Average xG For": "Away Avg xG", "Average xG Against": "Away Avg xGA",
                       "Average xG/D": "Away Avg xG/D"}, inplace=True)
    df.drop("Team", axis=1, inplace=True)
    
    return df


def calculate_normalised_salary(df, salary):
    salary["Salary P/W (K)"] = salary["Salary P/W (K)"]/1000 # Normalise to M units
    df = df.merge(salary, left_on=["Home Team", "Season"], right_on=["Team", "Season"], how="left")
    df.rename(columns={"Salary P/W (K)": "Home Team Salary"}, inplace=True)
    df = df.merge(salary, left_on=["Away Team", "Season"], right_on=["Team", "Season"], how="left")
    df.rename(columns={"Salary P/W (K)": "Away Team Salary"}, inplace=True)
    return df






#===========================FEATURE ENGINEERING===========================#
def calculate_team_strength(df):
    #Create a measure of team strength based on xG/D, G/D and salary.
    df = calculate_average_xGD(df)
    df = calculate_average_gd(df)
    df = calculate_normalised_salary(df, salaries)

    scaler = StandardScaler()
    print(df.columns)
    columns_to_scale = ["Home avg xG/D", "Home AvG G/D", "Home Team Salary", "Away Avg xG/D", "Away Avg G/D", "Away Team Salary"]
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    df["Home Team Strength"] = (0.75 * df["Home avg xG/D"]) + (0.2 * df["Home AvG G/D"]) + (0.05 * df["Home Team Salary"])
    df["Away Team Strength"] = (0.75 * df["Away Avg xG/D"]) + (0.2 * df["Away Avg G/D"]) + (0.05 * df["Away Team Salary"])
    df.to_csv("test.csv")
    print(df[["Away Team", "Season", "Away Team Strength"]])
    return df
    




def calculate_form(df):
# Create a column that tracks exponential 10-game xG
    home_form_stats = df[["GW", "Date", "Home Team", "Home xG", "Away xG"]].rename(
        columns={"Home Team": "Team", "Home xG":"xG For", "Away xG": "xG Against"}
    )
    away_form_stats = df[["GW", "Date", "Away Team", "Away xG", "Home xG",]].rename(
        columns={"Away Team": "Team", "Away xG":"xG For", "Home xG": "xG Against"}
    )
    form_data = pd.concat([home_form_stats, away_form_stats])
    form_data.sort_values(["Team", "Date"], inplace=True)
    form_data.set_index(["Team", "Date"], inplace=True)
    
    form_data['rolling xG For'] = form_data.groupby(level='Team', observed=True)['xG For'].transform(lambda x: x.ewm(span=10, adjust=False).mean().shift(1))
    form_data['rolling xG Against'] = form_data.groupby(level='Team',observed=True)['xG Against'].transform(lambda x: x.ewm(span=10, adjust=False).mean().shift(1))
    form_data_reset = form_data.reset_index()

    df = df.merge(form_data_reset[["Date", "Team", "rolling xG For", "rolling xG Against"]], left_on=["Date", "Home Team"],
                  right_on=["Date", "Team"], how="left")
    df = df.rename(columns={"rolling xG Against": "Home rolling xG Against", "rolling xG For": "Home rolling xG For"})
    df.drop("Team", axis=1, inplace=True)
    df = df.merge(form_data_reset[["Date", "Team", "rolling xG Against", "rolling xG For"]], left_on=["Date", "Away Team"],
                  right_on=["Date", "Team"], how="left")
    df = df.rename(columns={"rolling xG Against": "Away rolling xG Against", "rolling xG For": "Away rolling xG For"})
    df.drop("Team", axis=1, inplace=True)
    return df
    



def calculate_days_rest(df):
# Create a column that tracks number of days since previous match.
    df["Date"] = pd.to_datetime(df["Date"])

    home_date_stats = df[["GW", "Home Team", "Date"]].rename(
        columns={"Home Team": "Team"}
    )
    away_date_stats = df[["GW", "Away Team", "Date"]].rename(
        columns={"Away Team": "Team"}
    )

    dates_data = pd.concat([home_date_stats, away_date_stats])
    dates_data.sort_values(["Team", "Date"], inplace=True)
    dates_data.set_index(["Team", "GW"], inplace=True)

    dates_data["Days Rest"] = dates_data.groupby(level=0, observed=True)["Date"].diff().dt.days
    dates_data_reset = dates_data.reset_index()

    df = df.merge(dates_data_reset[["Date", "Team", "Days Rest"]], left_on=["Date", "Home Team"],
                right_on=["Date", "Team"], how="left")
    df.rename(columns={"Days Rest": "Home Days Rest"}, inplace=True)
    df.drop("Team", axis=1, inplace=True)

    df = df.merge(dates_data_reset[["Date", "Team", "Days Rest"]], left_on=["Date", "Away Team"],
                right_on=["Date", "Team"], how="left")
    df.rename(columns={"Days Rest": "Away Days Rest"}, inplace=True)
    df.drop("Team", axis=1, inplace=True)
    return df



def calculate_final_df(df):
# Create the final dataframe to be used in other modules

    home_data = df[["GW", "Home Team", "Home xG", "Home Avg xG", "Home Avg xGA", "Home Cumulative G/D",  "Home Days Rest",
                     "Away xG", "Away Avg xGA", "Away Avg xG", "Home rolling xG For", "Away rolling xG Against",
                     "Home rolling xG Against", "Away rolling xG For"]].rename(
                        columns={"Home xG": "xG For", "Away xG": "xG Against", "Home Avg xG": "Avg xG", "Home Avg xGA": "Avg xGA",
                                "Home Cumulative G/D": "Cumulative G/D",   "Home Days Rest":
                                "Days Rest", "Home GW Points": "GW Pts", "Home Team": "Team", "Away Avg xGA": "Opponent Avg xGA",
                                "Away Avg xG": "Opponent Avg xG", "Home rolling xG For":"rolling xG For", "Home rolling xG Against":
                                "rolling xG Against", "Away rolling xG For": "Opponent rolling xG For", "Away rolling xG Against":
                                "Opponent rolling xG Against"})
    home_data["Home?"] = 1


    away_data = df[["GW", "Away Team", "Away xG", "Away Avg xG", "Away Avg xGA", "Away Cumulative G/D",  "Away Days Rest",
                     "Home xG", "Home Avg xGA", "Home Avg xG","Home rolling xG For", "Away rolling xG Against",
                     "Home rolling xG Against", "Away rolling xG For"]].rename(
                        columns={"Away xG": "xG For", "Home xG": "xG Against", "Away Avg xG": "Avg xG", "Away Avg xGA": "Avg xGA",
                                "Away Cumulative G/D": "Cumulative G/D", "Away Days Rest":
                                "Days Rest" , "Away Team": "Team",  "Home Avg xGA": "Opponent Avg xGA",
                                "Home Avg xG": "Opponent Avg xG", "Away rolling xG For":"rolling xG For", "Away rolling xG Against":
                                "rolling xG Against", "Home rolling xG For": "Opponent rolling xG For", "Home rolling xG Against":
                                "Opponent rolling xG Against"})
    away_data["Home?"] = 0

    model_df = pd.concat([home_data, away_data]).sort_values(["GW"])
    return model_df


df = load_and_prepare_match_data(r"data\2024_match_stats.csv")
salaries = load_and_prepare_salary_data(r"C:\Users\Owner\dev\Premier League Simulation\data\prem_salaries.csv")
df = calculate_team_strength(df)
#df = calculate_days_rest(df)
#df = calculate_form(df)
#model_df = calculate_final_df(df)

