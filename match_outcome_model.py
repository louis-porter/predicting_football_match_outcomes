import numpy as np
from scipy.stats import poisson


def calculate_match_outcome_prob(xG_home, xG_away):
    max_goals = 10

    # Poisson probabilities for each team scoring from 0 to max_goals
    prob_home = poisson.pmf(np.arange(max_goals+1), xG_home) # np.arange is used here to create a list of integers 0 through max goals
    print([f"{p * 100:.1f}%" for p in prob_home])
    prob_away = poisson.pmf(np.arange(max_goals+1), xG_away)
    print([f"{p * 100:.1f}%" for p in prob_away])

    # Calculate probability of both seems scoring the same number of goals (draw)
    prob_draw = np.sum(prob_home * prob_away) 

    # Calculate probability of home team scoring more goals than away team (home win)
    prob_home_win = sum(prob_home[i] * sum(prob_away[:i]) for i in range (1, max_goals+1))

    # Calculate probability of away team scoring more goals than home team (away win)
    prob_away_win = sum(prob_away[i] * sum(prob_home[:i]) for i in range (1, max_goals+1))

    print(f"Home win %: {round(prob_home_win,2)} | Draw %: {round(prob_draw, 2)} | Away win %: {round(prob_away_win,2)}")

    return prob_home_win, prob_draw, prob_away_win



def simulate_match(prob_home_win, prob_draw, prob_away_win):
    outcomes = ["Home win", "Draw", "Away win"]

    # Calculating total probablity
    total_prob = prob_home_win + prob_draw + prob_away_win
    probabilities = [prob_home_win, prob_draw, prob_away_win]

    # Normalising the probablities to ensure sum = 1
    normalised_probs = probabilities / total_prob

    # Simulate the outcome based on supplied probabilities
    result = np.random.choice(outcomes, p=normalised_probs)
    return result


def test_simulation(n=10000):
    results = {"Home win":0, "Draw": 0, "Away win": 0}
    for i in range(n):
        outcome = simulate_match(home_win, draw, away_win)
        results[outcome] += 1
    print(results)
    

home_win, draw, away_win = calculate_match_outcome_prob(1.36, 1.5)

