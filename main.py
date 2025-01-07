from flask import Flask, render_template
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score

def calculate_away_wins(group):
    return len(group) - group['HomeWon'].sum()

last_season_data = pd.read_csv("23-24.csv")
this_season_data = pd.read_csv("24-25.csv")
week_14 = pd.read_csv("Week_14.csv")
week_15 = pd.read_csv("Week_15.csv")
week_16 = pd.read_csv("Week_16.csv")
week_17 = pd.read_csv("Week_17.csv")
week_18 = pd.read_csv("Week_18.csv")

def predictions(week):
    # Data
    all_data = pd.concat([last_season_data,this_season_data])

    # Average points scored (Home and Away)
    avg_home_score = all_data.groupby('Home')['HomeScore'].mean()
    avg_away_score = all_data.groupby('Visitor')['VisitorScore'].mean()
    overall_avg_points_scored = (avg_home_score + avg_away_score) / 2

    # Average points given up (Home and Away)
    avg_home_allowed = all_data.groupby('Home')['VisitorScore'].mean()
    avg_away_allowed = all_data.groupby('Visitor')['HomeScore'].mean()
    overall_avg_points_allowed = (avg_home_allowed + avg_away_allowed) / 2

    # Total wins (Home and Away)
    home_wins = all_data.groupby('Home')['HomeWon'].sum()
    away_wins = all_data.groupby('Visitor').apply(calculate_away_wins, include_groups=False)
    total_wins = home_wins + away_wins

    # Total games (Home and Away)
    total_home_games = all_data['Home'].value_counts()
    total_away_games = all_data['Visitor'].value_counts()
    total_games = total_home_games + total_away_games

    # Win rate
    win_rate = total_wins / total_games

    # Good plays (Home and Away)
    all_data['SuccessfulPlay'] = all_data['IsTouchdown'] | (~all_data['IsFumble'] & ~all_data['IsInterception'])
    overall_avg_conceded_plays = (all_data.groupby('Home')['SuccessfulPlay'].mean() + all_data.groupby('Visitor')['SuccessfulPlay'].mean()) / 2

    # Bad plays (Home and Away)
    all_data['Turnover'] = all_data['IsFumble'] | all_data['IsInterception']
    overall_avg_forced_turnovers = (all_data.groupby('Home')['Turnover'].mean() + all_data.groupby('Visitor')['Turnover'].mean()) / 2

    # Offensive yards (Home and Away)
    overall_avg_yards_per_play = (all_data.groupby('Home')['Yards'].mean() + all_data.groupby('Visitor')['Yards'].mean()) / 2
    overall_avg_yards_per_game = (all_data.groupby(['SeasonYear', 'Home'])['Yards'].sum() / all_data.groupby(['SeasonYear', 'Home']).size() + all_data.groupby(['SeasonYear', 'Visitor'])['Yards'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()).groupby(level=1).mean()

    # QB stuff (Home and Away)
    overall_avg_pass_completion_rate = (all_data.groupby('Home').apply(lambda x: 1 - x['IsIncomplete'].mean(), include_groups=False) + all_data.groupby('Visitor').apply(lambda x: 1 - x['IsIncomplete'].mean(),include_groups=False)) / 2
    overall_avg_touchdowns_per_game = (all_data.groupby(['SeasonYear', 'Home'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Home']).size() + all_data.groupby(['SeasonYear', 'Visitor'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()).groupby(level=1).mean()

    # Running (Home and Away)
    overall_avg_rush_success_rate = (all_data.groupby('Home').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean(), include_groups=False) + all_data.groupby('Visitor').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean(), include_groups=False)) / 2

    # Defensive yards (Home and Away)
    overall_avg_yards_allowed_per_play = (all_data.groupby('Home')['Yards'].mean() + all_data.groupby('Visitor')['Yards'].mean()) / 2
    total_yards_allowed_per_game_home = all_data.groupby(['SeasonYear', 'Home'])['Yards'].sum() / all_data.groupby(['SeasonYear', 'Home']).size()
    total_yards_allowed_per_game_visitor = all_data.groupby(['SeasonYear', 'Visitor'])['Yards'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()
    overall_avg_yards_allowed_per_game = (total_yards_allowed_per_game_home + total_yards_allowed_per_game_visitor).groupby(level=1).mean()

    # Completion percentage (Home and Away)
    overall_avg_pass_completion_allowed_rate = (all_data.groupby('Home').apply(lambda x: 1 - x['IsIncomplete'].mean(), include_groups=False) + all_data.groupby('Visitor').apply(lambda x: 1 - x['IsIncomplete'].mean(),include_groups=False)) / 2

    # Allowed tuddy (Home and Away)
    overall_avg_touchdowns_allowed_per_game = (all_data.groupby(['SeasonYear', 'Home'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Home']).size() + all_data.groupby(['SeasonYear', 'Visitor'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()).groupby(level=1).mean()
    overall_avg_rush_success_allowed_rate = (all_data.groupby('Home').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean(), include_groups=False) + all_data.groupby('Visitor').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean(), include_groups=False)) / 2

    # Add points scored, winrate, good plays, bad plays, offensive stats, and defensive stats
    team_features = pd.DataFrame({
        'AvgPointsScored': overall_avg_points_scored,
        'AvgPointsAllowed': overall_avg_points_allowed,
        'WinRate': win_rate,
        'AvgPointsDefended': overall_avg_points_allowed,
        'AvgConcededPlays': overall_avg_conceded_plays.values,
        'AvgForcedTurnovers': overall_avg_forced_turnovers.values,
        'AvgYardsPerPlay': overall_avg_yards_per_play.values,
        'AvgYardsPerGame': overall_avg_yards_per_game.values,
        'AvgPassCompletionRate': overall_avg_pass_completion_rate.values,
        'AvgTouchdownsPerGame': overall_avg_touchdowns_per_game.values,
        'AvgRushSuccessRate': overall_avg_rush_success_rate.values,
        'AvgYardsAllowedPerPlay': overall_avg_yards_allowed_per_play.values,
        'AvgYardsAllowedPerGame': overall_avg_yards_allowed_per_game.values,
        'AvgPassCompletionAllowedRate': overall_avg_pass_completion_allowed_rate.values,
        'AvgTouchdownsAllowedPerGame': overall_avg_touchdowns_allowed_per_game.values,
        'AvgRushSuccessAllowedRate': overall_avg_rush_success_allowed_rate.values
    })
    team_features.reset_index(inplace=True)
    team_features.rename(columns={'Home': 'Team'}, inplace=True)

    # Defence for home team
    upcoming_encoded_home = week.merge(team_features, left_on='Home', right_on='Team', how='left')
    # Offense for home team
    upcoming_encoded_both = upcoming_encoded_home.merge(team_features, left_on='Visitor', right_on='Team', suffixes=('_Home', '_Visitor'), how='left')

    # Difference in offense and defence for home team
    for col in ['AvgPointsScored', 'AvgPointsAllowed', 'WinRate', 'AvgPointsDefended', 'AvgConcededPlays', 'AvgForcedTurnovers',
                'AvgYardsPerPlay', 'AvgYardsPerGame', 'AvgPassCompletionRate', 'AvgTouchdownsPerGame', 'AvgRushSuccessRate',
                'AvgYardsAllowedPerPlay', 'AvgYardsAllowedPerGame', 'AvgPassCompletionAllowedRate', 'AvgTouchdownsAllowedPerGame', 'AvgRushSuccessAllowedRate']:
        upcoming_encoded_both[f'Diff_{col}'] = upcoming_encoded_both[f'{col}_Home'] - upcoming_encoded_both[f'{col}_Visitor']

    # Combine everything
    upcoming_encoded_final = upcoming_encoded_both[['Home', 'Visitor'] + [col for col in upcoming_encoded_both.columns if 'Diff_' in col]]

    # Score prediction
    def predict_team_score(team, avg_points_scored, avg_points_allowed):
        # For simplicity, this can be just an estimate based on historical averages.
        team_avg_score = avg_points_scored.get(team, 0)
        team_avg_allowed = avg_points_allowed.get(team, 0)
        
        # A simple prediction could be based on average points scored and points allowed
        predicted_score = (team_avg_score + team_avg_allowed)/2  # Home team advantage factored in
        return predicted_score

    # Estimate predicted scores for both teams
    upcoming_encoded_final['HomePredictedScore'] = upcoming_encoded_final['Home'].apply(
        lambda team: predict_team_score(team, avg_home_score, avg_away_allowed)
    )

    upcoming_encoded_final['VisitorPredictedScore'] = upcoming_encoded_final['Visitor'].apply(
        lambda team: predict_team_score(team, avg_away_score, avg_home_allowed)
    )

    # Merge all the data with the data frame
    training_encoded_home = all_data.merge(team_features, left_on='Home', right_on='Team', how='left')
    training_encoded_both = training_encoded_home.merge(team_features, left_on='Visitor', right_on='Team', suffixes=('_Home', '_Visitor'), how='left')

    # Calculate the difference in features
    for col in ['AvgPointsScored', 'AvgPointsAllowed', 'WinRate', 'AvgPointsDefended', 'AvgConcededPlays', 'AvgForcedTurnovers',
                'AvgYardsPerPlay', 'AvgYardsPerGame', 'AvgPassCompletionRate', 'AvgTouchdownsPerGame', 'AvgRushSuccessRate',
                'AvgYardsAllowedPerPlay', 'AvgYardsAllowedPerGame', 'AvgPassCompletionAllowedRate', 'AvgTouchdownsAllowedPerGame', 'AvgRushSuccessAllowedRate']:
        training_encoded_both[f'Diff_{col}'] = training_encoded_both[f'{col}_Home'] - training_encoded_both[f'{col}_Visitor']

    # Filtering out the required columns for the logistic regression
    training_data = training_encoded_both[[col for col in training_encoded_both.columns if 'Diff_' in col]]
    training_labels = training_encoded_both['HomeWon']

    # Initialize Logistic Regression model for home win prediction
    logreg = LogisticRegression(max_iter=1000)

    # Cleaning data for Logistic Regression
    training_data_cleaned = training_data.dropna()
    training_labels_cleaned = training_labels.loc[training_data_cleaned.index]

    # Train the logistic regression model on the entire cleaned training dataset
    logreg.fit(training_data_cleaned, training_labels_cleaned)

    # Predict the probability of the home team winning for the upcoming games
    upcoming_game_probabilities = logreg.predict_proba(upcoming_encoded_final[[col for col in upcoming_encoded_final.columns if 'Diff_' in col]])

    # Extract the probability that the home team will win (second column of the result)
    upcoming_game_prob_home_win = upcoming_game_probabilities[:, 1]

    # Add the predictions to the upcoming games dataframe
    upcoming_encoded_final['HomeWinProbability'] = upcoming_game_prob_home_win

    # Initialize Linear Regression model to predict final scores
    linear_reg_home = LinearRegression()
    linear_reg_visitor = LinearRegression()

    # Preparing features for score prediction
    score_features = upcoming_encoded_final[[col for col in upcoming_encoded_final.columns if 'Diff_' in col]]

    # Train Linear Regression for home and visitor scores (you can use home team's avg score, visitor team's avg score)
    linear_reg_home.fit(score_features, upcoming_encoded_final['HomePredictedScore'])
    linear_reg_visitor.fit(score_features, upcoming_encoded_final['VisitorPredictedScore'])

    # Add predicted scores to the upcoming predictions
    upcoming_predictions = upcoming_encoded_final[['Home', 'Visitor', 'HomePredictedScore', 'VisitorPredictedScore', 'HomeWinProbability']]

    # Show the final predictions
    print(upcoming_predictions)

predictions(week_18)