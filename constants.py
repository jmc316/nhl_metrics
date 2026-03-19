from nhlpy import NHLClient

# Create an instance of the NHLClient
nhl_client = NHLClient()

# format is 'Option Name': ['module_name', 'function_name']
main_options = {
    'NHL Team Stats': ['team_stats', 'nhl_team_stats'],
    'Predictions': ['skl_predictions', 'season_predictions'],
}

exit_option = {'Exit': ['terminal_ui', 'exit_program']}

team_stats_options = {
    'Standings': ['team_stats', 'nhl_team_standings'],
    'Individual Team Stats': ['team_stats', 'nhl_individual_team_stats'],
}

ot_score_diff = 0.25
max_single_season_games = 110

game_id_col = 'gameId'
season_col = 'season'
season_name_col = 'seasonName'
starttime_utc_col = 'startTimeUTC'
game_time_col = 'gameTimeUTC'
venue_timezone_col = 'venueTimezone'
venue_col = 'venue'
away_team_name_col = 'awayTeamName'
home_team_name_col = 'homeTeamName'
away_team_score_col = 'awayTeamScore'
home_team_score_col = 'homeTeamScore'
last_period_col = 'lastPeriod'
default_col = 'default'
team_name_col = 'teamName'

away_team_col = 'awayTeam'
home_team_col = 'homeTeam'
home_team_games_col = 'homeTeamGames'
away_team_games_col = 'awayTeamGames'
home_team_points_col = 'homeTeamPoints'
away_team_points_col = 'awayTeamPoints'
home_team_wins_col = 'homeTeamWins'
away_team_wins_col = 'awayTeamWins'
home_team_losses_col = 'homeTeamLosses'
away_team_losses_col = 'awayTeamLosses'
home_team_otls_col = 'homeTeamOTLs'
away_team_otls_col = 'awayTeamOTLs'
home_team_reg_wins_col = 'homeTeamRegWins'
away_team_reg_wins_col = 'awayTeamRegWins'
home_team_reg_ot_wins_col = 'homeTeamRegOTWins'
away_team_reg_ot_wins_col = 'awayTeamRegOTWins'
home_team_so_wins_col = 'homeTeamSOWins'
away_team_so_wins_col = 'awayTeamSOWins'
home_team_so_losses_col = 'homeTeamSOLosses'
away_team_so_losses_col = 'awayTeamSOLosses'
home_team_goals_for_col = 'homeTeamGoalsFor'
away_team_goals_for_col = 'awayTeamGoalsFor'
home_team_goals_against_col = 'homeTeamGoalsAgainst'
away_team_goals_against_col = 'awayTeamGoalsAgainst'
goal_diff_col = 'goalDifferential'
points_percentage_col = 'pointsPercentage'
division_name_col = 'divisionName'
wildcard_sequence_col = 'wildcardSequence'
division_sequence_col = 'divisionSequence'
conference_name_col = 'conferenceName'
division_seed_col = 'divisionSeed'
conference_seed_col = 'conferenceSeed'
playoff_seed_col = 'playoffSeed'
total_points_col = 'totalPoints'
total_wins_col = 'totalWins'
total_losses_col = 'totalLosses'
total_otls_col = 'totalOTLs'
playoff_per_col = 'playoff_%'
total_goals_for_col = 'totalGoalsFor'
total_goals_against_col = 'totalGoalsAgainst'
total_games_col = 'totalGames'
game_outcome_col = 'gameOutcome'
game_type_col = 'gameType' # 1 = preseason, 2 = regular season, 3 = playoffs, 4 = all-star game
home_team_prev_10_wins_col = 'homeTeamPrev10Wins'
away_team_prev_10_wins_col = 'awayTeamPrev10Wins'
home_team_prev_10_losses_col = 'homeTeamPrev10Losses'
away_team_prev_10_losses_col = 'awayTeamPrev10Losses'
home_team_prev_10_otl_col = 'homeTeamPrev10OTLs'
away_team_prev_10_otl_col = 'awayTeamPrev10OTLs'
home_team_prev_n_goals_for_col = 'homeTeamPrevGoalsFor_'
away_team_prev_n_goals_for_col = 'awayTeamPrevGoalsFor_'
home_team_prev_n_goals_against_col = 'homeTeamPrevGoalsAgainst_'
away_team_prev_n_goals_against_col = 'awayTeamPrevGoalsAgainst_'
game_date_col = 'gameDate'
home_team_days_since_last_game_col = 'homeTeamDaysSinceLastGame'
away_team_days_since_last_game_col = 'awayTeamDaysSinceLastGame'
home_team_travel_distance_7days_col = 'homeTeamTravelDistance7Days'
away_team_travel_distance_7days_col = 'awayTeamTravelDistance7Days'
home_team_points_percentage_col = 'homeTeamPointsPercentage'
away_team_points_percentage_col = 'awayTeamPointsPercentage'
home_team_prev_n_wins_col = 'homeTeamPrevWins_'
away_team_prev_n_wins_col = 'awayTeamPrevWins_'
home_team_prev_n_losses_col = 'homeTeamPrevLosses_'
away_team_prev_n_losses_col = 'awayTeamPrevLosses_'
home_team_prev_n_otls_col = 'homeTeamPrevOTLs_'
away_team_prev_n_otls_col = 'awayTeamPrevOTLs_'
home_team_prev_n_points_percentage_col = 'homeTeamPrevPointsPercentage_'
away_team_prev_n_points_percentage_col = 'awayTeamPrevPointsPercentage_'

date_format_yyyy_mm_dd = '%Y-%m-%d'
div_1_val = 'div_1'
div_2_val = 'div_2'
div_3_val = 'div_3'
wc_1_val = 'wc_1'
wc_2_val = 'wc_2'
missed_val = 'Missed'
atl_div_val = 'Atlantic'
metro_div_val = 'Metropolitan'
pac_div_val = 'Pacific'
cen_div_val = 'Central'
season_stdt = '09-23'
season_enddt = '06-30'

predict_cols = [home_team_score_col, away_team_score_col, last_period_col]
tiebreaker_cols = ['totalPoints', 'pointsPercentage', 'totalRegWins', 'totalRegOTWins', 'totalWins', 'goalDifferential', 'totalGoalsFor']
final_standings_col_order = ['conferenceName', 'conferenceSeed', 'divisionName', 'divisionSeed', 'playoffSeed', 'teamName', 'totalGames',
                 'totalWins', 'totalLosses', 'totalOTLs', 'totalPoints', 'pointsPercentage', 'totalRegWins', 'totalRegOTWins',
                 'goalDifferential', 'totalGoalsFor']


output_folder = 'output/'
util_data_folder = 'util_data/'
model_files_folder = 'model_files/'
season_sched_folder = output_folder + 'season_schedules/'
season_feature_sets_folder = output_folder + 'season_feature_sets/'
season_sched_filename = 'season_sched.csv'
season_sched_pred_filename = 'season_sched_pred.csv'
season_prediction_filename = 'season_prediction'
season_sched_pred_points_filename = 'season_sched_pred_points.csv'
final_standings_filename = 'final_standings.csv'
venue_geolocations_filename = 'venue_geolocations.csv'
feature_data_filename_suffix = '_feature_data.csv'
sklearn_model_filename = 'skl_rf_model.pkl'
playoff_spot_prediction_filename = 'playoff_predictions_{n}.csv'

team_colors = {
    'Anaheim Ducks': 'cyan',
    'Boston Bruins': 'yellow',
    'Buffalo Sabres': 'blue',
    'Calgary Flames': 'red',
    'Carolina Hurricanes': 'red',
    'Chicago Blackhawks': 'red',
    'Colorado Avalanche': 'blue',
    'Columbus Blue Jackets': 'blue',
    'Dallas Stars': 'green',
    'Detroit Red Wings': 'red',
    'Edmonton Oilers': 'blue',
    'Florida Panthers': 'red',
    'Los Angeles Kings': 'light_grey',
    'Minnesota Wild': 'green',
    'Montréal Canadiens': 'red',
    'Nashville Predators': 'yellow',
    'New Jersey Devils': 'red',
    'New York Islanders': 'blue',
    'New York Rangers': 'blue',
    'Ottawa Senators': 'red',
    'Philadelphia Flyers': 'white',
    'Pittsburgh Penguins': 'yellow',
    'San Jose Sharks': 'cyan',
    'Seattle Kraken': 'cyan',
    'St. Louis Blues': 'blue',
    'Tampa Bay Lightning': 'blue',
    'Toronto Maple Leafs': 'blue',
    'Utah Mammoth': 'cyan',
    'Vancouver Canucks': 'blue',
    'Vegas Golden Knights': 'dark_grey',
    'Washington Capitals': 'red',
    'Winnipeg Jets': 'light_grey'
}