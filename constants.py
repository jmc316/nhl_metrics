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
playoff_sched_format = [0, 2, 5, 7, 10, 12, 15]
final_sched_format = [0, 3, 6, 9, 12, 15, 18]
playoff_round_buffer = 2
api_timeout_wait_time = 3

game_id_col = 'gameId'
season_col = 'season'
season_name_col = 'seasonName'
starttime_utc_col = 'startTimeUTC'
starttime_est_col = 'startTimeEST'
game_time_col = 'gameTimeEST'
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
wildcard_seed_col = 'wildcardSeed'
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
home_team_series_score_col = 'homeTeamSeriesScore'
away_team_series_score_col = 'awayTeamSeriesScore'

date_format_yyyy_mm_dd = '%Y-%m-%d'
div_1_val = 'div_1'
div_2_val = 'div_2'
div_3_val = 'div_3'
wc_1_val = 'wc_1'
wc_2_val = 'wc_2'
missed_val = 'Missed'
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
images_folder = 'images/'
season_sched_folder = output_folder + 'season_schedules/'
season_feature_sets_folder = output_folder + 'season_feature_sets/'
season_pred_folder = output_folder + 'season_predictions/{date}/'
season_sched_filename = '{season}_season_sched.csv'
season_pred_filename = 'regularseason_predictions_{date}.csv'
final_standings_filename = 'regularseason_standings_{date}.csv'
playoff_pred_filename = 'playoff_tree_predictions_{date}.csv'
venue_geoloc_filename = 'venue_geolocations.csv'
feature_data_filename = '{season}_feature_data.csv'
sklearn_model_filename = 'skl_rf_model.pkl'
playoff_spot_pred_filename = 'playoff_spot_predictions_{date}_n{n}.csv'
playoff_tree_filename = '{season}_playoff_tree_{date}.png'
stanley_cup_image = 'stanley_cup.png'
model_features_filename = '{model}_model_features.txt'

# color tuple format is (B, G, R)
team_info = {
    'Anaheim Ducks': {'c1': (2, 76, 252), 'logo': images_folder + 'anaheim_ducks_logo.png'},
    'Boston Bruins': {'c1': (20, 181, 252), 'logo': images_folder + 'boston_bruins_logo.png'},
    'Buffalo Sabres': {'c1': (135, 48, 0), 'logo': images_folder + 'buffalo_sabres_logo.png'},
    'Calgary Flames': {'c1': (28, 0, 210), 'logo': images_folder + 'calgary_flames_logo.png'},
    'Carolina Hurricanes': {'c1': (38, 17, 206), 'logo': images_folder + 'carolina_hurricanes_logo.png'},
    'Chicago Blackhawks': {'c1': (44, 10, 207), 'logo': images_folder + 'chicago_blackhawks_logo.png'},
    'Colorado Avalanche': {'c1': (61, 38, 111), 'logo': images_folder + 'colorado_avalanche_logo.png'},
    'Columbus Blue Jackets': {'c1': (84,38,0), 'logo': images_folder + 'columbus_blue_jackets_logo.png'},
    'Dallas Stars': {'c1': (71, 104, 0), 'logo': images_folder + 'dallas_stars_logo.png'},
    'Detroit Red Wings': {'c1': (38,17,206), 'logo': images_folder + 'detroit_red_wings_logo.png'},
    'Edmonton Oilers': {'c1': (66, 30, 4), 'logo': images_folder + 'edmonton_oilers_logo.png'},
    'Florida Panthers': {'c1': (66, 30, 4), 'logo': images_folder + 'florida_panthers_logo.png'},
    'Los Angeles Kings': {'c1': (17, 17, 17), 'logo': images_folder + 'los_angeles_kings_logo.png'},
    'Minnesota Wild': {'c1': (48, 73, 2), 'logo': images_folder + 'minnesota_wild_logo.png'},
    'Montréal Canadiens': {'c1': (45, 30, 175), 'logo': images_folder + 'montreal_canadiens_logo.png'},
    'Nashville Predators': {'c1': (28, 184, 255), 'logo': images_folder + 'nashville_predators_logo.png'},
    'New Jersey Devils': {'c1': (38, 17, 206), 'logo': images_folder + 'new_jersey_devils_logo.png'},
    'New York Islanders': {'c1': (155, 83, 0), 'logo': images_folder + 'new_york_islanders_logo.png'},
    'New York Rangers': {'c1': (168, 56, 0), 'logo': images_folder + 'new_york_rangers_logo.png'},
    'Ottawa Senators': {'c1': (50, 26, 218), 'logo': images_folder + 'ottawa_senators_logo.png'},
    'Philadelphia Flyers': {'c1': (2, 73, 247), 'logo': images_folder + 'philadelphia_flyers_logo.png'},
    'Pittsburgh Penguins': {'c1': (28, 184, 255), 'logo': images_folder + 'pittsburgh_penguins_logo.png'},
    'San Jose Sharks': {'c1': (117, 109, 0), 'logo': images_folder + 'san_jose_sharks_logo.png'},
    'Seattle Kraken': {'c1': (40, 22, 0), 'logo': images_folder + 'seattle_kraken_logo.png'},
    'St. Louis Blues': {'c1': (135, 47, 0), 'logo': images_folder + 'st_louis_blues_logo.png'},
    'Tampa Bay Lightning': {'c1': (104, 40, 0), 'logo': images_folder + 'tampa_bay_lightning_logo.png'},
    'Toronto Maple Leafs': {'c1': (91, 32, 0), 'logo': images_folder + 'toronto_maple_leafs_logo.png'},
    'Utah Mammoth': {'c1': (231, 179, 105), 'logo': images_folder + 'utah_mammoth_logo.png'},
    'Vancouver Canucks': {'c1': (91, 32, 0), 'logo': images_folder + 'vancouver_canucks_logo.png'},
    'Vegas Golden Knights': {'c1': (91, 151, 185), 'logo': images_folder + 'vegas_golden_knights_logo.png'},
    'Washington Capitals': {'c1': (66, 30, 4), 'logo': images_folder + 'washington_capitals_logo.png'},
    'Winnipeg Jets': {'c1': (66, 30, 4), 'logo': images_folder + 'winnipeg_jets_logo.png'}
}

missing_geoloc = {
    'Amalie Arena': (27.9439, -82.4519),
    'FLA Live Arena': (26.1583, -80.3256),
    'Gila River Arena': (33.5325, -112.2611)
}