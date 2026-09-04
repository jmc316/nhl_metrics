playoff_sched_format = [0, 2, 5, 7, 10, 12, 15]
final_sched_format = [0, 3, 6, 9, 12, 15, 18]
playoff_round_buffer = 2
api_timeout_wait_time = 3

game_id_col = 'gameId'
season_col = 'season'
season_name_col = 'seasonName'
starttime_utc_col = 'startTimeUTC'
starttime_est_col = 'startTimeEST'
venue_timezone_col = 'venueTimezone'
venue_col = 'venue'
away_team_name_col = 'awayTeamName'
home_team_name_col = 'homeTeamName'
away_team_score_col = 'awayTeamScore'
home_team_score_col = 'homeTeamScore'
last_period_col = 'lastPeriod'
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
total_reg_wins_col = 'totalRegWins'
home_team_reg_ot_wins_col = 'homeTeamRegOTWins'
away_team_reg_ot_wins_col = 'awayTeamRegOTWins'
total_reg_ot_wins_col = 'totalRegOTWins'
home_team_so_wins_col = 'homeTeamSOWins'
away_team_so_wins_col = 'awayTeamSOWins'
home_team_so_losses_col = 'homeTeamSOLosses'
away_team_so_losses_col = 'awayTeamSOLosses'
home_team_goals_for_col = 'homeTeamGoalsFor'
away_team_goals_for_col = 'awayTeamGoalsFor'
home_team_goals_against_col = 'homeTeamGoalsAgainst'
away_team_goals_against_col = 'awayTeamGoalsAgainst'
total_goals_for_col = 'totalGoalsFor'
total_goals_against_col = 'totalGoalsAgainst'
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
total_games_col = 'totalGames'
game_outcome_col = 'gameOutcome'
game_type_col = 'gameType' # 1 = preseason, 2 = regular season, 3 = playoffs, 4 = all-star game
game_date_col = 'gameDate'
home_team_series_score_col = 'homeTeamSeriesScore'
away_team_series_score_col = 'awayTeamSeriesScore'
home_team_win_col = 'homeTeamWin'
home_win_prob_col = 'homeWinProb'
away_win_prob_col = 'awayWinProb'
game_time_secs_est_col = 'gameTimeSecondsEST'
day_of_week_col = 'dayOfWeek'
reg_game_num_perc_col = '{team}RegGameNumPerc'
playoff_game_num_col = '{team}PlayoffGameNum'
days_rest_col = '{pre}DaysRest'
goalie_days_rest_col = '{pre}GoalieDaysRest'
is_outdoor_venue_col = 'isOutdoorVenue'
road_trip_seq_col = 'roadTripSeq'
travel_dist_n_days_col = '{pre}TravelDist{n}Days'
games_played_n_days_col = '{pre}GamesPlayed{n}Days'
crossed_tz_n_days_col = '{pre}CrossedTZ{n}Days'
is_home_opener_col = 'isHomeOpener'
rival_match_col = 'rivalMatch'
market_intensity_col = 'marketIntensity'
is_ret_home_trap_col = 'isRetHomeTrap'
is_venue_alt_shock_col = 'isVenueAltShock'
playoff_series_score_col = 'playoffSeriesScore'
cor_outcome_col = 'correct_outcome'
home_odds_col = 'homeTeamOdds'
away_odds_col = 'awayTeamOdds'
winner_odds_col = 'winner_odds'
winnings_col = 'winnings'
bankroll_col = 'bankroll'
fake_tiebreaker_col = 'fakeTiebreaker'
point_per_n_col = '{pre}PointsPer{n}Games'
goal_diff_n_col = '{pre}GoalDiff{n}Games'
corsi_per_n_col = '{pre}CorsiPer{n}Games'
fenwick_per_n_col = '{pre}FenwickPer{n}Games'
home_shot_og_col = 'home_shot-on-goal'
away_shot_og_col = 'away_shot-on-goal'
home_shot_miss_col = 'home_missed-shot'
away_shot_miss_col = 'away_missed-shot'
home_shot_blk_col = 'home_blocked-shot'
away_shot_blk_col = 'away_blocked-shot'
elo_rat_col = '{pre}EloRating'
home_penalty_col = 'home_penalty'
away_penalty_col = 'away_penalty'
home_pp_goal_col = 'home_pp_goal'
away_pp_goal_col = 'away_pp_goal'
pp_per_n_col = '{pre}PPPer{n}Games'
pk_per_n_col = '{pre}PKPer{n}Games'
save_per_n_col = '{pre}SavePer{n}Games'
gaa_n_col = '{pre}GAA{n}Games'
num_starts_n_col = '{pre}NumStarts{n}Days'
num_season_starts_col = '{pre}NumSeasonStarts'
goalie_id_col = 'goalie_id'
home_goalie_name_col = 'home_goalie_name'
away_goalie_name_col = 'away_goalie_name'
home_goalie_id_col = 'home_goalie_id'
away_goalie_id_col = 'away_goalie_id'
home_starter_col = 'home_starter'
away_starter_col = 'away_starter'
home_toi_secs_col = 'home_toi_secs'
away_toi_secs_col = 'away_toi_secs'
home_ev_shots_against_col = 'home_ev_shots_against'
away_ev_shots_against_col = 'away_ev_shots_against'
home_ev_saves_col = 'home_ev_saves'
away_ev_saves_col = 'away_ev_saves'
home_ev_goals_against_col = 'home_ev_goals_against'
away_ev_goals_against_col = 'away_ev_goals_against'
home_sh_shots_against_col = 'home_sh_shots_against'
away_sh_shots_against_col = 'away_sh_shots_against'
home_sh_saves_col = 'home_sh_saves'
away_sh_saves_col = 'away_sh_saves'
home_sh_goals_against_col = 'home_sh_goals_against'
away_sh_goals_against_col = 'away_sh_goals_against'
home_pp_shots_against_col = 'home_pp_shots_against'
away_pp_shots_against_col = 'away_pp_shots_against'
home_pp_saves_col = 'home_pp_saves'
away_pp_saves_col = 'away_pp_saves'
home_pp_goals_against_col = 'home_pp_goals_against'
away_pp_goals_against_col = 'away_pp_goals_against'
home_tot_shots_against_col = 'home_tot_shots_against'
away_tot_shots_against_col = 'away_tot_shots_against'
home_tot_saves_col = 'home_tot_saves'
away_tot_saves_col = 'away_tot_saves'
home_tot_goals_against_col = 'home_tot_goals_against'
away_tot_goals_against_col = 'away_tot_goals_against'
home_decision_col = 'home_decision'
away_decision_col = 'away_decision'
home_lineup_col = 'homeTeamLineup'
away_lineup_col = 'awayTeamLineup'

pred_suf = '_predicted'
act_suf = '_actual'
lat_suf = '_lat'
long_suf = '_long'

date_format_yyyy_mm_dd = '%Y-%m-%d'
div_1_val = 'div_1'
div_2_val = 'div_2'
div_3_val = 'div_3'
wc_1_val = 'wc_1'
wc_2_val = 'wc_2'
missed_val = 'missed'
make_r2_val = 'make_round_2'
make_r3_val = 'make_round_3'
make_cup_final_val = 'make_cup_final'
win_cup_val = 'win_cup'
season_stdt = '09-20'
season_enddt = '06-30'
cur_season_name = '20262027'
est_tz = 'America/New_York'
sched_feat_windows = [4, 7]
team_feat_windows = [5, 20]
goalie_feat_windows = [3] #, 15, 'Season']
goalie_feat_windows_2 = [7] # [1, 3, 5, 7, 9, 11, 13, 14, 'Season']
goalie_feat_prefixes = [None, 'ev', 'pp', 'sh']

tiebreaker_cols = [total_points_col, points_percentage_col, total_reg_wins_col, total_reg_ot_wins_col,
                   total_wins_col, goal_diff_col, total_goals_for_col]
final_standings_col_order = [conference_name_col, conference_seed_col, division_name_col, division_seed_col,
                             playoff_seed_col, team_name_col, total_games_col, total_wins_col, total_losses_col,
                             total_otls_col, total_points_col, points_percentage_col, total_reg_wins_col,
                             total_reg_ot_wins_col, goal_diff_col, total_goals_for_col]
sched_feature_cols = [game_id_col, season_name_col, game_type_col, starttime_est_col, venue_timezone_col,
                     venue_col, home_team_name_col, away_team_name_col, home_team_score_col,
                     away_team_score_col, last_period_col, home_team_win_col,
                     home_win_prob_col, away_win_prob_col]
team_feature_cols = sched_feature_cols + [home_shot_og_col, away_shot_og_col, home_shot_miss_col,
                                         away_shot_miss_col, home_shot_blk_col, away_shot_blk_col,
                                         home_penalty_col, away_penalty_col, home_pp_goal_col, away_pp_goal_col]
goalie_feature_cols = sched_feature_cols + [home_goalie_name_col, away_goalie_name_col, home_goalie_id_col,
                                            away_goalie_id_col, home_starter_col, away_starter_col,
                                            home_toi_secs_col, away_toi_secs_col, home_ev_shots_against_col,
                                            away_ev_shots_against_col, home_ev_saves_col, away_ev_saves_col,
                                            home_ev_goals_against_col, away_ev_goals_against_col, 
                                            home_sh_shots_against_col, away_sh_shots_against_col,
                                            home_sh_saves_col, away_sh_saves_col, home_sh_goals_against_col,
                                            away_sh_goals_against_col, home_pp_shots_against_col,
                                            away_pp_shots_against_col, home_pp_saves_col, away_pp_saves_col,
                                            home_pp_goals_against_col, away_pp_goals_against_col,
                                            home_tot_shots_against_col, away_tot_shots_against_col,
                                            home_tot_saves_col, away_tot_saves_col, home_tot_goals_against_col,
                                            away_tot_goals_against_col, home_decision_col, away_decision_col]
player_feature_cols = sched_feature_cols + [home_lineup_col, away_lineup_col]

output_folder = 'output/'
util_data_folder = 'util_data/'
model_files_folder = 'model_files/'
images_folder = 'images/'
season_sched_folder = output_folder + 'season_schedules/'
season_feature_sets_folder = output_folder + 'season_feature_sets/'
sched_features_folder = season_feature_sets_folder + 'sched_features/'
sched_features_filename = '{season}_sched_features.csv'
goalie_features_folder = season_feature_sets_folder + 'goalie_features/'
goalie_features_filename = '{season}_goalie_features.csv'
goalie_data_filename = 'goalie_data.csv'
player_data_filename = 'player_data.csv'
player_features_folder = season_feature_sets_folder + 'player_features/'
player_features_filename = '{season}_player_features.csv'
team_features_folder = season_feature_sets_folder + 'team_features/'
team_features_filename = '{season}_team_features.csv'
feat_data_filename = '{season}_feature_data.csv'
season_pred_base_folder = output_folder + 'season_predictions/'
season_pred_folder = season_pred_base_folder + '{date}/'
season_sched_filename = '{season}_season_sched.csv'
season_pred_filename = 'regularseason_predictions_{date}.csv'
final_standings_filename = 'regularseason_standings_{date}.csv'
playoff_pred_filename = 'playoff_tree_predictions_{date}.csv'
venue_geoloc_filename = 'venue_geolocations.csv'
sklearn_model_filename = 'skl_rf_model.pkl'
season_results_prob_filename = 'season_results_probabilities_{date}_n{n}.csv'
playoff_tree_filename = '{season}_playoff_tree_{date}.png'
playoff_probability_filename = '{season}_playoff_probability_{date}_n{n}.png'
stanley_cup_image = 'stanley_cup.png'
model_features_filename = '{model}_model_features.txt'
sched_odds_filename = 'all_time_schedule_odds.csv'
pred_ret_filename = 'prediction_returns_{date_since}_to_{today_dt}.csv'
shap_filename = 'shap_{home}_{away}_{date}.png'

# color tuple format is (B, G, R)
team_info = {
    'Anaheim Ducks': {'c1': (2, 76, 252), 'logo': images_folder + 'anaheim_ducks_logo.png', 'conference': 'Western', 'division': 'Pacific'},
    'Boston Bruins': {'c1': (20, 181, 252), 'logo': images_folder + 'boston_bruins_logo.png', 'conference': 'Eastern', 'division': 'Atlantic'},
    'Buffalo Sabres': {'c1': (135, 48, 0), 'logo': images_folder + 'buffalo_sabres_logo.png', 'conference': 'Eastern', 'division': 'Atlantic'},
    'Calgary Flames': {'c1': (28, 0, 210), 'logo': images_folder + 'calgary_flames_logo.png', 'conference': 'Western', 'division': 'Pacific'},
    'Carolina Hurricanes': {'c1': (38, 17, 206), 'logo': images_folder + 'carolina_hurricanes_logo.png', 'conference': 'Eastern', 'division': 'Metropolitan'},
    'Chicago Blackhawks': {'c1': (44, 10, 207), 'logo': images_folder + 'chicago_blackhawks_logo.png', 'conference': 'Western', 'division': 'Central'},
    'Colorado Avalanche': {'c1': (61, 38, 111), 'logo': images_folder + 'colorado_avalanche_logo.png', 'conference': 'Western', 'division': 'Central'},
    'Columbus Blue Jackets': {'c1': (84,38,0), 'logo': images_folder + 'columbus_blue_jackets_logo.png', 'conference': 'Eastern', 'division': 'Metropolitan'},
    'Dallas Stars': {'c1': (71, 104, 0), 'logo': images_folder + 'dallas_stars_logo.png', 'conference': 'Western', 'division': 'Central'},
    'Detroit Red Wings': {'c1': (38,17,206), 'logo': images_folder + 'detroit_red_wings_logo.png', 'conference': 'Eastern', 'division': 'Atlantic'},
    'Edmonton Oilers': {'c1': (66, 30, 4), 'logo': images_folder + 'edmonton_oilers_logo.png', 'conference': 'Western', 'division': 'Pacific'},
    'Florida Panthers': {'c1': (66, 30, 4), 'logo': images_folder + 'florida_panthers_logo.png', 'conference': 'Eastern', 'division': 'Atlantic'},
    'Los Angeles Kings': {'c1': (17, 17, 17), 'logo': images_folder + 'los_angeles_kings_logo.png', 'conference': 'Western', 'division': 'Pacific'},
    'Minnesota Wild': {'c1': (48, 73, 2), 'logo': images_folder + 'minnesota_wild_logo.png', 'conference': 'Western', 'division': 'Central'},
    'Montréal Canadiens': {'c1': (45, 30, 175), 'logo': images_folder + 'montreal_canadiens_logo.png', 'conference': 'Eastern', 'division': 'Atlantic'},
    'Nashville Predators': {'c1': (28, 184, 255), 'logo': images_folder + 'nashville_predators_logo.png', 'conference': 'Western', 'division': 'Central'},
    'New Jersey Devils': {'c1': (38, 17, 206), 'logo': images_folder + 'new_jersey_devils_logo.png', 'conference': 'Eastern', 'division': 'Metropolitan'},
    'New York Islanders': {'c1': (155, 83, 0), 'logo': images_folder + 'new_york_islanders_logo.png', 'conference': 'Eastern', 'division': 'Metropolitan'},
    'New York Rangers': {'c1': (168, 56, 0), 'logo': images_folder + 'new_york_rangers_logo.png', 'conference': 'Eastern', 'division': 'Metropolitan'},
    'Ottawa Senators': {'c1': (50, 26, 218), 'logo': images_folder + 'ottawa_senators_logo.png', 'conference': 'Eastern', 'division': 'Atlantic'},
    'Philadelphia Flyers': {'c1': (2, 73, 247), 'logo': images_folder + 'philadelphia_flyers_logo.png', 'conference': 'Eastern', 'division': 'Metropolitan'},
    'Pittsburgh Penguins': {'c1': (28, 184, 255), 'logo': images_folder + 'pittsburgh_penguins_logo.png', 'conference': 'Eastern', 'division': 'Metropolitan'},
    'San Jose Sharks': {'c1': (117, 109, 0), 'logo': images_folder + 'san_jose_sharks_logo.png', 'conference': 'Western', 'division': 'Pacific'},
    'Seattle Kraken': {'c1': (40, 22, 0), 'logo': images_folder + 'seattle_kraken_logo.png', 'conference': 'Western', 'division': 'Pacific'},
    'St. Louis Blues': {'c1': (135, 47, 0), 'logo': images_folder + 'st_louis_blues_logo.png', 'conference': 'Western', 'division': 'Central'},
    'Tampa Bay Lightning': {'c1': (104, 40, 0), 'logo': images_folder + 'tampa_bay_lightning_logo.png', 'conference': 'Eastern', 'division': 'Atlantic'},
    'Toronto Maple Leafs': {'c1': (91, 32, 0), 'logo': images_folder + 'toronto_maple_leafs_logo.png', 'conference': 'Eastern', 'division': 'Atlantic'},
    'Utah Mammoth': {'c1': (231, 179, 105), 'logo': images_folder + 'utah_mammoth_logo.png', 'conference': 'Western', 'division': 'Central'},
    'Vancouver Canucks': {'c1': (91, 32, 0), 'logo': images_folder + 'vancouver_canucks_logo.png', 'conference': 'Western', 'division': 'Pacific'},
    'Vegas Golden Knights': {'c1': (91, 151, 185), 'logo': images_folder + 'vegas_golden_knights_logo.png', 'conference': 'Western', 'division': 'Pacific'},
    'Washington Capitals': {'c1': (66, 30, 4), 'logo': images_folder + 'washington_capitals_logo.png', 'conference': 'Eastern', 'division': 'Metropolitan'},
    'Winnipeg Jets': {'c1': (66, 30, 4), 'logo': images_folder + 'winnipeg_jets_logo.png', 'conference': 'Western', 'division': 'Central'}
}

defunct_team_info = {
    'Arizona Coyotes': {'c1': None, 'logo': None, 'conference': 'Western', 'division': 'Central'},
    'Utah Utah Hockey Club': {'c1': None, 'logo': None, 'conference': 'Western', 'division': 'Central'},
}

team_name_addrev_map = {
    'Anaheim Ducks': 'ANA',
    'Boston Bruins': 'BOS',
    'Buffalo Sabres': 'BUF',
    'Calgary Flames': 'CGY',
    'Carolina Hurricanes': 'CAR',
    'Chicago Blackhawks': 'CHI',
    'Colorado Avalanche': 'COL',
    'Columbus Blue Jackets': 'CBJ',
    'Dallas Stars': 'DAL',
    'Detroit Red Wings': 'DET',
    'Edmonton Oilers': 'EDM',
    'Florida Panthers': 'FLA',
    'Los Angeles Kings': 'LAK',
    'Minnesota Wild': 'MIN',
    'Montréal Canadiens': 'MTL',
    'Nashville Predators': 'NSH',
    'New Jersey Devils': 'NJD',
    'New York Islanders': 'NYI',
    'New York Rangers': 'NYR',
    'Ottawa Senators': 'OTT',
    'Philadelphia Flyers': 'PHI',
    'Pittsburgh Penguins': 'PIT',
    'San Jose Sharks': 'SJS',
    'Seattle Kraken': 'SEA',
    'St. Louis Blues': 'STL',
    'Tampa Bay Lightning': 'TBL',
    'Toronto Maple Leafs': 'TOR',
    'Utah Mammoth': 'UTA',
    'Vancouver Canucks': 'VAN',
    'Vegas Golden Knights': 'VGK',
    'Washington Capitals': 'WSH',
    'Winnipeg Jets': 'WPG'
}

missing_geoloc = {
    'Amalie Arena': (27.9439, -82.4519),
    'FLA Live Arena': (26.1583, -80.3256),
    'Gila River Arena': (33.5325, -112.2611)
}

outdoor_venues = [
    'Target Field', 'Nissan Stadium', 'Tim Hortons Field', 'Fenway Park', 'Carter-Finley Stadium',
    'Commonwealth Stadium', 'T-Mobile Park', 'MetLife Stadium', 'Wrigley Field', 'Ohio Stadium',
    'Raymond James Stadium', 'Princess Auto Stadium', 'Rice-Eccles Stadium'
]

# tiered by Google Gemini
market_intensity_map = {
    # Tier 4: Low Capacity / Unique Arena Traps
    'Utah Mammoth': 1, 'Anaheim Ducks': 1, 'San Jose Sharks': 1, 'Arizona Coyotes': 1,
    'Utah Utah Hockey Club': 1,
    
    # Tier 3: Performance Dependent / Destination Markets
    'Florida Panthers': 2, 'Tampa Bay Lightning': 2, 'Los Angeles Kings': 2,
    'Carolina Hurricanes': 2, 'Nashville Predators': 2, 'Dallas Stars': 2,
    'St. Louis Blues': 2, 'Columbus Blue Jackets': 2, 'Washington Capitals': 2,
    'New Jersey Devils': 2, 'New York Islanders': 2, 'Seattle Kraken': 2,
    
    # Tier 2: Large Stable Markets / Dedicated Hockey Cities
    'New York Rangers': 3, 'Philadelphia Flyers': 3, 'Minnesota Wild': 3,
    'Vegas Golden Knights': 3, 'Detroit Red Wings': 3, 'Pittsburgh Penguins': 3,
    'Chicago Blackhawks': 3, 'Colorado Avalanche': 3, 'Calgary Flames': 3,
    'Vancouver Canucks': 3, 'Winnipeg Jets': 3, 'Buffalo Sabres': 3, 'Ottawa Senators': 3,
    
    # Tier 1: Canadian Hotbeds & Original Six Elites
    'Montréal Canadiens': 4, 'Toronto Maple Leafs': 4, 'Edmonton Oilers': 4, 'Boston Bruins': 4
}

high_altitude_venues = ['Ball Arena', 'Delta Center', 'Rice-Eccles Stadium']

venue_timezone_map = {
    # Pacific Timezones
    'US/Pacific': 'Pacific', 'America/Los_Angeles': 'Pacific', 'America/Vancouver': 'Pacific',

    # Mountain Timezones
    'US/Mountain': 'Mountain', 'America/Denver': 'Mountain', 'America/Edmonton': 'Mountain', 'America/Phoenix': 'Mountain',

    # Central American Timezones
    'US/Central': 'Am_Central', 'America/Chicago': 'Am_Central', 'America/Winnipeg': 'Am_Central',

    # Eastern American Timezones
    'US/Eastern': 'Am_Eastern', 'America/New_York': 'Am_Eastern', 'America/Toronto': 'Am_Eastern', 'America/Montreal': 'Am_Eastern', 'America/Detroit': 'Am_Eastern',

    # Central European Timezones
    'Europe/Berlin': 'Eu_Central', 'Europe/Stockholm': 'Eu_Central', 'Europe/Prague': 'Eu_Central',

    # Eastern European Timezones
    'Europe/Helsinki': 'Eu_Eastern' 
}