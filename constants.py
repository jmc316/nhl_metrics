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

last_period_map = {'REG': 0, 'OT': 1, 'SO': 2}
shootout_rate = 20

id_col = 'id'
season_col = 'season'
starttime_utc_col = 'startTimeUTC'
game_year_col = 'gameYear'
game_month_col = 'gameMonth'
game_day_col = 'gameDay'
game_time_col = 'gameTimeUTC'
venue_timezone_col = 'venueTimezone'
venue_col = 'venue'
away_team_id_col = 'awayTeamId'
home_team_id_col = 'homeTeamId'
away_team_name_col = 'awayTeamName'
home_team_name_col = 'homeTeamName'
away_team_score_col = 'awayTeamScore'
home_team_score_col = 'homeTeamScore'
last_period_col = 'lastPeriod'

feature_cols = [id_col, season_col, game_year_col, game_month_col, game_day_col, game_time_col, venue_timezone_col, venue_col, away_team_id_col, home_team_id_col]
predict_cols = [away_team_score_col, home_team_score_col, last_period_col]

output_folder = 'output/'
season_sched_filename = 'season_sched.csv'
season_sched_pred_filename = 'season_sched_pred.csv'
season_sched_pred_points_filename = 'season_sched_pred_points.csv'

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