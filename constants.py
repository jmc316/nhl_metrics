from nhlpy import NHLClient

# Create an instance of the NHLClient
nhl_client = NHLClient()

# format is 'Option Name': ['module_name', 'function_name']
main_options = {
    'NHL Team Stats': ['team_stats', 'nhl_team_stats'],
}

exit_option = {'Exit': ['terminal_ui', 'exit_program']}

team_stats_options = {
    'Standings': ['team_stats', 'nhl_team_standings'],
    'Individual Team Stats': ['team_stats', 'nhl_individual_team_stats'],
}