import constants as cons
import pandas as pd


def team_info():
    
    while True:
        try:
            # Fetch the teams
            teams_df = pd.DataFrame(cons.nhl_client.teams.teams())
            break
        except Exception as ex:
            print(f'\t\t... {ex} ...')
            continue
        
    teams_df.rename(columns={'name': 'teamName'}, inplace=True)

    teams_df['conferenceName'] = teams_df['conference'].apply(lambda x: x['name'])
    teams_df['divisionName'] = teams_df['division'].apply(lambda x: x['name'])

    teams_df = teams_df[['teamName', 'conferenceName', 'divisionName']]

    return teams_df