import os
import pandas as pd
from nhlpy import NHLClient

OUTPUT_FOLDER = 'output/'

def main():
    # Create an instance of the NHLClient
    nhl_client = NHLClient()

    # Fetch the list of teams
    teams = nhl_client.teams.teams()
    
    # Convert the list of teams to a DataFrame
    teams_df = pd.DataFrame(teams)
    
    # Save the DataFrame to a CSV file
    teams_df.to_csv(os.path.join(OUTPUT_FOLDER, 'nhl_teams.csv'), index=False)
    
    print("NHL teams data has been saved to 'nhl_teams.csv'.")


if __name__ == "__main__":
    main()