import pandas as pd
import constants as cons

from time import sleep
from file_utils import csvSave
from predict import create_df_set
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

def geolocate_venues(feature_df, venue_col):

    geolocator = Nominatim(user_agent='nhl_metrics_app', timeout=10) 
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    geoloc_df = pd.DataFrame(columns=[venue_col, venue_col+'_lat', venue_col+'_long'])

    # add a new column with the geolocation of each game venue
    for venue in feature_df[venue_col].unique():
        while True:
            try:
                location = geocode(venue)
                if location:
                    print(f'{venue}: {location.latitude}, {location.longitude}')
                    geoloc_df = pd.concat([geoloc_df,
                                           pd.DataFrame(
                                               {venue_col: [venue],
                                                venue_col+'_lat': [location.latitude],
                                                venue_col+'_long': [location.longitude]
                                                })], ignore_index=True)
                else:
                    print(f'{venue}: Geolocation not found')
                    geoloc_df = pd.concat([geoloc_df,
                                           pd.DataFrame(
                                               {venue_col: [venue],
                                                venue_col+'_lat': [None],
                                                venue_col+'_long': [None]
                                                })], ignore_index=True)
                break
            except Exception as ex:
                print(f'\t\t... {ex} ...')
                sleep(5)

    return geoloc_df


if __name__ == "__main__":
    # create the feature set
    feature_df = create_df_set()

    # geolocate the venues
    geoloc_df = geolocate_venues(feature_df, cons.venue_col)

    # save the geolocation data to a csv file
    csvSave(geoloc_df, cons.util_data_folder, cons.venue_geolocations_filename)