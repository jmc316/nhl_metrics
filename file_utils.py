import os
import time

import numpy as np
import pandas as pd
import constants as cons


# standardized universal csv loading function
def csvLoad(folder, filename):

    filepath = os.path.join(folder, filename)
    dfdata = pd.read_csv(filepath)

    if cons.season_name_col in dfdata.columns:
        dfdata[cons.season_name_col] = dfdata[cons.season_name_col].astype(str)

    if cons.game_date_col in dfdata.columns:
        dfdata[cons.game_date_col] = pd.to_datetime(dfdata[cons.game_date_col]).dt.date

    for col in dfdata.columns:
        if isinstance(dfdata[col], np.int64):
            dfdata[col] = dfdata[col].astype(int)

    return dfdata


# standardized universal csv saving function
def csvSave(dfdata, folder, filename):

    try:
        if not os.path.exists(folder):
            os.makedirs(folder)

        filepath = os.path.join(folder, filename)
        dfdata.to_csv(filepath, index=False)

    except Exception as ex:
        print(f'Error saving CSV file: {ex}')


# standardized universal pickle loading function
def pklLoad(folder, filename):

    filepath = os.path.join(folder, filename)
    try:
        data = pd.read_pickle(filepath)
    except PermissionError as ex:
        print(f'Permission error loading pickle file: {ex}')
        time.sleep(5)
        return pklLoad(folder, filename)

    return data


# standardized universal pickle saving function
def pklSave(data, folder, filename):

    try:
        if not os.path.exists(folder):
            os.makedirs(folder)

        filepath = os.path.join(folder, filename)
        pd.to_pickle(data, filepath)

    except Exception as ex:
        print(f'Error saving pickle file: {ex}')


# standardized universal txt saving function
def txtSave(data, folder, filename):

    try:
        if not os.path.exists(folder):
            os.makedirs(folder)

        filepath = os.path.join(folder, filename)
        with open(filepath, 'w') as file:
            for line in data:
                file.write(f"{line}\n")

    except Exception as ex:
        print(f'Error saving txt file: {ex}')

    return data