import os

import pandas as pd
import numpy as np
import constants as cons


def fileLoad(folder, filename):

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


def fileSave(dfdata, folder, filename):

    if not os.path.exists(folder):
        os.makedirs(folder)

    filepath = os.path.join(folder, filename)
    dfdata.to_csv(filepath, index=False)