import playoffs

import pandas as pd
import constants as cons
import nhl_utils as nhlu
import terminal_ui as tui

from file_utils import csvLoad
from datetime import datetime as dt
from analyze import prediction_analysis
from predict import predict_season, playoff_spot_predictions, game_result_comparison


def get_asofdate():

    if dt.now().month < int(cons.season_stdt[:2]):
        cur_season_stdt = pd.to_datetime(f'{dt.now().year - 1}-{cons.season_stdt}').date()
    else:
        cur_season_stdt = pd.to_datetime(f'{dt.now().year}-{cons.season_stdt}').date()

    max_dt = dt.now().date()

    # get user input for date
    terminal_ui = tui.terminal_input_dt([cur_season_stdt, max_dt])
    terminal_ui.display_options()
    terminal_ui.receive_user_input()
    today_dt = terminal_ui.get_response()

    return today_dt


def ui_todate_predict():

    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)

    print('Updating predictions for current season...\n')
    feature_df = predict_season(to_csv=True, set_model_random_state=True, today_dt=today_dt)
    game_result_comparison(feature_df)
    season_results_df = nhlu.generate_final_standings(nhlu.assign_game_points(feature_df), today_dt, to_csv=True)
    nhlu.nhl_team_standings(season_results_df)
    playoffs.playoff_tree_predictions(feature_df, season_results_df, True, today_dt, display_image=False)


def ui_historic_predict():

    today_dt = get_asofdate()

    print(f'Updating predictions for current season as of {today_dt}...\n')
    feature_df = predict_season(to_csv=True, set_model_random_state=True, today_dt=today_dt)
    game_result_comparison(feature_df)
    season_results_df = nhlu.generate_final_standings(nhlu.assign_game_points(feature_df), today_dt, to_csv=True)
    nhlu.nhl_team_standings(season_results_df)
    playoffs.playoff_tree_predictions(feature_df, season_results_df, True, today_dt, display_image=False)


def ui_todate_playoff_spot_predict():

    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)

    # get user input for n
    terminal_ui = tui.terminal_input_int(range(1, cons.max_simulations + 1))
    terminal_ui.display_options()
    terminal_ui.receive_user_input()
    n_in = terminal_ui.get_response()

    playoff_spot_predictions(today_dt, n=n_in)


def ui_historic_playoff_spot_predict():

    today_dt = get_asofdate()

    # get user input for n
    terminal_ui = tui.terminal_input_int(range(1, cons.max_simulations + 1))
    terminal_ui.display_options()
    terminal_ui.receive_user_input()
    n_in = terminal_ui.get_response()

    playoff_spot_predictions(today_dt, n=n_in)


def ui_update_predictions():

    predict_ui = tui.predict_screen()

    func_map = {option: getattr(__import__(module), func) for option, (module, func) in cons.update_predictions_options.items()}

    # call the function associated with the user's choice
    func_map[predict_ui.get_response()]()


def ui_update_playoff_spot_probabilities():

    playoff_spot_ui = tui.playoff_spot_screen()

    func_map = {option: getattr(__import__(module), func) for option, (module, func) in cons.playoff_spot_prob_options.items()}

    # call the function associated with the user's choice
    func_map[playoff_spot_ui.get_response()]()


def ui_model_accuracy():

    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)

    season_prediction_df = csvLoad(cons.season_pred_folder.format(date=today_dt), cons.season_pred_filename.format(date=today_dt))

    prediction_analysis(season_prediction_df, '2026-02-24') # last day before Olympic Break ended