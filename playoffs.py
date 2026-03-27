import numpy as np
import pandas as pd
import features as ft
import constants as cons
import skl_utils as sklu

from datetime import datetime as dt
from file_utils import csvSave
from playoff_matchup import PlayoffMatchup
from playoff_tree import display_playoff_tree


def playoff_tree_predictions(regular_season_df, season_results_df, set_model_random_state, today_dt, to_csv=True):

    print('Predicting playoff tree...')

    # if the points columns are in the season results dataframe, remove them to avoid confusing the model
    if cons.home_team_points_col in regular_season_df.columns:
        regular_season_df.drop(columns=[cons.home_team_points_col], inplace=True)
    if cons.away_team_points_col in regular_season_df.columns:
        regular_season_df.drop(columns=[cons.away_team_points_col], inplace=True)
    
    # create a dataframe with the already scheduled playoff games
    scheduled_games_df = regular_season_df.loc[(regular_season_df[cons.game_type_col] == 3) &
                                        (regular_season_df[cons.season_name_col] == max(regular_season_df[cons.season_name_col]))].copy()

    # if there are no scheduled playoff games, will need to create schedules for all rounds before predictions
    if scheduled_games_df.empty:
        print('\tNo scheduled playoff games found for this season...')
        rounds_scheduled = 0
        playoff_df = pd.DataFrame()
    else:
        # can't test this yet
        # need to assume that a whole round has finished and is in the same format as the generated playoff schedules
        exit('Error: Found scheduled playoff games in the regular season schedule, but this functionality has not been implemented yet.\n' \
        'Please ensure that the regular season schedule does not contain any playoff games or that the playoff games are in the same format as the generated playoff schedules.')
    
    # load the venue map to establish each team's home venue and timezone
    venue_map_df = venue_map_load(regular_season_df)

    # initialize lists to store OOB predictions, MSE, and R-squared values for each playoff round
    oob_list, mse_list, rsq_list = [], [], []

    # initialize dictionary that hols all playoff matchups, which will be updated after each round of the playoffs
    all_matchups = {}

    # loop through every playoff round
    for pl_round in range(rounds_scheduled+1, 5):
        print(f'\nPlayoffs Round {pl_round}')

        # if there is no schedule for this round, create the schedule
        if rounds_scheduled+1 <= pl_round:

            # playoff round 1
            if pl_round == 1:
                # generate the round 1 playoff matchups based off the regular season standings
                east_playoff_matchups = generate_playoff_matchups(season_results_df.loc[season_results_df[cons.conference_name_col] == 'Eastern'], 1)
                west_playoff_matchups = generate_playoff_matchups(season_results_df.loc[season_results_df[cons.conference_name_col] == 'Western'], 1)

                round_matchups = east_playoff_matchups.copy()
                for matchup_num, matchup in west_playoff_matchups.items():
                    round_matchups.update({matchup_num+4: matchup})

                # create the round 1 playoff schedule and add it to the regular season schedule
                playoff_df = pd.concat([regular_season_df, create_playoff_round_schedule(round_matchups, venue_map_df, regular_season_df, playoff_df)], ignore_index=True, sort=False)
            # playoff rounds 2, 3, 4
            else:
                # generate the round n playoff matchups based off the regular season standings
                round_matchups = generate_playoff_matchups(playoff_df, pl_round, round_matchups)

                # create the round n playoff schedule and add it to the regular season schedule
                playoff_df = create_playoff_round_schedule(round_matchups, venue_map_df, regular_season_df, playoff_df)

            # predict games for this playoff round one day at a time
            for game_dt in playoff_df.loc[(playoff_df[cons.season_name_col] == max(playoff_df[cons.season_name_col])) &
                                          (playoff_df[cons.game_type_col] == 3) &
                                          (playoff_df[cons.last_period_col].isna()), cons.game_date_col].unique():
                
                # if there were scheduled games on this date that no longer exist, skip to the next date
                if playoff_df.loc[playoff_df[cons.game_date_col] == game_dt].empty:
                    continue

                playoff_df_filt = playoff_df.loc[playoff_df[cons.game_date_col] <= game_dt]

                # add dependent features to the playoff schedule dataframe
                playoff_df_filt = ft.dependent_feature_add(playoff_df_filt, backfill=False, debug=False)

                # predict games on selected date
                print(f'\tPredicting games for {game_dt.strftime("%Y-%m-%d")}...')
                playoff_df_filt = sklu.make_predictions(playoff_df_filt, oob_list, mse_list, rsq_list, set_model_random_state, today_dt, load_model=True, save_model=False)

                playoff_df = pd.concat([playoff_df_filt, playoff_df.loc[playoff_df[cons.game_date_col] > game_dt]], ignore_index=True)

                # check to see if any of the series are over based on the current series scores
                playoff_df, round_matchups = series_final_check(playoff_df, playoff_df_filt, round_matchups, game_dt)

            all_matchups.update({pl_round: round_matchups})

    # save playoff predictions to CSV
    if to_csv:
        print('\nSaving playoff predictions to CSV...')
        csvSave(playoff_df, cons.season_pred_folder.format(date=today_dt), cons.playoff_pred_filename.format(date=today_dt))

    display_tree = True
    if display_tree:
        display_playoff_tree(all_matchups, playoff_df[cons.season_name_col].max(), today_dt)

    return playoff_df


def generate_playoff_matchups(data_df, round_num, prev_round_matchups=None):

    matchups_dict = {}

    # create matchups for the first round of playoffs
    if round_num == 1:
        div_1_name = data_df.loc[data_df[cons.playoff_seed_col] == cons.div_1_val, [cons.division_name_col]+cons.tiebreaker_cols].sort_values(
            by=cons.tiebreaker_cols, ascending=False)[cons.division_name_col].values[0]
        
        # matchup 1: division winner with better record vs wildcard 2 team
        matchups_dict.update({0: PlayoffMatchup(
            data_df.loc[data_df[cons.playoff_seed_col] == cons.div_1_val, [cons.team_name_col]+cons.tiebreaker_cols].sort_values(
                by=cons.tiebreaker_cols, ascending=False)[cons.team_name_col].values[0], # highest seed in the conference
            data_df.loc[data_df[cons.playoff_seed_col] == cons.wc_2_val, cons.team_name_col].values[0], # second wildcard team in the conference
            data_df.loc[data_df[cons.playoff_seed_col] == cons.div_1_val, [cons.conference_seed_col]+cons.tiebreaker_cols].sort_values(
                by=cons.tiebreaker_cols, ascending=False)[cons.conference_seed_col].values[0], # division winner conference seed
            data_df.loc[data_df[cons.playoff_seed_col] == cons.wc_2_val, cons.conference_seed_col].values[0], # second wildcard team conference seed
            1, # playoff round number
            data_df.loc[data_df[cons.playoff_seed_col] == cons.div_1_val, [cons.division_name_col]+cons.tiebreaker_cols].sort_values(
                by=cons.tiebreaker_cols, ascending=False)[cons.division_name_col].values[0][:1] +\
            data_df.loc[data_df[cons.playoff_seed_col] == cons.div_1_val, [cons.playoff_seed_col]+cons.tiebreaker_cols].sort_values(
                by=cons.tiebreaker_cols, ascending=False)[cons.playoff_seed_col].values[0][-1:], # division winner playoff seed
            'WC' + data_df.loc[data_df[cons.playoff_seed_col] == cons.wc_2_val, cons.playoff_seed_col].values[0][-1:], # second wildcard team playoff seed
            data_df[cons.conference_name_col].values[0], # conference name
            data_df.loc[data_df[cons.playoff_seed_col] == cons.div_1_val, [cons.division_name_col]+cons.tiebreaker_cols].sort_values(
                by=cons.tiebreaker_cols, ascending=False)[cons.division_name_col].values[0] # division name (based off division winner)
        )})

        # matchup 2: inter-division matchup between division 2 & 3 seeds
        matchups_dict.update({1: PlayoffMatchup(
            data_df.loc[(data_df[cons.playoff_seed_col] == cons.div_2_val) &
                        (data_df[cons.division_name_col] == div_1_name), cons.team_name_col].values[0], # division 2 seed from division 1
            data_df.loc[(data_df[cons.playoff_seed_col] == cons.div_3_val) &
                        (data_df[cons.division_name_col] == div_1_name), cons.team_name_col].values[0], # division 3 seed from division 1
            data_df.loc[(data_df[cons.playoff_seed_col] == cons.div_2_val) &
                        (data_df[cons.division_name_col] == div_1_name), cons.conference_seed_col].values[0], # division 2 seed conference seed
            data_df.loc[(data_df[cons.playoff_seed_col] == cons.div_3_val) &
                        (data_df[cons.division_name_col] == div_1_name), cons.conference_seed_col].values[0], # division 3 seed conference seed
            1, # playoff round number
            div_1_name[:1] + '2', # division 2 seed playoff seed
            div_1_name[:1] + '3', # division 3 seed playoff seed
            data_df[cons.conference_name_col].values[0], # conference name
            div_1_name # division name
        )})

        div_2_name = data_df.loc[data_df[cons.playoff_seed_col] == cons.div_1_val, [cons.division_name_col]+cons.tiebreaker_cols].sort_values(
            by=cons.tiebreaker_cols, ascending=True)[cons.division_name_col].values[0]

        # matchup 3: division winner with worse record vs wildcard 1 team
        matchups_dict.update({2: PlayoffMatchup(
            data_df.loc[data_df[cons.playoff_seed_col] == cons.div_1_val, [cons.team_name_col]+cons.tiebreaker_cols].sort_values(
                by=cons.tiebreaker_cols, ascending=True)[cons.team_name_col].values[0], # division winner with worse record in the conference
            data_df.loc[data_df[cons.playoff_seed_col] == cons.wc_1_val, cons.team_name_col].values[0], # first wildcard team in the conference
            data_df.loc[data_df[cons.playoff_seed_col] == cons.div_1_val, [cons.conference_seed_col]+cons.tiebreaker_cols].sort_values(
                by=cons.tiebreaker_cols, ascending=True)[cons.conference_seed_col].values[0], # division winner conference seed
            data_df.loc[data_df[cons.playoff_seed_col] == cons.wc_1_val, cons.conference_seed_col].values[0], # first wildcard team conference seed
            1, # playoff round number
            data_df.loc[data_df[cons.playoff_seed_col] == cons.div_1_val, [cons.division_name_col]+cons.tiebreaker_cols].sort_values(
                by=cons.tiebreaker_cols, ascending=True)[cons.division_name_col].values[0][:1] +\
            data_df.loc[data_df[cons.playoff_seed_col] == cons.div_1_val, [cons.playoff_seed_col]+cons.tiebreaker_cols].sort_values(
                by=cons.tiebreaker_cols, ascending=True)[cons.playoff_seed_col].values[0][-1:], # division winner playoff seed
            'WC' + data_df.loc[data_df[cons.playoff_seed_col] == cons.wc_1_val, cons.playoff_seed_col].values[0][-1:], # first wildcard team playoff seed
            data_df[cons.conference_name_col].values[0], # conference name
            data_df.loc[data_df[cons.playoff_seed_col] == cons.div_1_val, [cons.division_name_col]+cons.tiebreaker_cols].sort_values(
                by=cons.tiebreaker_cols, ascending=True)[cons.division_name_col].values[0] # division name (based off division winner)
        )})
        
        # matchup 4: inter-division matchup between division 2 & 3 seeds
        matchups_dict.update({3: PlayoffMatchup(
            data_df.loc[(data_df[cons.playoff_seed_col] == cons.div_2_val) &
                        (data_df[cons.division_name_col] == div_2_name), cons.team_name_col].values[0], # division 2 seed from division 2
            data_df.loc[(data_df[cons.playoff_seed_col] == cons.div_3_val) &
                        (data_df[cons.division_name_col] == div_2_name), cons.team_name_col].values[0], # division 3 seed from division 2
            data_df.loc[(data_df[cons.playoff_seed_col] == cons.div_2_val) &
                        (data_df[cons.division_name_col] == div_2_name), cons.conference_seed_col].values[0], # division 2 seed conference seed
            data_df.loc[(data_df[cons.playoff_seed_col] == cons.div_3_val) &
                        (data_df[cons.division_name_col] == div_2_name), cons.conference_seed_col].values[0], # division 3 seed conference seed
            1, # playoff round number
            div_2_name[:1] + '2', # division 2 seed playoff seed
            div_2_name[:1] + '3', # division 3 seed playoff seed
            data_df[cons.conference_name_col].values[0], # conference name
            div_2_name # division name
        )})

    # create matchups for the second, third, and fourth rounds of the playoffs based off the winners from the previous round
    else:
        matchups_dict = playoff_matchups_234(round_num, prev_round_matchups)

    return matchups_dict


def playoff_matchups_234(round_id, prev_round_matchups):

    # find all series winners from the previous round
    series_winners_list = [[
        matchup.get_series_winner(),
        matchup.get_winner_conf_seed(),
        matchup.get_division(),
        matchup.get_conference(),
        'NHL']
        for _, matchup in prev_round_matchups.items()
        ]
    
    series_matchups_preview = {}
    matchups_dict = {}

    for prev_series_winner in series_winners_list:
        if prev_series_winner[round_id] not in series_matchups_preview.keys():
            series_matchups_preview.update({prev_series_winner[round_id]: [[prev_series_winner[0], prev_series_winner[1], prev_series_winner[2], prev_series_winner[3]]]})
        else:
            series_matchups_preview[prev_series_winner[round_id]].append([prev_series_winner[0], prev_series_winner[1], prev_series_winner[2], prev_series_winner[3]])

    # add the winners from the first round to the second round matchups dictionary and reconfigure the result list
    for ind, (_, matchup) in enumerate(series_matchups_preview.items()):
        matchups_dict.update({ind: PlayoffMatchup(
                matchup[0][0], # team 1 name
                matchup[1][0], # team 2 name
                matchup[0][1], # team 1 conference seed
                matchup[1][1], # team 2 conference seed
                round_id, # playoff round number
                conference=matchup[0][3], # conference name
                division=matchup[0][2] # division name
            )})
    
    return matchups_dict


def create_playoff_round_schedule(all_matchups, venue_map_df, feature_df, playoff_df):

    # if the playoff dataframe is empty, take the round start date from the regular season
    if playoff_df.empty:
        round_stdt = feature_df[cons.game_date_col].max() + pd.Timedelta(days=cons.playoff_round_buffer)
    else:
        round_stdt = playoff_df[cons.game_date_col].max() + pd.Timedelta(days=cons.playoff_round_buffer)

    # loop through matchups
    for _, matchup in all_matchups.items():

        # if the matchup is an western matchup, start series on matchday 2
        if matchup.get_conference() == 'Western':
            game_dt = round_stdt + pd.Timedelta(days=1)
            sched_format = cons.playoff_sched_format
        # if the matchup is an eastern matchup, start series on matchday 1
        elif matchup.get_conference() == 'Eastern':
            game_dt = round_stdt
            sched_format = cons.playoff_sched_format
        # if the matchup is the final, start series on matchday 1
        else:
            game_dt = round_stdt
            sched_format = cons.final_sched_format
        
        # list of game dates for the series
        game_dts = [game_dt + pd.Timedelta(days=val) for val in sched_format]

        # list of home and away teams for the series (higher seed is home first)
        if matchup.get_team1_conf_seed() < matchup.get_team2_conf_seed():
            home_teams = [matchup.get_team1()] * 2 + [matchup.get_team2()] * 2 + [matchup.get_team1()] + [matchup.get_team2()] + [matchup.get_team1()]
            away_teams = [matchup.get_team2()] * 2 + [matchup.get_team1()] * 2 + [matchup.get_team2()] + [matchup.get_team1()] + [matchup.get_team2()]
        else:
            home_teams = [matchup.get_team1()] * 2 + [matchup.get_team2()] * 2 + [matchup.get_team1()] + [matchup.get_team2()] + [matchup.get_team1()]
            away_teams = [matchup.get_team2()] * 2 + [matchup.get_team1()] * 2 + [matchup.get_team2()] + [matchup.get_team1()] + [matchup.get_team2()]

        # list of venues for the sesries based off the home team for each game
        venues = [list(venue_map_df.loc[venue_map_df[cons.home_team_name_col]==home_team][[cons.venue_col, cons.venue_timezone_col]].values[0]) for home_team in home_teams]

        # game ID does not exist for unscheduled playoff games
        game_id = np.nan

        # game type for playoff games is 3
        game_type = 3

        # season name is the current season for the feature dataframe
        season_name = max(feature_df[cons.season_name_col])

        # get the most popular game time per venue for the current season to use as the game time for the playoff matchups
        game_time_df = pd.DataFrame(feature_df.loc[
            feature_df[cons.season_name_col]==max(feature_df[cons.season_name_col])][[
                cons.game_time_col, cons.venue_col]].value_counts(), columns=['count'])
        game_time_df = game_time_df.loc[game_time_df['count'] > 5]
        game_time_df = game_time_df.loc[game_time_df.groupby(cons.venue_col)['count'].idxmax()]
        game_time_df.reset_index(inplace=True)
        game_time_utc = [int(game_time_df.loc[game_time_df[cons.venue_col]==venue[0]][cons.game_time_col].values[0]) for venue in venues]

        # add all data to a dataframe for the current matchup and append to the playoff dataframe
        matchup_df = pd.DataFrame({
            cons.game_id_col: [game_id] * len(game_dts),
            cons.season_name_col: [season_name] * len(game_dts),
            cons.game_type_col: [game_type] * len(game_dts),
            cons.venue_timezone_col: [venue[1] for venue in venues],
            cons.venue_col: [venue[0] for venue in venues],
            cons.home_team_name_col: home_teams,
            cons.away_team_name_col: away_teams,
            cons.away_team_score_col: [np.nan] * len(game_dts),
            cons.home_team_score_col: [np.nan] * len(game_dts),
            cons.last_period_col: [np.nan] * len(game_dts),
            cons.game_date_col: game_dts,
            cons.game_time_col: game_time_utc
        })

        # add the matchup games to the playoff dataframe
        playoff_df = pd.concat([playoff_df, matchup_df], ignore_index=True).sort_values(by=[cons.game_date_col, cons.game_time_col])

    return playoff_df


def venue_map_load(regular_season_df):

    venue_map_df = pd.DataFrame(regular_season_df.loc[
        regular_season_df[cons.season_name_col]==max(regular_season_df[cons.season_name_col])][[
            cons.home_team_name_col, cons.venue_col, cons.venue_timezone_col]].value_counts(), columns=['count'])
    venue_map_df = venue_map_df.loc[venue_map_df['count'] > 20]
    venue_map_df.drop(columns=['count'], inplace=True)
    venue_map_df.reset_index(inplace=True)

    return venue_map_df


def series_final_check(playoff_df, playoff_df_filt, all_matchups, game_dt):

    # check if there were any games played where a series could have been won
    series_win_check_df = playoff_df_filt.loc[(playoff_df_filt[cons.game_date_col]==max(playoff_df_filt[cons.game_date_col])) &
                                                  ((playoff_df_filt[cons.home_team_series_score_col] == 3) |
                                                   (playoff_df_filt[cons.away_team_series_score_col] == 3))]
    
    matchup_map = [[matchup.get_teams(), matchup_num] for matchup_num, matchup in all_matchups.items()]
    
    # loop through all games from previous game date
    for _, row in series_win_check_df.iterrows():

        # find the index for this matchup in the matchups list
        for matchup in matchup_map:
            if row[cons.home_team_name_col] in matchup[0]:
                matchup_ind = matchup[1]
                break

        # initialize series win flags for both teams, game 7 indicator
        home_team_wins, away_team_wins = False, False
        game_seven = bool((row[cons.home_team_series_score_col] == 3) and (row[cons.away_team_series_score_col] == 3))

        # if the team that was leading in the series won, the series is over
        if (row[cons.home_team_series_score_col] == 3 and row[cons.home_team_score_col] > row[cons.away_team_score_col]):
            home_team_wins = True
            all_matchups[matchup_ind].set_series_results(row[cons.home_team_name_col], row[cons.away_team_name_col], row[cons.away_team_series_score_col])
            
        elif (row[cons.away_team_series_score_col] == 3 and row[cons.away_team_score_col] > row[cons.home_team_score_col]):
            away_team_wins = True
            all_matchups[matchup_ind].set_series_results(row[cons.away_team_name_col], row[cons.home_team_name_col], row[cons.home_team_series_score_col])

        # if either team won and it was not game 7, remove all future scheduled series games
        if (home_team_wins or away_team_wins) and not game_seven:
            indeces_drop = playoff_df.loc[(playoff_df[cons.game_date_col] > game_dt) &
                                                    (((playoff_df[cons.home_team_name_col] == row[cons.home_team_name_col]) &
                                                    (playoff_df[cons.away_team_name_col] == row[cons.away_team_name_col])) |
                                                    ((playoff_df[cons.home_team_name_col] == row[cons.away_team_name_col]) &
                                                    (playoff_df[cons.away_team_name_col] == row[cons.home_team_name_col])))].index
            playoff_df.drop(index=indeces_drop, inplace=True)

        # print out the series results
        if home_team_wins:
            print(f'\t\t{row[cons.home_team_name_col]} advance past {row[cons.away_team_name_col]}: {int(row[cons.home_team_series_score_col]+1)}-{int(row[cons.away_team_series_score_col])}')
        elif away_team_wins:
            print(f'\t\t{row[cons.away_team_name_col]} advance past {row[cons.home_team_name_col]}: {int(row[cons.away_team_series_score_col]+1)}-{int(row[cons.home_team_series_score_col])}')
    
    return playoff_df, all_matchups


if __name__ == '__main__':
    from file_utils import csvLoad

    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)
    # today_dt = '2025-10-01' # beginning of 20252026 season
    # today_dt = '2026-02-24' # end of Olympic break

    feature_df = csvLoad(cons.season_pred_folder.format(date=today_dt), cons.season_pred_filename.format(date=today_dt))
    
    final_standings_df = csvLoad(cons.season_pred_folder.format(date=today_dt), cons.final_standings_filename.format(date=today_dt))

    playoff_tree_predictions(feature_df, final_standings_df, False, today_dt)