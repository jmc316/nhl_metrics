import ui_predict
import ui_schedule

import constants as cons
import terminal_ui as tui


def main():

    # display intro ui
    print('\nWelcome to NHL Metrics!')
    print('This program allows you to analyze team data, and view ML predictions.\n')

    while True:

        # display main ui
        main_ui = tui.terminal_input(cons.main_options, exit=True)
        main_ui.display_options()
        main_ui.receive_user_input()
        user_response = main_ui.get_response()

        match user_response:
            case 'Update Season Schedule':
                ui_schedule.ui_update_season_schedule()
            case 'Update Predictions':
                ui_predict.ui_update_predictions()
            case 'Playoff Spot Probability':
                ui_predict.ui_update_playoff_spot_probabilities()
            case 'Exit':
                tui.exit_program()


if __name__ == "__main__":
    main()