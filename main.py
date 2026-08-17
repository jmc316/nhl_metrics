from schedule import sched_update
from features import feat_update
from ui_trn_inf import ui_train_model, ui_run_inference, ui_update_playoff_spot_probabilities

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
            case 'Update Schedule Data':
                sched_update()
            case 'Update Feature Data':
                feat_update(save_feat_data=True, verbose=True)
            case 'Train Model':
                ui_train_model()
            case 'Run Inference':
                ui_run_inference()
            case 'Playoff Spot Probability':
                ui_update_playoff_spot_probabilities()
            case 'Exit':
                tui.exit_program()


if __name__ == "__main__":
    main()