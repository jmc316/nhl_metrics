from schedule import sched_update
from features.features import feat_update
from trn_inf import train_model, run_inference, update_playoff_spot_probabilities


def main():

    # display intro ui
    print('\nWelcome to NHL Metrics!')
    print('This program allows you to analyze team data, and view ML predictions.\n')

    while True:

        print('1. Update Schedule Data')
        print('2. Update Feature Data')
        print('3. Train Model')
        print('4. Run Inference')
        print('5. Playoff Spot Probability')
        print('6. Exit')
        user_response = input('> ')
        print()

        match user_response:
            case '1': # 'Update Schedule Data'
                sched_update()
            case '2': # 'Update Feature Data'
                feat_update(save_feat_data=True, verbose=True)
            case '3': # 'Train Model'
                train_model()
            case '4': # 'Run Inference'
                run_inference()
            case '5': # 'Playoff Spot Probability'
                update_playoff_spot_probabilities()
            case '6': # 'Exit'
                print('\nExiting the program. Goodbye!\n')
                exit()


if __name__ == "__main__":
    main()