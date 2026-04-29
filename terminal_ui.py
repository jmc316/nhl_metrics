import pandas as pd
import constants as cons

class terminal_input:

    def __init__(self, options, exit=False):
        if exit:
            options.update(cons.exit_option)
        self.options = options
        self.choice = None
        self.response = None

    def receive_user_input(self):
        while True:
            try:
                choice = int(input('> '))
                print()  # Add a newline for better readability
                if 1 <= choice <= len(self.options):
                    self.choice = choice
                    self.response = list(self.options)[choice-1]
                    break
                else:
                    print(f'Invalid input. Please enter a number between 1 and {len(self.options)}.')
            except ValueError:
                print('Invalid input. Please enter a valid number.')
                print()  # Add a newline for better readability

    def display_options(self):
        for idx, option in enumerate(self.options, start=1):
            print(f'{idx}. {option}')

    def get_response(self):
        return self.response
    

class terminal_input_int:
    def __init__(self, options):
        self.options = options
        self.choice = None
        self.response = None

    def receive_user_input(self):
        while True:
            try:
                choice = int(input('> '))
                print()  # Add a newline for better readability
                if min(self.options) <= choice <= max(self.options):
                    self.choice = choice
                    self.response = choice
                    break
                else:
                    print(f'Invalid input. Please enter a number between {min(self.options)} and {max(self.options)}.')
            except ValueError:
                print('Invalid input. Please enter a valid number.')
                print()  # Add a newline for better readability

    def display_options(self):
        print('Select the number of iterations to run the simulation for:')

    def get_response(self):
        return self.response
    

class terminal_input_dt:
    def __init__(self, options):
        self.min_dt = options[0]
        self.max_dt = options[1]
        self.choice = None
        self.response = None

    def receive_user_input(self):
        while True:
            try:
                choice = pd.to_datetime(input('> '), format='mixed').date()
                print()  # Add a newline for better readability
                if self.min_dt <= choice <= self.max_dt:
                    self.choice = choice
                    self.response = choice
                    break
                else:
                    print(f'Invalid input. Please enter a date between {self.min_dt.strftime(cons.date_format_yyyy_mm_dd)} and {self.max_dt.strftime(cons.date_format_yyyy_mm_dd)}.')
            except ValueError:
                print('Invalid input. Please enter a valid date in the format YYYY-MM-DD.')
                print()  # Add a newline for better readability

    def display_options(self):
        print('Select a date to run the simulations from:')

    def get_response(self):
        return self.response


def exit_program():
    print('\nExiting the program. Goodbye!\n')
    exit()


def team_stats_screen():

    options = cons.team_stats_options

    terminal_ui = terminal_input(options, exit=True)
    terminal_ui.display_options()
    terminal_ui.receive_user_input()

    return terminal_ui


def predict_screen():

    options = cons.update_predictions_options

    terminal_ui = terminal_input(options, exit=True)
    terminal_ui.display_options()
    terminal_ui.receive_user_input()

    return terminal_ui


def playoff_spot_screen():

    options = cons.playoff_spot_prob_options

    terminal_ui = terminal_input(options, exit=True)
    terminal_ui.display_options()
    terminal_ui.receive_user_input()

    return terminal_ui