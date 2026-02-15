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

def intro_screen():
    print('\nWelcome to NHL Metrics!')
    print('This program allows you to fetch and analyze NHL teams data.')
    print('Let\'s get started!\n')


def main_screen():

    options = cons.main_options

    terminal_ui = terminal_input(options, exit=True)
    terminal_ui.display_options()
    terminal_ui.receive_user_input()

    return terminal_ui


def exit_program():
    print('\nExiting the program. Goodbye!\n')
    exit()


def team_stats_screen():

    options = cons.team_stats_options

    terminal_ui = terminal_input(options, exit=True)
    terminal_ui.display_options()
    terminal_ui.receive_user_input()

    return terminal_ui