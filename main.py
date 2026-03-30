import constants as cons
import terminal_ui as tui


def main_user():

    # display intro ui
    tui.intro_screen('user')

    while True:

        # display main ui
        main_ui = tui.main_user_screen()

        func_map = {option: getattr(__import__(module), func) for option, (module, func) in cons.main_user_options.items()}

        # call the function associated with the user's choice
        func_map[main_ui.get_response()]()


def main_admin():

    # display intro ui
    tui.intro_screen('admin')

    while True:

        # display main ui
        main_ui = tui.main_admin_screen()

        func_map = {option: getattr(__import__(module), func) for option, (module, func) in cons.main_admin_options.items()}

        # call the function associated with the user's choice
        func_map[main_ui.get_response()]()


if __name__ == "__main__":

    role = 'admin'  # or 'user'

    if role == 'admin':
        main_admin()
    elif role == 'user':
        main_user()
    else:
        print("Invalid role. Please choose 'admin' or 'user'.")
        exit()