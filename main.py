import constants as cons
import terminal_ui as tui


def main():

    # display intro ui
    tui.intro_screen()

    # display main ui
    main_ui = tui.main_screen()

    func_map = {option: getattr(__import__(module), func) for option, (module, func) in cons.main_options.items()}

    # call the function associated with the user's choice
    func_map[main_ui.get_response()]()


if __name__ == "__main__":
    main()