# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/formatters/colors.py

CSI = "\033["
OSC = "\033]"
BEL = "\007"


def code_to_chars(code):
    return CSI + str(code) + "m"


def set_title(title):
    return OSC + "2;" + title + BEL


def clear_screen(mode=2):
    return CSI + str(mode) + "J"


def clear_line(mode=2):
    return CSI + str(mode) + "K"


class AnsiCodes:
    def __init__(self):
        # the subclasses declare class attributes which are numbers.
        # Upon instantiation we define instance attributes, which are the same
        # as the class attributes but wrapped with the ANSI escape sequence
        for name in dir(self):
            if not name.startswith("_"):
                value = getattr(self, name)
                setattr(self, name, code_to_chars(value))


class AnsiCursor:
    @staticmethod
    def UP(n=1):
        return CSI + str(n) + "A"

    @staticmethod
    def DOWN(n=1):
        return CSI + str(n) + "B"

    @staticmethod
    def FORWARD(n=1):
        return CSI + str(n) + "C"

    @staticmethod
    def BACK(n=1):
        return CSI + str(n) + "D"

    @staticmethod
    def POS(x=1, y=1):
        return CSI + str(y) + ";" + str(x) + "H"


class AnsiFore(AnsiCodes):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    RESET = 39

    # These are fairly well supported, but not part of the standard.
    LIGHTBLACK_EX = 90
    LIGHTRED_EX = 91
    LIGHTGREEN_EX = 92
    LIGHTYELLOW_EX = 93
    LIGHTBLUE_EX = 94
    LIGHTMAGENTA_EX = 95
    LIGHTCYAN_EX = 96
    LIGHTWHITE_EX = 97


class AnsiBack(AnsiCodes):
    BLACK = 40
    RED = 41
    GREEN = 42
    YELLOW = 43
    BLUE = 44
    MAGENTA = 45
    CYAN = 46
    WHITE = 47
    RESET = 49

    # These are fairly well supported, but not part of the standard.
    LIGHTBLACK_EX = 100
    LIGHTRED_EX = 101
    LIGHTGREEN_EX = 102
    LIGHTYELLOW_EX = 103
    LIGHTBLUE_EX = 104
    LIGHTMAGENTA_EX = 105
    LIGHTCYAN_EX = 106
    LIGHTWHITE_EX = 107


class AnsiStyle(AnsiCodes):
    BRIGHT = 1
    DIM = 2
    NORMAL = 22
    RESET_ALL = 0


Fore = AnsiFore()
Back = AnsiBack()
Style = AnsiStyle()
Cursor = AnsiCursor()
