from configparser import ConfigParser

class LoadConfig():
    def __init__(self, file_name):
        parser = ConfigParser()
        parser.optionxform = str  # make option names case sensitive
        found = parser.read(file_name)
        if not found:
            raise ValueError("No config file found.")
        for section in parser.sections():
            self.__dict__.update(parser.items(section))

config = LoadConfig('../config.ini')