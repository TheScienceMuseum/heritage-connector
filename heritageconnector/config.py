from configparser import ConfigParser
import os
import sys

this_path = os.path.dirname(__file__)


class LoadConfig:
    def __init__(self, file_name):
        parser = ConfigParser()
        parser.optionxform = str  # make option names case sensitive
        found = parser.read(file_name)
        if not found:
            raise ValueError("No config file found.")
        for section in parser.sections():
            # if in the LOCALPATH section of config, resolve path relative to config
            if section == "LOCALPATH":
                config_items = [
                    (i[0], os.path.join(this_path, i[1])) for i in parser.items(section)
                ]
            else:
                config_items = parser.items(section)

            self.__dict__.update(config_items)


class LoadFieldMapping:
    def __init__(self):
        sys.path.append(os.path.join(this_path, "../config"))
        import field_mapping

        self.__dict__.update(field_mapping.__dict__)


config = LoadConfig(os.path.join(this_path, "../config/config.ini"))
field_mapping = LoadFieldMapping()
