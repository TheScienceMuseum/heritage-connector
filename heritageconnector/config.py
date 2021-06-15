from configparser import ConfigParser
import os
import sys
from heritageconnector import logging

this_path = os.path.dirname(__file__)

logger = logging.get_logger(__name__)


class LoadConfig:
    def __init__(self, file_name):
        parser = ConfigParser()
        parser.optionxform = str  # make option names case sensitive
        found = parser.read(file_name)
        if not found:
            logger.error(
                "No config file found. Ensure there is a file called `config.ini` in the config folder in the root of this repo. See `config/config.sample.ini` for an example."
            )
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
        if os.path.exists(os.path.join(this_path, "../config/field_mapping.py")):
            sys.path.append(os.path.join(this_path, "../config"))
            import field_mapping

            self.__dict__.update(field_mapping.__dict__)
        else:
            logger.error(
                "Could not find field mapping. Ensure there is a file called `field_mapping.py` in the config folder in the root of this repo. See `config/field_mapping.sample.py` for an example."
            )


config = LoadConfig(os.path.join(this_path, "../config/config.ini"))
field_mapping = LoadFieldMapping()
