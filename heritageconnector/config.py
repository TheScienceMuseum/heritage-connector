from configparser import ConfigParser
import os
import sys
from heritageconnector import logging

this_path = os.path.dirname(__file__)

logger = logging.get_logger(__name__)


class LoadConfig:
    def __init__(self, file_name: str):
        self.parser = ConfigParser()
        self.parser.optionxform = str  # make option names case sensitive
        self._load_config_from_file(
            os.path.join(this_path, "config.defaults.ini"), defaults=True
        )
        self._load_config_from_file(file_name)

    def _load_config_from_file(self, file_name: str, defaults: bool = False):
        found = self.parser.read(file_name)
        if not found and not defaults:
            logger.error(
                f"No config file found at {file_name}. Add one to use your own values. See an example at `config/config.sample.ini`."
            )
        for section in self.parser.sections():
            config_items = self.parser.items(section)

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
