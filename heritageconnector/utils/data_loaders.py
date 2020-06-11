from ..config import config
import pandas as pd
from logging import getLogger
logger = getLogger(__file__)

class local_loader():
    def __init__(self):
        self.catalogue_data_path = config.MIMSY_CATALOGUE_PATH
        self.people_data_path = config.MIMSY_PEOPLE_PATH

    def load_all(self):
        """
        Generic method for loading all data
        """

        all_data = dict()

        # load all data here
        all_data.update(self.load_mimsy_data())

        print(f"Loaded data from local path: {', '.join(list(all_data.keys()))}")

        return all_data

    def load_mimsy_data(self):
        """
        Load Mimsy Catalogue & People data from CSV paths specified in config.ini.

        Returns:
            mimsy_dict (dict): key-value pair for each mimsy data source
        """
        catalogue_df = self.load_mimsy_catalogue_data()
        people_orgs_df = self.load_mimsy_people_data()

        return {"mimsy_catalogue": catalogue_df, "mimsy_people": people_orgs_df}

    def load_mimsy_catalogue_data(self):
        """
        Loads catalogue data from local CSV file.

        Returns:
            catalogue_df (pd.dataframe)
        """

        catalogue_df = pd.read_csv(self.catalogue_data_path, low_memory=False)

        return catalogue_df

    def load_mimsy_people_data(self):
        """
        Loads people/organisation data from local CSV file.

        Returns:
            people_orgs_df (pd.dataframe)
        """

        people_orgs_df = pd.read_csv(self.people_data_path, low_memory=False)
        
        return people_orgs_df