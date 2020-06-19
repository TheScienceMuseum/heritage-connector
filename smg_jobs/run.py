import sys

sys.path.append("..")
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from tqdm import tqdm
import pandas as pd
import os

from heritageconnector.utils import data_loaders
from heritageconnector.lookup import from_url

tqdm.pandas()


def main():
    j = jobs()
    named_methods = {"lookup": j.lookup}
    method_names = ", ".join(list(named_methods.keys()))

    if len(sys.argv) == 1 or "help" in sys.argv[1]:
        print(f"Add an argument from the following: {method_names}")
        sys.exit()

    method_name = sys.argv[1]

    if method_name in named_methods.keys():
        return named_methods[method_name]()
    else:
        print(f"Choose from {method_names}")


class jobs:
    """
    load_data: load data from mimsy catalogue
    """

    def __init__(self):
        """
        Config goes here
        """
        self.data_folder = "../GITIGNORE_DATA"

        self.config = {
            "custom_patterns": [
                (
                    r"(?:Union List of Artist Names Online|ULAN)[A-Za-z\s:.]+ID:\s?(500\d{6})",
                    "P245",
                ),
                (r"doi:10.1093/ref:odnb/(\d{1,6})", "P1415",),
            ],
            "data_model": {"people_freetext": ["DESCRIPTION", "NOTE"]},
        }

    def load_data(self):
        loader = data_loaders.local_loader()
        data_dict = loader.load_all()
        self.catalogue_df, self.people_df = (
            data_dict["mimsy_catalogue"],
            data_dict["mimsy_people"],
        )

    def lookup(self, export=True):
        """
        Performs manual lookup on URLs and custom patterns, and exports results to data_folder/lookup_result.pkl.
        """
        self.load_data()

        print("Performing lookup...")
        df = self.people_df.copy()
        cols_look_for_url = self.config["data_model"]["people_freetext"]

        df[cols_look_for_url] = df[cols_look_for_url].astype(str)
        df["res_ALL_NOTES"] = df[cols_look_for_url].agg(" --- ".join, axis=1)

        wid = from_url.wikidata_id(custom_patterns=self.config["custom_patterns"])

        def text_to_urls(text):
            return wid.get_from_free_text(text, return_urls=True)

        df["res_WIKIDATA_IDs"], df["res_URLS"] = zip(
            *df["res_ALL_NOTES"].progress_apply(text_to_urls)
        )

        if export:
            df.to_pickle(os.path.join(self.data_folder, "lookup_result.pkl"))


if __name__ == "__main__":
    main()
