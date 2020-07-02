import sys
import warnings
import pandas as pd
from tqdm import tqdm
import os
from fuzzywuzzy import fuzz

sys.path.append("..")
warnings.simplefilter(action="ignore", category=FutureWarning)
tqdm.pandas()

from heritageconnector.utils import data_loaders
from heritageconnector.entity_matching import lookup, filtering


def main():
    j = jobs()
    named_methods = {
        "lookup": j.lookup,
        "match_people_orgs": j.match_people_orgs,
        "urls_from_wikidata": j.urls_from_wikidata,
    }
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
            "wikidata_endpoint": "https://query.wikidata.org/sparql",
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

        wid = lookup.wikidata_id(custom_patterns=self.config["custom_patterns"])

        def text_to_urls(text):
            return wid.get_from_free_text(text, return_urls=True)

        df["res_WIKIDATA_IDs"], df["res_URLS"] = zip(
            *df["res_ALL_NOTES"].progress_apply(text_to_urls)
        )

        if export:
            export_path = os.path.join(self.data_folder, "results/lookup_result.pkl")
            df.to_pickle(export_path)
            print(f"Results exported to {export_path}")

        return df

    def match_people_orgs(self, export=True):
        """
        Record matching on people and organisations. 
        """

        print("Running lookup...")
        # df_lookup = self.lookup(export=False)
        df_lookup = pd.read_pickle(os.path.join(self.data_folder, "lookup_result.pkl"))
        people = df_lookup[df_lookup["GENDER"].isin(["M", "F"])].copy()
        orgs = df_lookup[df_lookup["GENDER"] == "N"].copy()

        print("Running filtering on found Wikidata references...")
        print("PEOPLE")
        f = filtering.Filter(dataframe=people, qcode_col="res_WIKIDATA_IDs")
        f.add_instanceof_filter("Q5", False)
        f.add_label_filter(
            "PREFERRED_NAME",
            threshold=80,
            include_aliases=True,
            fuzzy_match_scorer=fuzz.token_sort_ratio,
        )
        f.add_date_filter("BIRTH_DATE", "birthYear", 8)
        f.process_dataframe()
        f.view_stats()

        print("ORGS")
        fo = filtering.Filter(dataframe=orgs, qcode_col="res_WIKIDATA_IDs")
        fo.add_instanceof_filter("Q43229", True)
        fo.add_label_filter(
            "PREFERRED_NAME",
            threshold=80,
            include_aliases=True,
            fuzzy_match_scorer=fuzz.token_set_ratio,
        )
        fo.process_dataframe()
        fo.view_stats()

        people_filtered = f.get_dataframe()
        orgs_filtered = fo.get_dataframe()
        df_filtered = pd.concat([people_filtered, orgs_filtered])

        if export:
            export_path = os.path.join(
                self.data_folder, "results/filtering_people_orgs_result.pkl"
            )
            df_filtered.to_pickle(export_path)
            print(f"Results exported to {export_path}")

    def urls_from_wikidata(self, export=True):
        wikidata_endpoint = self.config["wikidata_endpoint"]

        res_df = lookup.get_internal_urls_from_wikidata(
            "collection.sciencemuseum.org.uk", wikidata_endpoint
        )

        if export:
            export_path = os.path.join(
                self.data_folder, "results/wikidata_url_lookup.csv"
            )
            res_df.to_csv(export_path, index=False)
            print(f"Results exported to {export_path}")


if __name__ == "__main__":
    main()
