import sys

sys.path.append("..")
from heritageconnector.disambiguation.pipelines import build_training_data

if __name__ == "__main__":
    build_training_data(
        "wikidump_humans", "PERSON", page_size=100, search_limit=20, limit=100
    )
