"""Runs on the JSON file produced by `run_blink`.
Takes two arguments: json path [1] and elasticsearch index [2].
"""

import sys

sys.path.append("..")

from heritageconnector.datastore import BLINKLoader

if __name__ == "__main__":
    if sys.argv[1] == "--help":
        print(
            """Runs on the JSON file produced by `run_blink`.
        Takes two arguments: json path [1] and elasticsearch index [2].
        """
        )

        sys.exit()

    json_path, es_index = sys.argv[1], sys.argv[2]
    BLINKLoader().load_blink_results_to_es_from_json(
        json_path, es_index, blink_threshold=0.9, raise_on_error=True
    )
