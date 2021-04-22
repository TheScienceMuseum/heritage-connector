import argparse
import json
import os
import sys
import time
import urllib.parse
from datetime import datetime

from utils.extract import bz2Reader
from utils.transforming import object_joins

"""
SETTING UP THE ARGUMENT PARSER
Example `python3 scripts/join-table-generations.py -j ./data/elastic-export/objects/all `
"""

def folder_date():
    date_today = datetime.now()
    return date_today.strftime("%Y%m%d")

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json_input", help="BZ2 JSON file path with objects to import", required=True)
    parser.add_argument("-o", "--json_output", help="Path for content table output", required=True)
    return parser.parse_args(args[1:])


def main(argv):
    args = parse_args(argv)

    # Extract records from the esdump file
    documents = bz2Reader(args.json_input)

    if documents:
        output_path = args.json_output
        filename = output_path + '/' + folder_date() + '/joins'  + \
            '.ndjson'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # truncate file if it already exists
        open(filename, 'w').close()


        # Modify individual records
        for original_document in documents:
            # Send document for transforming based on index
            joins = object_joins(
                    original_document)
            # Append document to ndjson file
            # Post the modified individual object to json file
            if joins:
                for join in joins:
                    with open(filename, "a+") as dest:
                        dest.write(json.dumps(join))
                        dest.write("\n")
                

        print("Completed Script")
        exit(0)

    else:
        print("No documents extracted")
        exit(1)


# Run the main() script method
if __name__ == "__main__":
    sys.exit(main(sys.argv))
