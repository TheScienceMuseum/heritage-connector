import argparse
import json
import os
import sys
import time
import urllib.parse
from datetime import datetime

from utils.extract import bz2Reader, gzipReader
from utils.transforming import (objects_transforming,
                              organisations_transforming,
                              persons_transforming,
                              events_transforming)

"""
SETTING UP THE ARGUMENT PARSER
Example: `python3 scripts/content-table-generations.py -i objects -j ./data/elastic-export/objects/all `
"""

def folder_date():
    date_today = datetime.now()
    return date_today.strftime("%Y%m%d")

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", help="Name of index to use", required=True)
    parser.add_argument("-j", "--json_input", help="JSON file path to import", required=True)
    parser.add_argument("-o", "--json_output", help="Path for content table output", required=True)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-b", "--bz2_format", help="Is input file in bz2 format", action='store_true', default=False)
    group.add_argument("-g", "--gzip_format", help="Is input file in gzip format", action='store_true', default=False)
    
    return parser.parse_args(args[1:])


def main(argv):
    args = parse_args(argv)
    index = args.index

    # Extract records from the esdump file
    if args.gzip_format:
        documents = gzipReader(args.json_input)
    else:
        documents = bz2Reader(args.json_input)

    if documents:
        output_path = args.json_output
        filename = output_path + '/' + folder_date() + '/' + index  + \
            '.ndjson'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # truncate file if it already exists
        open(filename, 'w').close()

        # Modify individual records
        for original_document in documents:
            # Send document for transforming based on index
            if index == "objects":
                document = objects_transforming(
                    original_document)
            elif index == "persons":
                document = persons_transforming(original_document)
            elif index == "organisations":
                document = organisations_transforming(original_document)
            elif index == "events":
                document = events_transforming(original_document)
            else:
                document = original_document

            # Append document to ndjson file
            # Post the modified individual object to json file
            
            with open(filename, "a+") as dest:
                dest.write(json.dumps(document))
                dest.write("\n")
                

        print("Completed Script")
        exit(0)
    
    else:
        print("No documents extracted")
        exit(1)


# Run the main() script method
if __name__ == "__main__":
    sys.exit(main(sys.argv))
