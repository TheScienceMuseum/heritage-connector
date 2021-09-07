# VANDA Jobs

## VANDA preparatory work

- `GITIGNORE_DATA/elastic-export/` contains an elastic dump of index data
- `GITIGNORE_DATA/hc-import/` contains a dated folder with the processed data for use in the heritage connector
- `GITIGNORE_DATA/make_data/output` contains heritage connector output files

### Extract V&A Elastic Data for use in Heritage Connector

Run elastic dump of the authority indices events, organisations, persons.
Run a dump of either all or a subset of objects.
There is a custom query script for dumping subsets of object data in the vanda/etc_apps repo. Currently I am using the following query to extract objects that are likely to have a significant overlap with The Science Museum.

```bash
# Collection codes are:

# "THES48601": "Textiles and Fashion Collection",
# "THES48602": "Theatre and Performance Collection",

# the category codes are:

# THES48903 - Prints
# THES48976 - Clocks & Watches
# THES252963 - Posters
# THES488881 - Transport

elasticdump \
    --input=$ELASTIC_URL/$INDEX \
    --output=$DATA_PATH/$INDEX.jsonl \
    --searchBody="{\"query\": {\"bool\": {\"filter\": [{\"bool\": {\"should\": [{\"term\":{\"collectionCode.id\": \"THES48601\"}},{\"term\": {\"categories.id\": \"THES48903\"}},{\"term\": {\"categories.id\": \"THES252963\"}},{\"term\": {\"collectionCode.id\": \"THES48602\"}},{\"term\": {\"categories.id\": \"THES48976\"}},{\"term\": {\"categories.id\": \"THES48881\"}}]}}]}}}" \
    --limit=1000 \
    --type=data \
    --sourceOnly \
    --fileSize=50mb\
    --fsCompress
```

### Generate HC content tables

Currently taking a short extract from objects, person, organisations

Run from the root of repo:

- `python3 vanda_jobs/scripts/content-table-generations.py -i objects -g -j ./GITIGNORE_DATA/elastic_export/objects/custom -o ./GITIGNORE_DATA/hc_import/content`
- `python3 vanda_jobs/scripts/content-table-generations.py -i persons -b -j ./GITIGNORE_DATA/elastic_export/persons/all -o ./GITIGNORE_DATA/hc_import/content`
- `python3 vanda_jobs/scripts/content-table-generations.py -i organisations -b -j ./GITIGNORE_DATA/elastic_export/organisations/all -o ./GITIGNORE_DATA/hc_import/content`
- `python3 vanda_jobs/scripts/content-table-generations.py -i events -b -j ./GITIGNORE_DATA/elastic_export/events/all -o ./GITIGNORE_DATA/hc_import/content`

### Generate HC join tables

Run from the root of repo:

- `python3 vanda_jobs/scripts/join-table-generations.py -g -j ./GITIGNORE_DATA/elastic_export/objects/custom -o ./GITIGNORE_DATA/hc_import/join`

### Load tables into Elastic

Run from vanda_jobs:

- cd vanda_jobs
- Update paths in vanda_loader.py
- `python vanda_loader.py`

## Using The Heritage Connector

Current focus is on Named Entity Recognition (NER) and Entity Linkage (EL).

### Create a dictionary of collection labels

- Run `python scripts/make_collection_dictionary.py`
- Set output path to: `../GITIGNORE_DATA/make_data/output/dictionary_matcher_collection.jsonl`
