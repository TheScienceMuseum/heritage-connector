# Various scripts for SMG

# Elasticsearch credentials: elastic_cluster, elastic_user, elastic_password. In .gitignore.
include credentials.mk

PATH_TO_WIKIDATA_DUMP=/Volumes/Kalyan_SSD/SMG/wikidata/latest-all-170521.json.bz2
PATH_TO_ELASTIC_WIKIDATA=~/Documents/SMG/elastic-wikidata
PATH_TO_PROPERTIES_CONFIG=./wikidump_pids.cfg
DUMP_INDEX=wikidump
OUTPUT_PATH_WIKISTREAM=/Volumes/Kalyan_SSD/SMG/wikidata/wikidata_all_no_articles_170521.ndjson

.PHONY: wikistream wikidump dictionary_exhibitions

# --- Wikidata dump ---
# - `wikistream` creates an ndjson file from a Wikidata dump using Maxlath's Wikibase dump filter
# - `wikidump` imports specific properties from the ndjson file created by `wikistream` into an Elasticsearch index using elastic-wikidata
# ---------------------
# remove scholarly articles, taxon, wikimedia list article, wikinews article
# page showing categories is here: https://www.wikidata.org/wiki/Wikidata:Statistics
wikistream: 
	npm install -g wikibase-dump-filter
	brew install pbzip2
	npm install --global load-balance-lines
	NODE_OPTIONS=--max_old_space_size=4096
	nice -n+19 pbzip2 -cd < $(PATH_TO_WIKIDATA_DUMP) | nice -n+19 load-balance-lines wikibase-dump-filter --claim '~P31:Q13442814&~P31:Q16521&~P31:Q13406463&~P31:Q17633526' --languages en > $(OUTPUT_PATH_WIKISTREAM)

wikidump: wikidump_pids.cfg
	ew dump -p $(OUTPUT_PATH_WIKISTREAM) --cluster $(elastic_cluster) --user $(elastic_user) --password $(elastic_password) --index $(DUMP_INDEX) --properties $(PATH_TO_PROPERTIES_CONFIG)

# --- spaCy dictionaries ---
# `make dictionaries` -> make all dictionaries
# - `dictionary_exhibitions` creates a dictionary from exhibition lists
# - `dictionary_collection` creates a dictionary from names in the heritage connector graph`
dictionary_exhibitions:
	python scripts/make_events_dictionary.py -i ../GITIGNORE_DATA/make_data/events_NER/Temporary_Exhibitions_1912-1988.xlsx ../GITIGNORE_DATA/make_data/events_NER/ExhibitionsMimsy.xlsx -o ../GITIGNORE_DATA/make_data/output/dictionary_matcher_events.jsonl

dictionary_collection:
	python scripts/make_collection_dictionary.py -o ../GITIGNORE_DATA/make_data/output/dictionary_matcher_collection.jsonl

dictionaries: dictionary_exhibitions dictionary_collection
