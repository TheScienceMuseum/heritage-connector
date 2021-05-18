# Various scripts for SMG:
# - `wikistream` creates an ndjson file from a Wikidata dump using Maxlath's Wikibase dump filter
# - `wikidump` imports specific properties from the ndjson file created by `wikistream` into an Elasticsearch index using elastic-wikidata

PATH_TO_WIKIDATA_DUMP=/Volumes/Kalyan_SSD/SMG/wikidata/wikidata-dump.json.gz
PATH_TO_ELASTIC_WIKIDATA=~/Documents/SMG/elastic-wikidata
PATH_TO_PROPERTIES_CONFIG=./wikidump_pids.cfg
DUMP_INDEX=wikidump
OUTPUT_PATH_WIKISTREAM=/Volumes/Kalyan_SSD/SMG/wikidata/wikidata_stream.ndjson

# environment variables
elastic_cluster = $(ELASTICSEARCH_CLUSTER)
elastic_user = $(ELASTICSEARCH_USER)
elastic_password = $(ELASTICSEARCH_PASSWORD)

.PHONY: wikistream wikidump

# remove scholarly articles, taxon, wikimedia list article, wikinews article
# page showing categories is here: https://www.wikidata.org/wiki/Wikidata:Statistics
wikistream: 
	npm install -g wikibase-dump-filter
	brew install pbzip2
	npm install --global load-balance-lines
	NODE_OPTIONS=--max_old_space_size=4096
	nice -n+19 pbzip2 -cd < latest-all.json.bz2 | nice -n+19 load-balance-lines wikibase-dump-filter --languages en --claim '~P31:Q13442814&~P31:Q16521&~P31:Q13406463&~P31:Q17633526' > $(OUTPUT_PATH_WIKISTREAM)

wikidump: wikidump_pids.cfg
	ew dump -p $(OUTPUT_PATH_WIKISTREAM) --cluster $(elastic_cluster) --user $(elastic_user) --password $(elastic_password) --index $(DUMP_INDEX) --properties $(PATH_TO_PROPERTIES_CONFIG)