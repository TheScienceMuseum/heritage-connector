#!/bin/bash
# A shell script that uses esdump to batch extract an entire elastic index
# Angela Wolff - 10/Dec/2020

INDEX=${1:-objects}

if [[ -z "${ES_HOST}" ]]; then
    ELASTIC_URL=http://elastic:changme@localhost:9200
else
    ELASTIC_URL="${ES_HOST}"
fi

# Set date time path variables
TODAY_PATH=$(date +%Y/%b/%d)
BASE_PATH=/Users/whatapalaver/Documents/github/heritage-connector-vam
DATA_PATH=$BASE_PATH/data/elastic-export/$INDEX/all

# Create directories for log files
mkdir -p $DATA_PATH/

elasticdump \
    --input=$ELASTIC_URL/$INDEX \
    --output=$DATA_PATH/$INDEX.jsonl \
    --limit=1000 \
    --type=data \
    --sourceOnly \
    --fileSize=50mb

bzip2 $DATA_PATH/*.jsonl