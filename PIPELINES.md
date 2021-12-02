# Data load pipelines

## Data ingestion

Three ingestion scripts are at `smg_jobs/smg_loader.py`, `smg_jobs/smg_blog_journal_loader.py` and `vanda_jobs/vanda_loader.py`. They require ensuring that `config/field_mapping.py` is the correct version for the collection (there is also a `field_mapping_vanda.py` in the same folder). 

These scripts run the load from source CSVs and any CSVs created from URL lookup and record linkage processes. They then run NER and internal EL on the collection. Results of NER and NEL can be saved and then loaded using the `entity_list_save_path` and `entity_list_data_path` arguments.


## Lookup of existing URLs

See example in `run.py > jobs.match_people_orgs()`.


## Disambiguation/Record Linkage

Examples of running this process can be found in the `demos` folder. A CSV file with predicted `owl:sameAs` connections between SMG records and Wikidata records is outputted by each of these notebooks. This CSV can then be loaded into the Elasticsearch index using the chosen predicate - see `smg_jobs/smg_loader.py > load_sameas_from_disambiguator()`.

## Running BLINK - entity linking to Wikidata

There are a few steps to this: 

1. [Set up a BLINK endpoint](https://github.com/TheScienceMuseum/BLINK) on your localhost.
2. Run the script `smg_jobs/run_blink.py` for each Elasticsearch index you want to run through BLINK. You can change the entity types (different `hc:` predicates) that it runs on in the script. Each run of this script will produce a `.jsonl` file.
3. For each JSONL file, manually inspect some results and decide on a suitable confidence threshold. The default is 0.9.
4. For each JSONL file, then run `smg_jobs/load_blink_from_json.py` to load these links into the corresponding Elasticsearch index.

### Updating BLINK

The version of BLINK used for Heritage Connector is based on Wikipedia and Wikidata dumps from 2019. It would be ideal to be able to update this model to reflect the latest changes in both these sources, however a the moment there is no way to do this.

There are a few options for building up-to-date entity linking to Wikidata:

1. Consider using GENRE - a newer version of BLINK - which [seems to have the potential to be updated from Wikidata and Wikipedia](https://github.com/facebookresearch/GENRE/issues/12). Note that this training process is likely to be *expensive*.
2. Use another, cheaper entity linker, like [OpenTapioca](https://github.com/wetneb/opentapioca) or spaCy's EL, which is either kept in sync with Wikidata or is easier to update. *For Heritage Connector, I found that many of these systems performed too poorly for what we were trying to achieve. Some also do NER and EL in one step. Ideally you want one which looks up the results of the NER model, as it will a) produce better results and b) be easier to plug in to this existing code.*
3. Have a look for another, more up-to-date solution in the [benchmarks for entity linking to Wikidata](https://paperswithcode.com/sota/entity-linking-on-kilt-wned-wiki) (note that there might be another dataset/benchmark for entity linking to Wikidata by the time anyone reads this!). An easy way to integrate a new solution would to be to wrap it in a REST API which is similar to [TheScienceMuseum/BLINK](https://github.com/TheScienceMuseum/BLINK), as the code in this repo could then be used as-is.

## Creating a Wikidata cache.

See [elastic-wikidata](https://github.com/TheScienceMuseum/elastic-wikidata).

## Creating an RDF file from Elasticsearch indices

The `smg_jobs/es_to_csv.py` script does the heavy lifting to create an ntriples or CSV file from multiple JSON-LD Elasticsearch indices, as well as caching relevant parts of Wikidata. This could easily be rewritten for another project.