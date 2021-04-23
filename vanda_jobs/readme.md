# VANDA data prep notes

- `GITIGNORE_DATA/elastic-export/` contains an elastic dump of index data
- `GITIGNORE_DATA/hc-import/` contains a dated folder with the processed data for use in the heritage connector

## Generate content tables

Currently taking a short extract from objects, person, organisations

Run from the root of repo:

- `python3 vanda_jobs/scripts/content-table-generations.py -i objects -j ./GITIGNORE_DATA/elastic_export/objects/all -o ./GITIGNORE_DATA/hc_import/content`
- `python3 vanda_jobs/scripts/content-table-generations.py -i persons -j ./GITIGNORE_DATA/elastic_export/persons/all -o ./GITIGNORE_DATA/hc_import/content`
- `python3 vanda_jobs/scripts/content-table-generations.py -i organisations -j ./GITIGNORE_DATA/elastic_export/organisations/all -o ./GITIGNORE_DATA/hc_import/content`

## Generate join tables

Run from the root of repo:

- `python3 vanda_jobs/scripts/join-table-generations.py -j ./GITIGNORE_DATA/elastic_export/objects/all -o ./GITIGNORE_DATA/hc_import/join`

## Load tables into Elastic

Run from vanda_jobs:

- `python vanda_loader.py`

### TODO

- [ ] Add places index
- [ ] Add specificity to relationship type based on association
- [x] Add content fields
- [x] Add associated fields
