Creating a SPARQL Store
=======================

The Heritage Connector includes scripts to create a `Fuseki <https://jena.apache.org/documentation/fuseki2/>`_ SPARQL store from the main HC database, an Elasticsearch index containing JSON-LD formatted documents. 

Prerequisites
-------------

* Java runtime environment: https://www.java.com/en/download/
* Apache Fuseki (server is the version with a UI for querying): https://jena.apache.org/documentation/fuseki2/
* Apache Jena: https://jena.apache.org/download/

The scripts for this tutorial are in *./bin/fuseki* in the repo.

Steps
-----

.. code-block:: bash

    # go to scripts directory and make both scripts executable
    cd ./bin/fuseki
    chmod +x ./load_data_from_es.sh ./start_fuseki_server.sh

    # --- 1. LOAD DATA FROM ELASTICSEARCH ---
    # path_to_tdb2_db = path to an empty folder
    # path_for_rdf_file = filename to store the Elasticsearch RDF export

    ./load_data_from_es.sh $path_to_tdb2_db $path_for_rdf_file

    # --- 2. START FUSEKI SERVER ---

    ./start_fuseki_server.sh $path_to_tdb2_db
