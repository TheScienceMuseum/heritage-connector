Getting Started
===============

To set up an instance of Heritage Connector you need to complete the following steps.

* :ref:`set-up-dependencies`
* :ref:`prepare-collection-data`
* :ref:`import-collection-data`

.. _set-up-dependencies:

1. Set up databases and config.ini
--------------------------------------

Two databases are required for all functions of Heritage Connector to work (see *Architecture* page). The file :code:`config.ini` should be completed with details of these databases as per the example below.

Config.ini example
*******************

.. code-block:: ini

    [WIKIDATA]
    WIKIDATA_SPARQL_ENDPOINT = https://query.wikidata.org/sparql
    ; Supply an email address or other contact details as per 
    ; Wikimedia user-agent guidelines. HC takes care of formatting
    ; the User-Agent as per the guidelines.
    ; https://meta.wikimedia.org/wiki/User-Agent_policy
    CUSTOM_USER_AGENT = <contact-details>

    ; Any SPARQL endpoint from your hosted DB is fine here.
    [FUSEKI]
    FUSEKI_ENDPOINT = http://<RDF-endpoint>

    [ELASTIC]
    ; Fill in Elasticsearch details below, or comment the next
    ; 3 lines if using a local ES instance.
    ELASTIC_SEARCH_CLUSTER = <ES-endpoint>
    ELASTIC_SEARCH_USER = <ES-user>
    ELASTIC_SEARCH_PASSWORD = <ES-password>
    ; Change ELASTIC_SEARCH_INDEX to a suitable value
    ELASTIC_SEARCH_INDEX = heritageconnector
    ; ELASTIC_SEARCH_WIKI_INDEX holds a Wikidata dump in ES
    ELASTIC_SEARCH_WIKI_INDEX = wikidump
    ; These parameters can be left as default unless you are 
    ; having problems with bulk text search
    ES_BULK_CHUNK_SIZE = 1000
    ES_BULK_QUEUE_SIZE = 8

    ; TODO
    [DISAMBIGUATION]
    PIDS_IGNORE = P2283 P27
    PIDS_CATEGORICAL = P106 


Elasticsearch - Heritage Connector index
*****************************************

This index holds the JSON-LD formatted Heritage Connector graph. It is also the source data from which the RDF triplestore is created.

Either a local or hosted instance of Elasticsearch can be used. If using hosted instance the **ELASTIC_SEARCH_CLUSTER**, **ELASTIC_SEARCH_USER** and **ELASTIC_SEARCH_PASSWORD** parameters in :code:`config.ini` must be set. If these are not set Heritage Connector will try to use an Elasticsearch instance on localhost.

**ELASTIC_SEARCH_INDEX** can also be set to a value other than *heritageconnnector*. In practice we've found it useful to keep a *heritageconnector_test* index aside for testing by changing this parameter and running the import twice.

RDF Triplestore
****************

Any triplestore with a SPARQL endpoint can be used, with URI to the SPARQL endpoint assigned to **FUSEKI_ENDPOINT** in :code:`config.ini`.

A guide and Dockerfile to run a Fuseki instance from a hosted N-Triples (`.nt`) file can be found `here <https://github.com/TheScienceMuseum/fuseki-docker/>`_. We run our Fuseki instance on AWS :code:`t3a.large` due to Fuseki's high memory usage after a large number of subsequent API calls.

Elasticsearch - Wikidata dump
******************************

Running bulk text search on Wikidata is slow and unstable, so to circumvent this we create an Elasticsearch index containing a simplified version of the Wikidata JSON dump for a relevant subset of Wikidata. We do this using :code:`elastic_wikidata` [#elastic_wikidata]_.

The Elasticsearch index containing the Wikidata dump is denoted in :code:`config.ini` as **ELASTIC_SEARCH_WIKI_INDEX**.

.. _prepare-collection-data:

2. Prepare collection data and field_mapping.py
------------------------------------------------

There are just three requirements for collection data for it to be imported into the Heritage Connector graph:

1. it can be imported into a pandas DataFrame [#pandas_io]_;
2. DataFrames are separated into *content tables* which contain record information, and *join tables* which contain information about connections between records;
3. a :code:`field_mapping.py` file is provided which maps column names in content tables to RDF predicates.


Writing a field_mapping.py for content tables
**********************************************

The purpose of :code:`field_mapping.py` is to map column names in a tabular dataset to RDF predicates in the knowledge graph. It must contain two variables as per the example below:

* :code:`non_graph_predicates`: a list of RDF predicates that should be loaded into the *data* instead of *graph* field in Elasticsearch, meaning their values won't appear in the triplestore;
* :code:`mapping`: a dictionary with keys referring to each source table, which contains the mapping from source table column names to RDF predicates.

**Example field_mapping.py**

.. code-block:: python
    
    # These lines are necessary to import namespaces from heritageconnector.namespace
    import sys
    sys.path.append("..")

    # Here you can import all namespaces needed. Each namespace is an instance of rdflib.namespace.Namespace
    # or a class that inherits from rdflib.namespace.Namespace.
    from heritageconnector.namespace import XSD, FOAF, OWL, RDF, RDFS, PROV, SDO, WD, WDT, SKOS

    # PIDs to store in in _source.data rather than _source.graph in Elasticsearch, meaning they do not end up in the triplestore. You may want to do this for fields such as descriptions that won't add any more connections between entities in the graph.
    non_graph_predicates = [
        XSD.description,
    ]

    # The `mapping` variable stores the mappings between column names and RDF predicates for each content table.
    mapping = {
        "TABLENAME_1": 
            {   
                # Each column name takes the form 
                # {"dataframe-column-name": {
                #   "RDF": NAMESPACE.predicate_name
                # },
                # ... }
                # Some examples:
                "TITLE_NAME": {
                    "RDF": FOAF.title
                },
                "PREFERRED_NAME": {
                    "RDF": RDFS.label,
                },
                "FIRSTMID_NAME": {
                    "RDF": FOAF.givenName,
                },
                "LASTSUFF_NAME": {
                    "RDF": FOAF.familyName,
                },
                # TODO: add date -> year guidance in docs
                "BIRTH_DATE": {
                    "RDF": SDO.birthDate,
                },
                "DEATH_DATE": {
                    "RDF": SDO.deathDate,
                },
            },
        "TABLENAME_2": 
            {   
                # repeat for all other tables
            }
    }

.. _import-collection-data:

3. Import collection data into the Heritage Connector graph
------------------------------------------------------------


---

.. [#elastic_wikidata] https://github.com/TheScienceMuseum/elastic-wikidata
.. [#pandas_io] a full list of pandas functions to read in data from different formats is available here https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html?highlight=read

