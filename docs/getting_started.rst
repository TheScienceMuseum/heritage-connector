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

Either a local or hosted instance of Elasticsearch can be used. If using hosted instance the **ELASTIC_SEARCH_CLUSTER**, **ELASTIC_SEARCH_USER** and **ELASTIC_SEARCH_PASSWORD** parameters in :code:`config.ini` must be set. If these are not set Heritage Connector will try to use an Elasticsearch instance on localhost.

**ELASTIC_SEARCH_INDEX** can also be set to a value other than 'heritageconnector'. In practice we've found it useful to keep a 'heritageconnector_test' index aside for testing by changing this parameter and running the import twice.

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

There are 3 requirements for collection data for it to be imported into the Heritage Connector graph:

1. it exists as a series of pandas DataFrames [#pandas_io]_;
2. the DataFrames are separated into *content tables* which contain record information, and *join tables* which contain information about connections between records;
3. a :code:`field_mapping.py` file is provided which maps column names in content tables to RDF predicates.

Examples of data tables
************************

**Content tables** must contain a column *URI* with unique URI identifiers of each record. All other columns to be loaded must be in :code:`field_mapping.py`.

+----------------------------------------------------------------+---------------------------------+-------------+
| URI                                                            | object_name                     | date_made   |
+================================================================+=================================+=============+
| https://collection.sciencemuseumgroup.org.uk/objects/co38127   | Gestetner diaphragm duplicator  | 1900        |
+----------------------------------------------------------------+---------------------------------+-------------+
| https://collection.sciencemuseumgroup.org.uk/objects/co38127   | Polaroid Land camera Model 95   | 1948        |
+----------------------------------------------------------------+---------------------------------+-------------+

**Join tables** must contain two columns containing URI identifiers of different records, which are already in the graph. You may have a third column which specifies the type of the relationship between the two records.

+-----------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| URI_1                                                           | URI_2                                                        | relationship    |
+=================================================================+==============================================================+=================+
| https://collection.sciencemuseumgroup.org.uk/objects/co146411   | https://collection.sciencemuseumgroup.org.uk/people/cp37182  | made_by         |
+-----------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| https://collection.sciencemuseumgroup.org.uk/objects/co8085283  | https://collection.sciencemuseumgroup.org.uk/people/cp61136  | manufactured_by |
+-----------------------------------------------------------------+--------------------------------------------------------------+-----------------+

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

All methods for loading data into the graph are in :py:meth:`heritageconnector.datastore.RecordLoader`, an instance of which can be created in a script to load data as follows.

.. code-block:: python

    from heritageconnector import datastore
    from heritageconnector.config import field_mapping

    record_loader = datastore.RecordLoader(
        collection_name="SMG", field_mapping=field_mapping
    )

:code:`RecordLoader` takes 2 arguments: a collection name, which is added to each Elasticsearch doc in :code:`doc['_source']['collection']` (but not the graph), and a :code:`field_mapping` dictionary which should come from HC config.

Adding data from content tables 
********************************

The :py:meth:`heritageconnector.datastore.RecordLoader.add_records` method is used to add records in bulk from a content table DataFrame, as per the example below.

.. autofunction:: heritageconnector.datastore.RecordLoader.add_records
    :noindex:

The required the argument :code:`table_name` sets the values of :code:`doc['_source']['type']` for each Elasticsearch record and :code:`skos:hasTopConcept` for each entity in the triplestore, meaning that an entity's source table can always be idenfified from both databases.

The optional argument :code:`add_type` sets the value of :code:`RDF.type` for all records in the table. A Wikidata entity is recommended here for entity matching purposes.

**Example of importing a content table using RecordLoader.add_records**

.. code-block:: python

    """
    Assumes an instance of RecordLoader has already been created.
    """

    from heritageconnector.namespace import WD

    # people_df is imported from a CSV, and already contains a column named 'URI'
    people_df = pd.read_csv(people_df, low_memory=False)
 
    table_name = "PERSON"
    record_loader.add_records(table_name, people_df, add_type=WD.Q5)

Adding data from join tables
*****************************

With the :py:meth:`heritageconnector.datastore.RecordLoader.add_triples` method you can add triple relationships between pairs of entities from a join table. 

.. autofunction:: heritageconnector.datastore.RecordLoader.add_triples
    :noindex:

**Example of importing a join table using RecordLoader.add_triples**

In the example below we also split the DataFrame to load several sets of triples in at the same time.

.. code-block:: python

    """
    Assumes an instance of RecordLoader has already been created.
    """

    from heritageconnector.namespace import FOAF, WDT

    person_object_relationships = pd.read_csv(maker_data_path, low_memory=False)

    # first load all triples with value of 'relation' column equal to 'maker'
    maker_df = person_object_relationships[person_object_relationships['relation'] == 'maker']
    record_loader.add_triples(maker_df, predicate=FOAF.maker, subject_col='object_uri', object_col='person_uri')

    # then add all triples with value of 'relation' column equal to 'user'.
    # WDT.P1535 is the Wikidata property for 'used by'
    user_df = person_object_relationships[person_object_relationships['relation'] == 'user']
    record_loader.add_triples(user_df, predicate=WDT.P1535, subject_col='object_uri', object_col='person_uri')


---

.. [#elastic_wikidata] https://github.com/TheScienceMuseum/elastic-wikidata
.. [#pandas_io] a full list of pandas functions to read in data from different formats is available here https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html?highlight=read

