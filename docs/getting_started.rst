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

.. _import-collection-data:

3. Import collection data into the Heritage Connector graph
------------------------------------------------------------


---

.. [#elastic_wikidata] https://github.com/TheScienceMuseum/elastic-wikidata

