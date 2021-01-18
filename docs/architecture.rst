Architecture
============

Heritage Connector contains three main modules: 

* **Ingestion**, for converting tabular data to RDF and mining it for Wikidata links; 
* **Entity Matching**, for matching entities to Wikidata using :code:`owl:sameAs` links; and 
* **Information Retrieval**, for generating new entities and relations in the graph from text data.  

The diagram below shows the parts of Heritage Connector as well as its dependencies.

.. figure:: ./_static/architecture.png
    
    Overall architecture of the Heritage Connector system. Some parts are optional.


Data
----

Any data can be imported into Heritage Connector as long as it can be processed into a Pandas DataFrame, and an RDF predicate is specified for each column in :code:`field_mapping.py`.

Dependencies
------------

There are two database dependencies for Heritage Connector: Elasticsearch and an RDF triplestore. We use Apache Fuseki for our triplestore, however any with a SPARQL endpoint should be compatible. Details to access these databases are set in :code:`config.ini` (see `Config`_ for details).

The JSON-LD version of the graph sits in an Elasticsearch index called `heritageconnector` by default. To enable entity matching with Wikidata another index must also be created from Wikidata as in ⓵ above. This can be created using elastic-wikidata. **TODO: Instructions/e-w config.**

Config
------

There are two necessary config files: 

* :code:`config.ini` specifies the details to connect to the databases, some extra Elasticsearch parameters, the Wikidata endpoint and user-agent, and some parameters for the disambiguator. 

* :code:`field_mapping.py` specifies the RDF predicate for each column in the tabular data. **TODO: needs example**

Examples of both are kept up to date on the *master* branch at :code:`config.sample.ini` and :code:`field_mapping.sample.py`.