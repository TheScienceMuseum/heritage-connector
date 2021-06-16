.. Heritage Connector documentation master file, created by
   sphinx-quickstart on Mon Aug  3 16:43:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Heritage Connector documentation
================================

The Heritage Connector is a framework for automatically connecting items within and across `GLAM`_ collections and  publications through linking with `Wikidata`_. It achieves this by applying the following techniques *(links in sidebar)*:

- data ingestion: tabular data to knowledge graphs (RDF); mining existing data for Wikidata connections;
- record linkage: training a machine learning model to predict :code:`owl:sameAs` links between collection items and Wikidata items;
- information retrieval: adding new entities and relations to the knowledge graph from text using named entity recognition (NER) and entity linking (NEL) techniques.

.. _glam: https://en.wikipedia.org/wiki/GLAM_(industry_sector)
.. _wikidata: https://www.wikidata.org/

.. note:: We're still actively developing the Heritage Connector and documentation will be written as features are added.
   
   Meanwhile, see some useful links below. 

Useful links
------------

- our `GitHub`_
- `Main project page`_ on the Science Museum Group website
- `Project blog`_ for project updates and technical explainers

.. _github: https://github.com/TheScienceMuseum/heritage-connector
.. _main project page: https://www.sciencemuseumgroup.org.uk/project/heritage-connector/
.. _project blog: https://thesciencemuseum.github.io/heritageconnector/

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   architecture
   getting_started
   
.. toctree::
   :maxdepth: 2
   :caption: Tasks
   
   reconciling_fields
   record_linkage
   
.. toctree::
   :maxdepth: 2
   :caption: Info
   
   bibliography
   about
   
.. toctree::
   :maxdepth: 4
   :caption: Reference
