Reconciling Fields to Wikidata QIDs
====================================

After loading data into a JSON-LD knowledge graph you can quickly start to introduce `Wikidata QIDs`_ into your data. The :py:meth:`heritageconnector.entity_matching.reconciler` module lets you convert values in a content table column into QIDs.

.. _wikidata qids: https://www.wikidata.org/wiki/Wikidata:Identifiers

This reconciler is meant as a **complement, not replacement** to other reconciliation services. It:

* Uses an Elasticsearch Wikidata dump instead of an OpenRefine reconciliation API [#reconciliation_api]_ for speed with large datasets.
* Allows control over areas of the Wikidata ontology to include and exclude when searching for QIDs.
* Is best used to map sets of values which are well-represented in Wikidata. There may be better data sources or reconciliation services for your data, such as *GeoNames* for place names or *OpenCorporates* for company names.

Running the Reconciler
-----------------------

Below is a minimal and fully-functioning example showing how to use the reconciler. Each step is broken down afterwards.

.. code-block:: python

    import pandas as pd
    from heritageconnector.entity_matching.reconciler import (
        Reconciler, 
        export_map_df_to_csv, 
        import_map_df_from_csv, 
        create_column_from_map_df
    )
    
    # Create a Reconciler instance using the Wikidata dump Elasticsearch index specified 
    # in config.ini
    rec = Reconciler(es_connector="from_config", es_index="from_config")
    
    # Process a column of a DataFrame to create a DataFrame mapping original column 
    # values to candidate QIDs
    data = pd.DataFrame.from_dict({"item_name": ["photograph", "camera", "model"]})
    map_df = rec.process_column(
        data["item_name"],
        multiple_vals=False,
        class_include="Q488383",
        class_exclude=["Q5", "Q43229", "Q28640", "Q618123", "Q16222597"],
    )

    # Export mapping DataFrame to CSV to review and make any changes, then import 
    # it again
    export_map_df_to_csv(map_df, "./review_data.csv")
    imported_map_df = import_map_df_from_csv("./test_data.csv")

    # Create a new column from the mapping DataFrame
    mapped_column = create_column_from_map_df(
        data["item_name"], imported_map_df, multiple_vals=False
    )

Detailed Steps
--------------

1. Create a Wikidata dump
*************************

See the *Elasticsearch - Wikidata dump* section of `First Steps <getting_started>`_ for instructions on setting up an Elasticsearch Wikidata dump using :code:`elastic_wikidata` [#elastic_wikidata]_.

2. Check data format
*********************

:code:`Reconciler` takes in a pandas Series (equivalent to a DataFrame column). For good results, the data in this Series should:

* be categorical (many values appear many times);
* correspond to an area of the Wikidata ontology. 

The Series must also have either all *list-like* or *string-like* values, which will determine the value of the boolean parameter :code:`multiple_valuess` in the next step.

1. Process data to create a mapping table
******************************************

First, create an instance of :py:meth:`heritageconnector.entity_matching.reconciler.Reconciler`. The default :code:`"from_config"` values means that the Elasticsearch instance and index defined in :code:`config.ini` are used.

.. code-block:: python

    rec = Reconciler(
        es_connector: Union[elasticsearch.client.Elasticsearch, str] = "from_config",
        es_index: str = "from_config"
    )


The :py:meth:`heritageconnector.entity_matching.reconciler.Reconciler.process_column` method can then be used to create a *mapping table* between values in a pandas Series and Wikidata QIDs, according to a series of constraints.

The constraints rely on the concept of the *Wikidata class tree*: a series of *subclass of (P279)* properties connecting together QIDs in an ontology structure. For example we want *camera* (photography equipment) to resolve to *camera (Q15328)* which is in the subclass tree of *object (Q488383)*, and not *camera (Q97301845)*, which is in the subclass tree of *geographical feature (Q618123)*. This constraint can be expressed using the :code:`class_include` and :code:`class_exclude` arguments as in the following example.

Instead of providing a value for the :code:`class_include` argument, the :code:`pid` argument can be passed instead, which uses the property *subject item of this property (P1629)* to get a value for :code:`class_include` from a PID.

.. code-block:: python

    >>> data = pd.DataFrame.from_dict({"item_name": ["photograph", "camera", "model"]}) # sample data with object types

    >>> map_df = rec.process_column(
    >>>     data["item_name"],
    >>>     multiple_vals=False,
    >>>     class_include="Q488383",
    >>>     class_exclude=["Q618123"],
    >>> )
    
    >>> map_df
                count                                               qids          filtered_qids
    photograph      1                                          [Q125191]              [Q125191]
    camera          1                      [Q15328, Q5025979, Q97301845]               [Q15328]
    model           1  [Q1979154, Q1941828, Q10929058, Q4610556, Q573...  [Q1979154, Q10929058]

There are two columns containing QIDs in the mapping table:

* :code:`qids` - QIDs returned by searching for the term in the Wikidata dump
* :code:`filtered_qids` - the QIDs in :code:`qids`, filtered according to the subclass tree specified in :code:`rec.process_column()`
    

4. Manually review the mapping table
*************************************

The mapping table produced in step 3 can be used directly to create a new column, but may be a good idea to review it to fill in any gaps manually, or to refine the subclass tree defined in the last step.

The following methods safely export and import the mapping table for use with the :code:`Reconciler` class. 

* To export the mapping table to CSV for review, use :py:meth:`heritageconnector.entity_matching.reconciler.export_map_df_to_csv`
* To import the modified mapping table from CSV, use :py:meth:`heritageconnector.entity_matching.reconciler.import_map_df_from_csv`


5. Create a new column from the mapping table
**********************************************

Finally, you can create a new Series from the mapping table. 

.. code-block:: python

    >>> reconciler.create_column_from_map_df(data.item_name, map_df, multiple_vals=False)
    100%|█████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 2485.27it/s]
    0                [Q125191]
    1                 [Q15328]
    2    [Q1979154, Q10929058]
    Name: item_name, dtype: object
    
It may be useful to save the mapping table CSV for reproducibility.


---

.. [#reconciliation_api] https://docs.openrefine.org/technical-reference/reconciliation-api. The official list of reconcilation APIs that can be used with OpenRefine is here: https://reconciliation-api.github.io/testbench/
.. [#elastic_wikidata] https://github.com/TheScienceMuseum/elastic-wikidata

