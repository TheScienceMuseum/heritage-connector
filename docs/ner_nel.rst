Named Entity Recognition and Entity Linking
============================================

Named Entity Recognition (NER) and Entity Linking (EL) are two different machine learning approaches that Heritage Connector uses to create knowledge graph connections from text. The pipeline to do this is as follows (see our blog post [#nelblogpost]_ for more details):

1. NER is run on all the *description* fields in the Elasticsearch index, and the extracted entities are saved back into the index, or into a local JSON file.
2. EL is run on these extracted entities to convert entity mentions into references to collection records, i.e. a connection between two entities in the KG. For example, the mention :code:`Colne Robotics` would be converted into https://collection.sciencemuseumgroup.org.uk/people/cp43944/colne-robotics-company-limited.

In the second step, any text mentions that can't be resolved to items in your collection will be left as text mentions, leaving some ambiguity. If this is an issue for your collection, you could add a third step to link to an external knowledge base such as Wikipedia or Wikidata. For Heritage Connector, we've adapted Facebook Research's `BLINK <https://github.com/facebookresearch/BLINK>`_ to provide an easy to use REST API which can be used to resolve an entity mention to either a Wikipedia page or a Wikidata record. [#smg_blink]_


Background Reading
-------------------

Heritage Connector's NER functionality uses `spaCy <https://spacy.io>`_, with some extensions for museum data and low-data environments provided by `heritage-connector-nlp <https://github.com/TheScienceMuseum/heritage-connector-nlp>`_. If you're not familiar with spaCy, check out its `NER implementation <https://spacy.io/usage/linguistic-features#named-entities>`_ and `models <https://spacy.io/models/en>`_ [#spacy_english]_. For more details on the extensions made to the models, see our paper [#paper]_.

The EL model is built directly into Heritage Connector's code, but is based strongly on *Klie et al.* [#klie_et_al]_. It's optimised for low training data (hundreds of labelled examples) rather than absolute performance. We recommend that you read the paper to understand the advantages and disadvantages of such a model.

The features we use for the EL model are below (slightly different to [#klie_et_al]_).

.. list-table:: Features used for Entity Linking model
   :widths: 10 10 25
   :header-rows: 1

   * - Named Entity property
     - Link Candidate property
     - Comparison Methods
   * - Mention
     - Title
     - Token sort similarity, Levenshtein similarity, Jaro-Winkler similarity, Mention in Title, Title in Mention
   * - Label (predicted type)
     - Type
     - One-hot-encoded exact match
   * - Context
     - Description
     - Jaro-Winkler similarity, Jaccard similarity, Sørenson-dice Similarity, cosine distance between sBERT embeddings [#sBERT]_

Process
--------

Methods to run NER and EL are all contained within the :py:meth:`heritageconnector.datastore.NERLoader` class, which persists data on predicted entities in :py:meth:`heritageconnector.datastore.NERLoader.entity_list` until this data is exported to JSON or loaded into Elasticsearch.

The steps to set up and run these processes are as follows.

NER
****

1. **Set up your problem.** 
   
   - What entity types do you want to detect, out of the `spaCy types <https://spacy.io/models/en#en_core_web_trf-labels>`_? 
   - Which of these will you be able to, and do you want to, link to records in your collection (e.g. we can only link :code:`PERSON` entities if the collection contains records for people)? 
   - Are you linking to the same Elasticsearch index as your descriptions are stored in, or a different one? 
   - Is there any text preprocessing you want to do to your descriptions before NER runs on them?

   Answers to these questions will form the :code:`kwargs` to the :py:meth:`heritageconnector.datastore.NERLoader` class.

   .. raw:: html

        <details>
        <summary><em>Code snippet</em></summary>

   .. code-block:: python

        from heritageconnector.datastore import RecordLoader, NERLoader
        from .my_utils import preprocess_text_for_ner

        # You should already have an instance of RecordLoader from 'First Steps'.
        record_loader = datastore.RecordLoader(
            collection_name="SMG", field_mapping=field_mapping
        )

        ner_loader = NERLoader(
            record_loader=record_loader,
            source_es_index="heritageconnector_test",
            target_es_index="heritageconnector_test",
            source_description_field="data.http://www.w3.org/2001/XMLSchema#description",
            target_context_field="data.http://www.w3.org/2001/XMLSchema#description",
            target_title_field="graph.@rdfs:label.@value",
            target_type_field="graph.@skos:hasTopConcept.@value",
            entity_types_to_link={
                "PERSON",
                "OBJECT",
                "ORG",
            },
            target_record_types=("PERSON", "OBJECT", "ORGANISATION"),
            text_preprocess_func=preprocess_text_for_ner,
        )
        

   .. raw:: html

        </details>

2. **Run NER.** The :py:meth:`heritageconnector.datastore.NERLoader.get_list_of_entities_from_source_index` method produces a JSON of record IDs and their named entities, which can then be used for entity linking or to load into the Heritage Connector Elasticsearch index. To perform this step you'll need to have selected a model type from the `spaCy models <https://spacy.io/models/en>`_. We recommend experimenting with batch_size - if running on a smaller model and CPU, you should be able to increase it to greater than the *16* below.

   .. raw:: html

        <details>
        <summary><em>Code snippet</em></summary>

   .. code-block:: python

        ner_loader.get_list_of_entities_from_source_index(
            model_type="en_core_web_trf", spacy_batch_size=16
        )

   .. raw:: html
        
        </details>

3. **Save the results to JSON or load them into the Elasticsearch index.** :code:`NERLoader.load_entities_into_source_index` loads the retrieved entities into the JSON-LD Elasticsearch index with the predicates :code:`hc:entityTYPE`, where type is the spaCy entity type. You can also export the entity list to a JSON file, so that in future the retrieved entities can be loaded into the Elasticsearch index without rerunning the NER model.

   .. raw:: html
        
        <details>
        <summary><em>Code snippet</em></summary>

   .. code-block:: python

        # To save the retrieved entities to JSON.
        # For now there are no link candidates (see next step) so we set `include_link_candidates=False`.
        ner_loader.export_entity_list_to_json(
            output_path="./entity_json_DATE.json", include_link_candidates=False
        )

        # To load the retrieved entities into the JSON-LD Elasticsearch index.
        # Because we have no trained linker, we set `force_load_without_linker=True`.
        ner_loader.load_entities_into_source_index(
            force_load_without_linker=True,
        )

   .. raw:: html
        
        </details>

EL
***

The entity linker, similarity to the record linker, works in two steps. First, a **search step** searches an entity mention in the target Elasticsearch index and retrieves a list of *link candidates*: possible records that represent the same real-world entity as the entity mention. Second, a **classification (or ranking) step** uses a machine learning classifier to compare the entity mention, its type and the text it was mentioned in to each link candidate, its type, and its description.

1. **Get link candidates (search).** Link candidates are retrieved by searching the entity mention against the title field (and an optional alias field specified using :code:`target_alias_field`), and retrieving the top *N* results. *N* should be selected so that it's high enough that the correct link appears in the top *N* results the majority of the time, but not too high that the computation overhead of the classifier becomes large. A good value to start with is 10 or 15.

   .. raw:: html

        <details>
        <summary><em>Code snippet</em></summary>

   .. code-block:: python

        N = 15
        ner_loader.get_link_candidates_from_target_index(candidates_per_entity_mention=N)

   .. raw:: html

        </details>
        
2. **Label training data.** Here we export training data for the entity linking model to Excel, via a DataFrame. We (only!) export the first 200,000 rows as Excel has a file size limit. 

   The exported Excel file will have a :code:`link_correct` column. To label the data, for each row, fill its value with a 1 if the entity and candidate should link, and a 0 if not. Any rows with no value in the :code:`link_correct` column will be ignored when the Excel file is imported as training data, so there's no need to delete rows you don't label.

   :code:`pip` might prompt you to install one or two more libraries to encode the dataframe to Excel, depending on the format you choose.
   
   .. raw:: html
        
        <details>
        <summary><em>Code snippet</em></summary>

   .. code-block:: python

        links_data = ner_loader.get_links_data_for_review()
        links_data.head(200000).to_excel("el_training_data.xlsx")

   .. raw:: html

        </details>

3. **Predict links using trained model, and load these links into Elasticsearch.** Finally, we import our labelled training data and train the EL model with it. When :code:`NERLoader.load_entities_into_source_index` is then called, it predicts and loads in links for every entity mention for which link candidates have been retrieved.

   The entity linker is a binary classifier (multilayer perceptron) whose threshold can be set by the :code:`linking_confidence_threshold` keyword argument in :code:`NERLoader.load_entities_into_source_index`.

   .. raw:: html

        <details>
        <summary><em>Code snippet</em></summary>

   .. code-block:: python

        train_df = pd.read_excel("el_training_data_labelled.xlsx", index_col=0)
        ner_loader.train_entity_linker(train_df)
        ner_loader.load_entities_into_source_index(
            linking_confidence_threshold = 0.8, 
            batch_size=32768,
            force_load_without_linker=False,
        )

   .. raw:: html

        </details>

    
---

.. [#nelblogpost] *History, AI and Knowledge Graphs* - https://thesciencemuseum.github.io/heritageconnector/post/2021/03/17/history-ai/

.. [#smg_blink] https://github.com/TheScienceMuseum/BLINK

.. [#spacy_english] Note that the Heritage Connector is designed around the English language, so we can't guarantee the extensions in *heritage-connector-nlp* will work well for other languages.

.. [#paper] Dutia, K, Stack, J. Heritage connector: A machine learning framework for building linked open data from museum collections. *Applied AI Letters. 2021*;e23. https://doi.org/10.1002/ail2.23

.. [#klie_et_al] Klie, Jan-Christoph, Eckart de Castilho, Richard, and Gurevych, Iryna. "From zero to hero: Human-in-the-loop entity linking in low resource domains." *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020.* http://dx.doi.org/10.18653/v1/2020.acl-main.624

.. [#sBERT] :code:`stsb-distilbert-base` model from `sentence_transformers <https://sbert.net/>`_