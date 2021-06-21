import os
import spacy
from hc_nlp import pipeline, constants
from heritageconnector.config import config


def load_model(
    model_type: str,
    collection_thesaurus_path: str = None,
    events_thesaurus_path: str = None,
):
    """
    Load the best spaCy pipeline for SMG data. Uses COLLECTION_THESAURUS_PATH and EVENTS_THESAURUS_PATH
    from config.ini.

    Args:
        model_type (str): spacy model type

    Returns:
        spacy model
    """
    _collection_thesaurus_path = collection_thesaurus_path or os.path.join(
        os.path.dirname(__file__),
        "../GITIGNORE_DATA/make_data/output/dictionary_matcher_collection.jsonl",
    )
    _events_thesaurus_path = events_thesaurus_path or os.path.join(
        os.path.dirname(__file__),
        "../GITIGNORE_DATA/make_data/output/dictionary_matcher_events.jsonl",
    )

    if model_type == "en_core_web_trf":
        activated = spacy.prefer_gpu()
        if activated:
            print("spacy using GPU")
        else:
            print("spacy tried to use GPU but failed")

    nlp = spacy.load(model_type)

    nlp.add_pipe("date_matcher", before="ner")
    nlp.add_pipe(
        "pattern_matcher",
        before="date_matcher",
        config={"patterns": constants.COLLECTION_NAME_PATTERNS},
    )
    nlp.add_pipe(
        "thesaurus_matcher",
        config={
            "case_sensitive": False,
            "overwrite_ents": False,
            "thesaurus_path": _collection_thesaurus_path,
        },
        after="ner",
    )
    nlp.rename_pipe("thesaurus_matcher", "thesaurus_matcher_collection")
    nlp.add_pipe(
        "thesaurus_matcher",
        config={
            "case_sensitive": True,
            "overwrite_ents": False,
            "thesaurus_path": _events_thesaurus_path,
        },
        before="ner",
    )
    nlp.rename_pipe("thesaurus_matcher", "thesaurus_matcher_events")
    nlp.add_pipe("entity_filter", config={"ent_labels_ignore": ["DATE"]}, last=True)
    nlp.add_pipe("map_entity_types", last=True)
    nlp.add_pipe("entity_joiner", last=True)
    # DuplicateEntityDetector should go after EntityJoiner, as it can then use any first names
    # referred to as 'firstname and othername surname' to mark other occurrences of that name
    # as duplicates.
    # See https://github.com/TheScienceMuseum/heritage-connector-nlp/commit/ada3b098eeff3d40dc847f8d92b64f215c7f4be3
    nlp.add_pipe("duplicate_entity_detector", last=True)

    return nlp
