import os
import spacy
from hc_nlp import pipeline, constants


def load_model(model_type: str, thesaurus_path=None):
    """
    Load the best spaCy pipeline for SMG data

    Args:
        model_type (str): spacy model type
        thesaurus_path (str): path to jsonl thesaurus (labels_all_unambiguous_types_people_orgs.jsonl)

    Returns:
        spacy model
    """
    if thesaurus_path is None:
        thesaurus_path = os.path.join(
            os.path.dirname(__file__),
            "../GITIGNORE_DATA/labels_all_unambiguous_types_people_orgs.jsonl",
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
            "thesaurus_path": thesaurus_path,
        },
        after="ner",
    )
    nlp.add_pipe("entity_filter", config={"ent_labels_ignore": ["DATE"]}, last=True)
    nlp.add_pipe("map_entity_types", last=True)

    return nlp
