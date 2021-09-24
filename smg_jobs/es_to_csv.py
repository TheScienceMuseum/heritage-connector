"""
Script to convert data stored in several Elasticsearch indices to triples. Also creates a relevant Wikidata cache.

Indices:
- heritageconnector: SMG collection
- heritageconnector_blog: SMG blog
- heritageconnector_journal: SMG journal
- heritageconnector_vanda: V&A collection
- config.ELASTIC_SEARCH_WIKI_INDEX (wikidump): Wikidata cache
"""

import sys
import csv
from typing import Literal

import rdflib
from rdflib.namespace import SDO
from tqdm.auto import tqdm

sys.path.append("..")

from heritageconnector.config import config
from heritageconnector.datastore import es_to_rdflib_graph, wikidump_to_rdflib_graph
from heritageconnector.logging import get_logger
from heritageconnector.namespace import SMGD, SMGO, SMGP, FOAF, OWL, HC, WD

logger = get_logger(__name__)

entity_terms = [
    "entityPERSON",
    "entityORG",
    "entityNORP",
    "entityFAC",
    "entityLOC",
    "entityOBJECT",
    "entityLANGUAGE",
    "entityDATE",
    "entityEVENT",
]


def postprocess_heritageconnector_graph(g: rdflib.Graph) -> rdflib.Graph:
    """Fixing issues in the graph after they've happened. All things on here should also exist as TODOs in the loader or elsewhere in the code,
    and should be taken out of this function when they're fixed.

    Args:
        g (rdflib.Graph): original graph

    Returns:
        rdflib.Graph: fixed graph
    """

    # Issue 1: caused by documents loading in NaN makers (people) as SMGP:nan.
    # Although this only applies to people, we remove NaN objects and documents just to be sure.
    g.remove((None, None, SMGP["nan"]))
    g.remove((None, None, SMGO["nan"]))
    g.remove((None, None, SMGD["nan"]))

    # Issue 2: `foaf:page` is used to represent identity relationships between Mimsy and
    # Adlib records, as the disambiguator relies on `owl:sameAs` only being between SMG
    # and Wikidata records.
    # Here we replace `foaf:page` with `owl:sameAs`.
    for s, p, o in g.triples((None, FOAF.page, None)):
        g.remove((s, p, o))
        g.add((s, OWL.sameAs, o))

    # Issue 3: literal objects for triples with `hc:entityTYPE` predicates have not been normalised
    # in any way. Here we convert them to lowercase.
    # TODO: in future, it may also be useful to lemmatize them too.
    for term in entity_terms:
        for s, p, o in g.triples((None, HC[term], None)):
            if isinstance(o, rdflib.Literal):
                g.remove((s, p, o))
                g.add((s, p, rdflib.Literal(o.lower())))

    # Issue 4: CSV encoding doesn't work as some descriptions contain newline characters, which are then
    # added to the entity spans.
    # Here we replace "\n" with " " for all objects in the graph, where the predicate is entityTYPE.
    # TODO: replace newline characters with spaces in loader.
    for term in entity_terms:
        for s, p, o in g.triples((None, HC[term], None)):
            if isinstance(o, rdflib.Literal) and ("\n" in o or "\r" in o):
                g.remove((s, p, o))
                g.add(
                    (
                        s,
                        p,
                        rdflib.Literal(o.replace("\r", " ").replace("\n", " ").strip()),
                    )
                )

    return g


def remove_unlabelled_wikidata_entities(g: rdflib.Graph) -> rdflib.Graph:
    """Remove Wikidata entities without labels from the graph. Operates inplace on `g`.

    By definition, entities with titles that are proper nouns should be the only Wikidata entities in our KG.
    Therefore, all Wikidata entities with lowercase titles (which, by the Wikidata style guide means they're not
    proper nouns) should not be in the KG.

    This should be run on the KG *after* the Wikidata cache is added.
    """
    logger.info("Removing Wikidata entities with no labels")

    for s, p, o in tqdm(
        g.triples((None, SDO.potentialAction, rdflib.Literal("delete")))
    ):
        g.remove((s, p, o))
        g.remove((None, None, s))


if len(sys.argv) == 1:
    raise ValueError(
        "output format (csv/ntriples) and filename must be provided as arguments"
    )
if len(sys.argv) == 2:
    raise ValueError("missing either output format or filename")

method = sys.argv[1]
file_path = sys.argv[2]

logger.info(
    "Creating and combining graphs from SMG collection, blog and journal, and V&A collection"
)
g_collection = es_to_rdflib_graph(index="heritageconnector")
g_blog = es_to_rdflib_graph(index="heritageconnector_blog")
g_journal = es_to_rdflib_graph(index="heritageconnector_journal")
g_vanda = es_to_rdflib_graph(index="heritageconnector_vanda")
g = g_collection + g_blog + g_journal + g_vanda

logger.info("Postprocessing graph")
g = postprocess_heritageconnector_graph(g)

logger.info("Creating Wikidata cache")
unique_wikidata_qids = [
    i[0].replace("http://www.wikidata.org/entity/", "")
    for i in list(
        g.query(
            """
    SELECT DISTINCT ?o WHERE{  
        ?s ?p ?o.
        FILTER(STRSTARTS(STR(?o), "http://www.wikidata.org/entity/")).
    }
    """
        )
    )
]

wiki_g = wikidump_to_rdflib_graph(
    config.ELASTIC_SEARCH_WIKI_INDEX, qids=unique_wikidata_qids, pids=None
)

g = g + wiki_g
remove_unlabelled_wikidata_entities(g)

if method == "csv":
    res = g.query(
        """
    SELECT ?s ?p ?o WHERE {?s ?p ?o}
    """
    )

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quotechar='"')
        writer.writerows(res)

elif method in ["ntriples", "nt"]:
    g.serialize(file_path, format="ntriples")
