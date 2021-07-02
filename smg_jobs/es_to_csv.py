import sys

import rdflib

sys.path.append("..")

from heritageconnector.config import config
from heritageconnector.datastore import es_to_rdflib_graph, wikidump_to_rdflib_graph
from heritageconnector.logging import get_logger
from heritageconnector.namespace import SMGD, SMGO, SMGP
import csv

logger = get_logger(__name__)


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

    return g


if len(sys.argv) == 1:
    raise ValueError(
        "output format (csv/ntriples) and filename must be provided as arguments"
    )
if len(sys.argv) == 2:
    raise ValueError("missing either output format or filename")

method = sys.argv[1]
file_path = sys.argv[2]

logger.info("Creating and combining graphs from collection, blog and journal")
g_collection = es_to_rdflib_graph(index="heritageconnector")
g_blog = es_to_rdflib_graph(index="heritageconnector_blog")
g_journal = es_to_rdflib_graph(index="heritageconnector_journal")
g = g_collection + g_blog + g_journal

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
logger.info("Postprocessing graph")
g = postprocess_heritageconnector_graph(g)

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
