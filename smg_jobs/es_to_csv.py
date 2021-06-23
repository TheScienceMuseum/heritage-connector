import sys

sys.path.append("..")

from heritageconnector.config import config
from heritageconnector.datastore import es_to_rdflib_graph, wikidump_to_rdflib_graph
from heritageconnector.logging import get_logger
import csv

logger = get_logger(__name__)

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
