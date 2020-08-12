import sys

sys.path.append("..")

from heritageconnector.datastore import es_to_rdflib_graph
import csv


if len(sys.argv) == 1:
    raise ValueError(
        "output format (csv/ntriples) and filename must be provided as arguments"
    )
if len(sys.argv) == 2:
    raise ValueError("missing either output format or filename")

method = sys.argv[1]
file_path = sys.argv[2]

g = es_to_rdflib_graph()

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
