import sys

sys.path.append("..")

from heritageconnector.datastore import es_to_rdflib_graph
import csv

if len(sys.argv) == 1:
    raise ValueError("filename must be provided as argument")

file_path = sys.argv[1]

g = es_to_rdflib_graph()

res = g.query(
    """
SELECT ?s ?p ?o WHERE {?s ?p ?o}
"""
)

# to csv
# with open(file_path, "w", newline="") as f:
#     writer = csv.writer(f, delimiter="\t", quotechar='"')
#     writer.writerows(res)

# to ntriples
g.serialize(file_path, format="ntriples")
