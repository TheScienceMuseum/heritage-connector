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

with open(file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(res)
