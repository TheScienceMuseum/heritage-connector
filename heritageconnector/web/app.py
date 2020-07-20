import uvicorn
from fastapi import FastAPI
from heritageconnector import datastore
import logging
import os

app = FastAPI()
logger = logging.getLogger("heritageconnector.app")

# application data
data = {}


@app.on_event("startup")
def startup():
    data["graph"] = datastore.es_to_rdflib_graph()
    logger.info(f"{len(data['graph'])} triples loaded into RDF store")


@app.post("/query")
@app.get("/query")
def query(sparql: str):
    res = data["graph"].query(sparql)

    if "construct" in sparql.lower():
        return [{"subject": i[0], "predicate": i[1], "object": i[2]} for i in res]
    else:
        # e.g. SELECT
        return [i for i in res]


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    uvicorn.run(app, host="0.0.0.0", port=9000)
