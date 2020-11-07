import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from heritageconnector import datastore, logging

this_path = os.path.dirname(__file__)
logger = logging.get_logger(__name__)

app = FastAPI()
app.mount(
    "/static", StaticFiles(directory=os.path.join(this_path, "static")), name="static"
)
templates = Jinja2Templates(directory=os.path.join(this_path, "static/templates"))

# application data
data = {}


@app.on_event("startup")
def startup():
    data["graph"] = datastore.es_to_rdflib_graph()
    logger.info(f"{len(data['graph'])} triples loaded into RDF store")


@app.get("/")
@app.post("/")
async def home(request: Request):
    res = await request.form()
    if "sparql_query" in res:
        data["request"] = res["sparql_query"]
        query(data["request"])

    return templates.TemplateResponse("forcedirected.html", {"request": request})


@app.get("/get_latest_response")
def get_latest_response():
    if "latest_response" in data:
        return data["latest_response"]
    else:
        return {}


@app.post("/query")
@app.get("/query")
def query(sparql: str):
    res = data["graph"].query(sparql)

    if "construct" in sparql.lower():
        data["latest_response"] = [
            {"subject": i[0], "predicate": i[1], "object": i[2]} for i in res
        ]
        return data["latest_response"]

    elif "select ?s ?p ?o" in sparql.lower():
        data["latest_response"] = [
            {"subject": i[0], "predicate": i[1], "object": i[2]} for i in res
        ]
        return data["latest_response"]

    else:
        # e.g. SELECT
        data["latest_response"] = [i for i in res]

        return data["latest_response"]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
