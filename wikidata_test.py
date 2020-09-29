from heritageconnector.utils.sparql import get_sparql_results
from heritageconnector.config import config

endpoint = config.WIKIDATA_SPARQL_ENDPOINT


def test_entitysearch_query():
    query = """SELECT DISTINCT ?item ?itemLabel 
    WHERE
    {
        ?item wdt:P31/wdt:P279* wd:Q43229.
        SERVICE wikibase:mwapi {
            bd:serviceParam wikibase:api "EntitySearch" .
            bd:serviceParam wikibase:endpoint "www.wikidata.org" .
            bd:serviceParam mwapi:search "bank" .
            bd:serviceParam mwapi:language "en" .
            ?item wikibase:apiOutputItem mwapi:item .
            ?num wikibase:apiOrdinal true .
            }

        SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
        }
    }"""

    res = get_sparql_results(endpoint, query)

    # more than one result returned
    assert len(res["results"]["bindings"]) > 0


def test_shortestpath_query():
    query = """PREFIX gas: <http://www.bigdata.com/rdf/gas#>

    SELECT ?super (?aLength + ?bLength as ?length) WHERE {
    SERVICE gas:service {
        gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ;
                    gas:in wd:Q22687 ;
                    gas:traversalDirection "Forward" ;
                    gas:out ?super ;
                    gas:out1 ?aLength ;
                    gas:maxIterations 10 ;
                    gas:linkType wdt:P279 .
    }
    SERVICE gas:service {
        gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ;
                    gas:in wd:Q43229 ;
                    gas:traversalDirection "Forward" ;
                    gas:out ?super ;
                    gas:out1 ?bLength ;
                    gas:maxIterations 10 ;
                    gas:linkType wdt:P279 .
    }  
    } ORDER BY ?length
    LIMIT 1"""

    res = get_sparql_results(endpoint, query)
    assert int(float(res["results"]["bindings"][0]["length"]["value"])) == 2


def test_propertylookup_query():
    query = """
    SELECT ?item ?itemLabel ?itemDescription ?altLabel ?P570Label ?P569Label
    WHERE {
        VALUES (?item) { (wd:Q106481) (wd:Q46633) }
        OPTIONAL{ ?item wdt:P570 ?P570 .}
        OPTIONAL{ ?item wdt:P569 ?P569 .}

        OPTIONAL {
        ?item skos:altLabel ?altLabel .
        FILTER (lang(?altLabel) = "en")
        }

        SERVICE wikibase:label { 
        bd:serviceParam wikibase:language "en" .
        }
    }
    """

    res = get_sparql_results(endpoint, query)

    # one result for each entity
    assert len(res["results"]["bindings"]) == 2

    # one column for each value in the SELECT slug
    assert len(res["results"]["bindings"][0]) == 6
