# Heritage Connector

Transforming text into data to extract meaning and make connections. In development.

A set of tools to:

- load tabular collection data to a knowledge graph
- find links between collection entities and Wikidata
- perform NLP to create more links in the graph (also see [hc-nlp](https://github.com/TheScienceMuseum/heritage-connector-nlp))
- explore and analyse a collection graph ways that aren't possible in existing collections systems

![diagram: Relational DB vs Knowledge Graph](https://thesciencemuseum.github.io/heritageconnector/post_files/charts-knowledge-graphs-ml-post/1-relational-db-vs-knowledge-graph.png)
*Collections as tabular data (left) vs knowledge graphs (right)*

## Further Reading

The [main project page is here](https://www.sciencemuseumgroup.org.uk/project/heritage-connector/). We're also writing about our research on the [project blog](https://thesciencemuseum.github.io/heritageconnector) as we develop these tools and methods.

Some blog highlights:

- **Why are we doing this?** [*Sidestepping The Limitations Of Collection Catalogues With Machine Learning And Wikidata*](https://thesciencemuseum.github.io/heritageconnector/post/2020/09/23/sidestepping-the-limitations-of-collections-catalogues-with-machine-learning-and-wikidata/)
- **How will it work?** [*Knowledge Graphs, Machine Learning And Heritage Collections*](https://thesciencemuseum.github.io/heritageconnector/post/2020/11/06/knowledge-graphs-machine-learning-and-heritage-collections/)
- **Webinars:** [*Wikidata And Cultural Heritage Collections*](https://thesciencemuseum.github.io/heritageconnector/events/2020/06/22/wikidata-and-cultural-heritage-collections-webinar/), [*Connecting The UK's Cultural Heritage*](https://thesciencemuseum.github.io/heritageconnector/events/2020/11/06/connecting-the-uks-cultural-heritage/)

## For Developers (TODO: put in docs)

- Python 3
- Create a new branch / Pull Request for each new feature / unit of functionality

### Installation

We use pipenv for dependency management. You can also install dependencies from `requirements.txt` and dev dependencies from `requirements_dev.txt`.

**Optional dependencies (for experimental features):**

- `torch`, `dgl`, `dgl-ke`: KG embeddings
- `spacy-nightly`: export to spaCy KnowledgeBase for Named Entity Linking 
### Running tests

Run `python -m pytest` with optional `--cov=heritageconnector` for a coverage report.

We use `pytest` for tests, and all tests are in *./test*. 

## Running

To run web app (in development): `python -m heritageconnector.web.app`

## Deployment
TODO
