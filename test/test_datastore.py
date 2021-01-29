from heritageconnector import datastore
from heritageconnector.config import field_mapping
import pytest


@pytest.mark.skip(reason="relies on local fuseki instance running")
def test_ner_loader():
    """WARNING: this test has not been run but is meant to demonstrate NERLoader."""

    record_loader = datastore.RecordLoader("SMG_test", field_mapping)
    ner_loader = datastore.NERLoader(record_loader, batch_size=1024)
    ner_loader.add_ner_entities_to_es("en_core_web_lg", limit=None)
