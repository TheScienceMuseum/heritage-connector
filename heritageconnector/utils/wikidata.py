import requests
from typing import List, Set, Union
from tqdm import tqdm
import re
import os
from itertools import product
import pandas as pd
from elastic_wikidata.wd_entities import get_entities, simplify_wbgetentities_result
from heritageconnector.config import config
from heritageconnector.utils.sparql import get_sparql_results
from heritageconnector.utils.generic import cache, paginate_list, flatten_list_of_lists
from heritageconnector import logging, errors
from heritageconnector.namespace import WDT

logger = logging.get_logger(__name__)


class wbentities:
    def __init__(self, api_timeout=8):
        self.timeout = api_timeout
        self.ge = get_entities()

    def get_properties(
        self,
        qids: list,
        pids: list,
        pids_to_label: Union[list, str] = None,
        replace_values_with_labels: bool = False,
        page_size: int = 50,
    ) -> pd.DataFrame:
        """
        Get Wikidata properties specified by `pids` and `pids_to_label` for entities specified by `qids`.

        Args:
            qids (list): list of Wikidata entities
            pids (list): list of Wikidata properties
            pids_to_label (Union[list, str], optional): list of Wikidata properties to get labels for (if their values are entities).
                Use "all" to get labels for all PIDs; None to get labels for no PIDs; or a list of PIDs to get labels for a subset of PIDs.
                If any PIDs in `pids_to_label` aren't in `pids`, values will still be returned for them. Defaults to None.
            page_size (int, optional): page size for API calls. Defaults to 50.

        Returns:
            pd.DataFrame: table of specified property values for entities, with null values specified by empty strings.
        """
        res_generator = self.ge.result_generator(
            qids, page_limit=page_size, timeout=self.timeout
        )

        if pids_to_label is not None:
            if isinstance(pids_to_label, list):
                pids_all = list(set(pids + pids_to_label))
            elif pids_to_label == "all":
                pids_all = list(set(pids))
                pids_to_label = pids_all
        else:
            pids_all = list(set(pids))

        docs = flatten_list_of_lists(
            [
                simplify_wbgetentities_result(
                    doc, lang="en", properties=pids_all, use_redirected_qid=False
                )
                for doc in res_generator
            ]
        )
        doc_df = pd.json_normalize(docs)

        # add columns with empty string values for any that are missing
        proposed_cols = self._pids_to_df_cols(pids_all)
        actual_cols = [col for col in doc_df.columns if col.startswith("claims")]
        extra_cols = list(set(proposed_cols) - set(actual_cols))

        for c in extra_cols:
            doc_df[c] = ""

        self.doc_df = doc_df

        if pids_to_label is not None:
            self.get_labels_for_properties(
                pids_to_label, replace_qids=replace_values_with_labels
            )

    def _pids_to_df_cols(self, pids: list) -> list:
        """Transform PIDs to doc_df column names: P79 -> claims.P79"""
        return [f"claims.{pid}" for pid in pids]

    def _replace_qid_with_label(self, v, return_v_if_missing: bool):
        """Replace QID with label from qid_label_mapping. Return original value if
        QID is not in keys of qid_label_mapping."""
        if (not isinstance(v, list)) or len(v) == 0:
            return v if return_v_if_missing else ""

        else:
            return [self.qid_label_mapping.get(i, i) for i in v]

    def _copy_and_clean_df_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace one-item lists with strings across the dataframe.
        Replace empty lists with empty strings.
        Replace nan values with empty strings.
        """

        export_df = df.copy()

        export_df = export_df.applymap(
            lambda i: i[0] if isinstance(i, list) and len(i) == 1 else i
        )
        export_df = export_df.applymap(
            lambda i: "" if isinstance(i, list) and len(i) == 0 else i
        )
        export_df = export_df.fillna("")

        return export_df

    def get_labels_for_properties(self, pids: list, replace_qids: bool):
        """
        Get labels for properties and add to self.doc_df.

        Args:
            pids (list): PIDs to get labels for
        """
        cols = self._pids_to_df_cols(pids)

        # make list of qids to get labels for
        qids_getlabels = []
        for idx, row in self.doc_df.iterrows():
            for col in cols:
                if isinstance(row[col], str) and is_qid(row[col]):
                    qids_getlabels.append(row[col])

                elif isinstance(row[col], list):
                    [qids_getlabels.append(val) for val in row[col] if is_qid(val)]

        # get labels if the list is not empty
        if len(qids_getlabels) > 0:
            qids_getlabels = list(set(qids_getlabels))
            self.qid_label_mapping = self.ge.get_labels(
                qids_getlabels, timeout=self.timeout
            )

            if replace_qids:
                for col in cols:
                    self.doc_df[col] = self.doc_df[col].map(
                        lambda i: self._replace_qid_with_label(
                            i, return_v_if_missing=True
                        )
                    )
            else:
                for col in cols:
                    self.doc_df[col + "Label"] = self.doc_df[col].map(
                        lambda i: self._replace_qid_with_label(
                            i, return_v_if_missing=False
                        )
                    )

    def get_results(self) -> pd.DataFrame:
        """
        Get dataframe with results.

        Returns:
            pd.DataFrame
        """

        return self._copy_and_clean_df_for_export(self.doc_df)


@cache(os.path.join(os.path.dirname(__file__), "../entitydistance.cache"))
def get_distance_between_entities_cached(
    qcode_set: Set[str],
    bidirectional: bool = False,
    vertex_pid: str = "P279",
    reciprocal: bool = False,
    max_path_length: int = 10,
) -> float:
    res = get_distance_between_entities(
        qcode_set, bidirectional, vertex_pid, reciprocal, max_path_length
    )

    return res


def get_distance_between_entities(
    qcode_set: Set[str],
    bidirectional: bool = False,
    vertex_pid: str = "P279",
    reciprocal: bool = False,
    max_path_length: int = 10,
) -> float:
    """
    Get the length of the shortest path between two entities in `qcode_set`along the 'subclass of' axis. 
    Flag `reciprocal=True` returns 1/(1+l) where l is the length of the shortest path, which can be treated as a similarity measure.

    Args:
        qcode_set (Set[str])
        bidirectional (bool, optional): If True, paths between entities where the direction is reversed (only once) will be considered. 
            Otherwise only the forward direction specified by the PID in `link_type` will be considered. Defaults to False.
        vertex_pid (str, optional): this PID specifies the edge types to use for the calculation.
        reciprocal (bool, optional): Return 1/(1+l), where l is the length of the shortest path. Defaults to False.
        max_iterations (int, optional): Maximum iterations to look for the shortest path. If the actual shortest path is
            greater than max_iterations, 10*max_iterations (reciprocal=False) or 1/(1+10*max_iterations) (reciprocal=True) is returned.

    Returns:
        float: distance (d <= max_iterations or max_iterations*10) or reciprocal distance (0 < d <= 1)
    """

    if len(qcode_set) == 1:
        # identity - assume two values have been passed in even though the set will have length 1
        return 1 if reciprocal else 0

    if len(qcode_set) != 2:
        raise ValueError("Input variable qcode_set must contain exactly 1 or 2 items")

    qcodes = [i for i in qcode_set]

    if (qcodes[0] == "") or (qcodes[1] == ""):
        # at least one value is empty so return maximum dissimilarity
        return 0 if reciprocal else 1
    else:
        raise_invalid_qid(qcodes[0])
        raise_invalid_qid(qcodes[1])

    if bidirectional:
        query = f"""PREFIX gas: <http://www.bigdata.com/rdf/gas#>

        SELECT ?super (?aLength + ?bLength as ?length) WHERE {{
        SERVICE gas:service {{
            gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ;
                        gas:in wd:{qcodes[0]} ;
                        gas:traversalDirection "Forward" ;
                        gas:out ?super ;
                        gas:out1 ?aLength ;
                        gas:maxIterations {max_path_length} ;
                        gas:linkType wdt:{vertex_pid} .
        }}
        SERVICE gas:service {{
            gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ;
                        gas:in wd:{qcodes[1]} ;
                        gas:traversalDirection "Forward" ;
                        gas:out ?super ;
                        gas:out1 ?bLength ;
                        gas:maxIterations {max_path_length} ;
                        gas:linkType wdt:{vertex_pid} .
        }}  
        }} ORDER BY ?length
        LIMIT 1
        """
    else:
        # NOTE: two distances are returned in this query to account for the fact that we don't know whether
        # qcodes[0] or qcodes[1] is higher in the hierarchy, and setting gas:traversalDirection "Undirected"
        # gives a WDQS error. One of these distances is zero as it's the distance between an entity and itself,
        # so the max of the two is returned by this function.

        query = f"""
        PREFIX gas: <http://www.bigdata.com/rdf/gas#>

        SELECT ?aLength ?bLength WHERE {{
        SERVICE gas:service {{
            gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ;
                        gas:in wd:{qcodes[0]} ;
            gas:traversalDirection "Forward" ;
                                gas:out ?super ;
                                gas:out1 ?aLength ;
                                gas:maxIterations {max_path_length} ;
            gas:linkType wdt:P279 .
        }}

        SERVICE gas:service {{
            gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ;
                        gas:in wd:{qcodes[1]} ;
            gas:traversalDirection "Forward" ;
                                gas:out ?super ;
                                gas:out1 ?bLength ;
                                gas:maxIterations {max_path_length} ;
            gas:linkType wdt:{vertex_pid} .
        }} 
        FILTER (?super in (wd:{qcodes[0]}, wd:{qcodes[1]})).
        }}
        """

    result = get_sparql_results(config.WIKIDATA_SPARQL_ENDPOINT, query)["results"][
        "bindings"
    ]

    if len(result) == 0:
        distance = 10 * max_path_length
    else:
        if bidirectional:
            distance = int(float(result[0]["length"]["value"]))
        else:
            distance = int(
                max(
                    float(result[0]["aLength"]["value"]),
                    float(result[0]["bLength"]["value"]),
                )
            )

    return 1 / (1 + distance) if reciprocal else distance


def get_distance_between_entities_multiple(
    qcode_set: Set[Union[str, tuple]],
    bidirectional: bool = False,
    vertex_pid: str = "P279",
    reciprocal: bool = False,
    max_path_length: int = 10,
    use_cache: bool = True,
) -> float:
    """
    Get the length of the shortest path between entities or sets of entities along the 'subclass of' axis. When two entities
    [a,b] are passed the shortest path is returned, and when two groups [(a,b), (c,d)] are passed the shortest path between
    the closest entities across those two groups is returned.

    Flag `reciprocal=True` returns 1/(1+l) where l is the length of the shortest path, which can be treated as a similarity measure.

    Args:
        qcode_set (Set[Union[str, list], Union[str, list]])
        reciprocal (bool, optional): Return 1/(1+l), where l is the length of the shortest path. Defaults to False.
        max_iterations (int, optional): Maximum iterations to look for the shortest path. If the actual shortest path is
            greater than max_iterations, 10*max_iterations (reciprocal=False) or 1/(1+10*max_iterations) (reciprocal=True) is returned.
        use_cache (bool, optional): whether to use a query cache stored on disk. Defaults to True

    Returns:
        Union[float, int]: distance (int <= max_iterations or max_iterations*10) or reciprocal distance (float, 0 < f <= 1)
    """

    if len(qcode_set) == 1:
        # identity - assume two values have been passed in even though the set will have length 1
        return 1 if reciprocal else 0

    if len(qcode_set) != 2:
        raise ValueError("Input variable qcode_set must contain exactly 1 or 2 items")

    # Convert set into list so we can access its individual values.
    # (The result of the distance function is independent of order)
    qcode_list = [i for i in qcode_set]
    qcode_1, qcode_2 = qcode_list

    if (qcode_1 == "") or (qcode_2 == ""):
        # one value is empty so return maximum dissimilarity
        return 0 if reciprocal else 1

    if isinstance(qcode_1, list):
        qcode_1 = tuple(qcode_1)
    if isinstance(qcode_2, list):
        qcode_2 = tuple(qcode_2)

    if isinstance(qcode_1, str):
        raise_invalid_qid(qcode_1)
        qcode_1 = [qcode_1]
    elif isinstance(qcode_1, tuple) and len(qcode_1) > 0:
        [raise_invalid_qid(q) for q in qcode_1]
    else:
        raise ValueError(
            f"Item of qcode_set {qcode_1} is either not a string or is an empty list"
        )

    if isinstance(qcode_2, str):
        raise_invalid_qid(qcode_2)
        qcode_2 = [qcode_2]
    elif isinstance(qcode_2, tuple) and len(qcode_2) > 0:
        [raise_invalid_qid(q) for q in qcode_2]
    else:
        raise ValueError(
            f"Item of qcode_set {qcode_2} is either not a string or is an empty list"
        )

    combinations = product(list(set(qcode_1)), list(set(qcode_2)))
    result_list = []

    if use_cache:
        for q1, q2 in combinations:
            result_list.append(
                get_distance_between_entities_cached(
                    {q1, q2},
                    bidirectional=bidirectional,
                    vertex_pid=vertex_pid,
                    reciprocal=reciprocal,
                    max_path_length=max_path_length,
                )
            )
    else:
        for q1, q2 in combinations:
            result_list.append(
                get_distance_between_entities(
                    {q1, q2},
                    bidirectional=bidirectional,
                    vertex_pid=vertex_pid,
                    reciprocal=reciprocal,
                    max_path_length=max_path_length,
                )
            )

    if reciprocal:
        return max(result_list)
    else:
        return min(result_list)


def get_wikidata_equivalents_for_properties(
    properties: List[str], raise_missing=False, warn_missing=True
) -> dict:
    """
    Get Wikidata equivalents for RDF properties.

    Args:
        properties (List[str]): list of URIs of properties
        raise_missing (bool, optional): If True, raises a ValueError if Wikidata equivalents
            can't be found for any of the specified properties. Defaults to False.
        warn_missing (bool, optional): If True, logs a warning if Wikidata equivalents
            can't be found for any of the specified properties. Defaults to True.

    Returns:
        dict: {property: wikidata_value, ...}. Any properties that don't have a corresponding
            Wikidata value will have value None unless they are already a Wikidata property value,
            in which case their key is the same as their value.
    """

    wiki_properties = [p for p in properties if p.startswith(str(WDT))]
    lookup_properties = list(set(properties) - set(wiki_properties))

    values_slug = " ".join(["<" + uri + ">" for uri in lookup_properties])

    query = f"""SELECT * WHERE {{
    VALUES ?internal_property {{ {values_slug} }}.
    ?wiki_property wdt:P1628 ?internal_property.
    }}"""

    res = get_sparql_results(config.WIKIDATA_SPARQL_ENDPOINT, query)["results"][
        "bindings"
    ]

    internal_wikidata_mapping = {
        item["internal_property"]["value"]: item["wiki_property"]["value"]
        for item in res
    }
    internal_wikidata_mapping.update({p: p for p in wiki_properties})

    missing_internal_vals = set(properties) - set(internal_wikidata_mapping.keys())

    if len(missing_internal_vals) > 0:
        if raise_missing:
            raise ValueError(
                f"Values {missing_internal_vals} are missing from results. To disable this raising an exception, set input raise_missing to False."
            )

        if warn_missing:
            logger.warning(f"Values {missing_internal_vals} are missing from results.")

    for val in missing_internal_vals:
        internal_wikidata_mapping[val] = None

    return internal_wikidata_mapping


def url_to_qid(url: Union[str, list], raise_invalid=True) -> Union[str, list]:
    """
    Maps Wikidata URL of an entity to QID e.g. http://www.wikidata.org/entity/Q7187777 -> Q7187777.

    Args:
        raise_invalid (bool, optional): whether to raise if a QID can't be found in the URL. If False,
            for any string in which a QID can't be found an empty string is returned.
    """

    if isinstance(url, str):
        found = re.findall(r"(Q\d+)", url)
        if len(found) == 1:
            return found[0]
        else:
            if raise_invalid:
                raise ValueError("URL does not contain a single QID")
            else:
                return ""

    elif isinstance(url, list):
        return [url_to_qid(i, raise_invalid) for i in url]


def qid_to_url(qid: Union[str, list]) -> Union[str, list]:
    """
    Maps QID of an entity to a Wikidata URL e.g. Q7187777 -> http://www.wikidata.org/entity/Q7187777.
    """

    if isinstance(qid, str):
        return f"http://www.wikidata.org/entity/{qid}"
    elif isinstance(qid, list):
        return [qid_to_url(i) for i in qid]


def url_to_pid(url: Union[str, list]) -> Union[str, list]:
    """
    Maps Wikidata URL of an entity to PID e.g. http://www.wikidata.org/prop/direct/P570 -> P570.
    """

    if isinstance(url, str):
        return re.findall(r"(P\d+)", url)[0]
    elif isinstance(url, list):
        return [url_to_pid(i) for i in url]


def pid_to_url(pid: str) -> str:
    """
    Maps PID of an entity to a Wikidata URL e.g. P570 -> http://www.wikidata.org/prop/direct/P570.
    """

    if isinstance(pid, str):
        return f"http://www.wikidata.org/entity/{pid}"
    elif isinstance(pid, list):
        return [pid_to_url(i) for i in pid]


def raise_invalid_qid(qid: str) -> str:
    """
    Raise ValueError if supplied value is not a valid QID
    """

    if not isinstance(qid, str):
        raise ValueError(f"QID {qid} is not of type string")

    if len(re.findall(r"(Q\d+)", qid)) != 1:
        raise ValueError(f"QID {qid} is not a valid QID")


def is_qid(val: str, case_sensitive=False) -> bool:
    """
    Returns boolean describing whether provided string is a QID.

    Args:
        val (str)
        case_sensitive (bool, optional): Defaults to False.

    Returns:
        bool
    """

    if not isinstance(val, str):
        return False

    if case_sensitive:
        return len(re.findall(r"(Q\d+)", val)) == 1
    else:
        return len(re.findall(r"(q\d+)", val.lower())) == 1


def join_qids_for_sparql_values_clause(qids: list) -> str:
    """
    Return joined list of QIDs for VALUES clause in a SPARQL query.
    E.g. VALUES ?item {wd:Q123 wd:Q234}

    Args:
        qids (list): list of QIDs

    Returns:
        str: QIDs formatted for VALUES clause
    """

    return " ".join([f"wd:{i}" for i in qids])


def year_from_wiki_date(datestr: Union[str, list], raise_invalid: bool = False) -> int:
    """
    Get the year from a date returned by the Wikidata wbgetentities API.
    Examples of dates are +1665-01-01T00:00:00Z, -0347-00-00T00:00:00Z.
    If a list is passed in, the function is called for each item in the list.

    Args:
        datestr (str): date string returned by the wbgetentities API
        raise_invalid (bool, optional): if False, the original value of datestr 
            is returned if it doesn't look like a date

    Returns:
        int: year (positive or negative)
    """

    if isinstance(datestr, list):
        return [year_from_wiki_date(item) for item in datestr]

    if len(re.findall(r"^(?:\+|-)\d{4}", str(datestr))) == 0:
        # invalid wikidate
        if raise_invalid:
            raise ValueError(
                f"Parameter datestr ({datestr}) doesn't start with +/- so is probably an invalid wbgetentities date."
            )
        else:
            return datestr

    if datestr[0] == "+":
        multiplier = 1
    elif datestr[0] == "-":
        multiplier = -1

    year = int(datestr[1:5])

    return multiplier * year


def filter_qids_in_class_tree(
    qids: list, higher_class: Union[str, list], classes_exclude: Union[str, list] = None
) -> list:
    """
    Returns filtered list of QIDs that exist in the class tree below the QID or any of
    the QIDs defined by `higher_class`. Raises if higher_class is not a valid QID.

    Args:
        qids (list): list of QIDs
        higher_class (Union[str, list]): QID or QIDs of higher class to filter on
        classes_exclude (Union[str, list]): QID or QIDs of higher classes to exclude. Defaults to None.

    Returns:
        list: unique list of filtered QIDs
    """

    formatted_qids = join_qids_for_sparql_values_clause(qids)

    # assume format of each item of qids has already been checked
    # TODO: what's a good pattern for coordinating this checking so it's not done multiple times?

    generate_exclude_slug = (
        lambda c: f"""MINUS {{?item wdt:P279* wd:{c}. hint:Prior hint:gearing "forward".}}."""
    )

    if classes_exclude:
        if isinstance(classes_exclude, str):
            raise_invalid_qid(classes_exclude)
            exclude_slug = generate_exclude_slug(classes_exclude)

        elif isinstance(classes_exclude, list):
            [raise_invalid_qid(c) for c in classes_exclude]
            exclude_slug = "\n".join(
                [generate_exclude_slug(c) for c in classes_exclude]
            )

        else:
            errors.raise_must_be_str_or_list("classes_exclude")

    else:
        exclude_slug = ""

    if isinstance(higher_class, str):
        raise_invalid_qid(higher_class)

        query = f"""SELECT DISTINCT ?item WHERE {{
        VALUES ?item {{ {formatted_qids} }}
        ?item wdt:P279* wd:{higher_class}.
        hint:Prior hint:gearing "forward".
        {exclude_slug}
        }}"""

    elif isinstance(higher_class, list):
        [raise_invalid_qid(c) for c in higher_class]
        classes_str = ", ".join(["wd:" + x for x in higher_class])

        query = f"""SELECT DISTINCT ?item WHERE {{
        VALUES ?item {{ {formatted_qids} }}
        ?item wdt:P279* ?tree.
        hint:Prior hint:gearing "forward".
        FILTER (?tree in ({classes_str}))
        {exclude_slug}
        }}"""

    else:
        errors.raise_must_be_str_or_list("higher_class")

    res = get_sparql_results(config.WIKIDATA_SPARQL_ENDPOINT, query)

    return [url_to_qid(i["item"]["value"]) for i in res["results"]["bindings"]]
