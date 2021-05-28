import json
import click
import pandas as pd


@click.command(
    help="""Make jsonl NER dictionary from events input files.
Inputs:
1. Master list of exhibitions (xlsx). Must contain columns 'Include in Events Dictionary' and 'Temporary Exhibiton Name cleaned'.
2. Cleaned list of exhibitions exported from Mimsy (xlsx). Must contain columns 'FULL_NAME' and 'KEEP'.
"""
)
@click.option(
    "--input",
    "-i",
    nargs=2,
    type=click.Path(exists=True),
    help="Paths of input files: exhibitions -1988; Mimsy exhibitions export. Separate by colon (unix) or semicolon (Windows).",
)
@click.option("--output", "-o", type=click.Path(), prompt="Output file path.")
def main(input, output):
    """
    Make jsonl NER dictionary from events input files.
    """

    exhibitions_path_1, exhibitions_path_2 = input
    museum_names = (
        "Science Museum",
        "Science and Media Museum",
        "Science and Industry Museum",
        "Museum of Science and Industry",
        "National Railway Museum",
        "Locomotion",
    )

    # 1. Process Exhibitions list to 1988
    exhibitions_df_1 = pd.read_excel(exhibitions_path_1, engine="openpyxl")
    for col_name in (
        "Include in Events Dictionary",
        "Temporary Exhibition Name cleaned",
    ):
        assert (
            col_name in exhibitions_df_1.columns
        ), f"column '{col_name}' not in file 1"

    exhibition_names_1 = (
        exhibitions_df_1.loc[
            exhibitions_df_1["Include in Events Dictionary"] == 1,
            "Temporary Exhibition Name cleaned",
        ]
        .unique()
        .tolist()
    )
    #  remove all one-word and lowercased names, remove leading/trailing whitespace
    exhibition_names_1 = [
        i.strip()
        for i in exhibition_names_1
        if (len(i.split(" ")) > 1) and (i.lower() != i)
    ]

    # 2. Process Mimsy exhibitions export
    exhibitions_df_2 = pd.read_excel(exhibitions_path_2, engine="openpyxl")
    for col_name in ("FULL_NAME", "KEEP"):
        assert (
            col_name in exhibitions_df_2.columns
        ), f"column '{col_name}' not in file 2"

    exhibition_names_2 = exhibitions_df_2.loc[
        exhibitions_df_2["KEEP"] == 1, "FULL_NAME"
    ]
    # Smith Centre is a private room so remove exhibitions there
    exhibition_names_2 = [i for i in exhibition_names_2 if "Smith Centre" not in i]
    #  remove all one-word and lowercased names, remove leading/trailing whitespace
    exhibition_names_2 = [
        i.strip()
        for i in exhibition_names_2
        if (len(i.split(" ")) > 1) and (i.lower() != i)
    ]
    # remove museum names ("Science Museum: An Exhibition" -> "An Exhibition")
    exhibition_names_2 = [
        i.replace(f"{museum_name}: ", "")
        for i in exhibition_names_2
        for museum_name in museum_names
        if i.startswith(f"{museum_name}: ")
    ]

    # 3. Join both to jsonl for spaCy DictionaryMatcher
    jsonl_list = []
    for name in set(exhibition_names_1 + exhibition_names_2):
        jsonl_list.append({"label": "EVENT", "pattern": name})

    # 4. Export
    with open(output, "w", encoding="utf-8") as f:
        for item in jsonl_list:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
