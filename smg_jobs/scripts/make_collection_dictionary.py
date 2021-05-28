import sys

sys.path.append("..")

import click
from heritageconnector.nlp_export import labels_ids_to_jsonl


@click.command(
    help="""Make jsonl collection dictionary from Heritage Connector graph."""
)
@click.option("--output", "-o", type=click.Path(), prompt="Output file path.")
def main(output):
    labels_ids_to_jsonl(
        jsonl_path=output,
        topconcepts_to_ignore={"DOCUMENT"},
    )


if __name__ == "__main__":
    main()
