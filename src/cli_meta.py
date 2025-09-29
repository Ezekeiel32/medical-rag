import os
import sys
import click
from rich import print
from .ocr_structuring import build_ocr_documents


@click.command()
@click.option("--results", "results_path", type=click.Path(dir_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out", "results.json")))
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
def main(results_path: str, out_dir: str) -> None:
    if not os.path.isfile(results_path):
        print(f"[red]results.json not found:[/red] {results_path}")
        sys.exit(1)
    jsonl_path, index_path = build_ocr_documents(results_path, out_dir)
    print(f"Structured documents written:\n- {jsonl_path}\n- {index_path}")


if __name__ == "__main__":
    main()


