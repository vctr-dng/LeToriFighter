"""Write the dataset card (Markdown + card.json) of a collection run.

    uv run python scripts/make_dataset_card.py data/m5/v1 [--out docs/m5-dataset-card.md]

Exit status 1 if the integrity check fails (a non-train episode, a duplicate seed, or
a sidecar that disagrees with the LeRobot metadata).
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from letorifighter.data.card import write_card

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_dir", help="the run's out_dir")
    parser.add_argument("--out", help="card path (default <out_dir>/DATASET_CARD.md)")
    args = parser.parse_args(argv)
    path, card = write_card(args.out_dir, args.out)
    print(
        f"wrote {path}: {card['episodes']} episodes, integrity "
        f"{'PASS' if card['integrity']['ok'] else 'FAIL'}"
    )
    return 0 if card["integrity"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
