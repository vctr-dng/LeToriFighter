"""Train the M6 VLA. See docs/m6-vla.md and `letorifighter.vla.train`.

uv run --group vla python scripts/train_vla.py --dataset <root> --head ce
"""

from letorifighter.vla.train import main

if __name__ == "__main__":
    main()
