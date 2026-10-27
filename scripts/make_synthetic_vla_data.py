"""Write the synthetic learnable LeRobot v3 dataset used to test the VLA pipeline.

uv run --group vla python scripts/make_synthetic_vla_data.py --out data/synthetic/both
"""

import argparse

from letorifighter.vla.synthetic import make_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="dataset root (must not exist)")
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--length", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=300)
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--instructions", type=int, default=2)
    parser.add_argument("--info", choices=["both", "image", "state"], default="both")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    description = make_synthetic_dataset(
        args.out,
        episodes=args.episodes,
        length=args.length,
        image_size=args.image_size,
        state_dim=args.state_dim,
        n_scenes=args.scenes,
        n_instructions=args.instructions,
        info=args.info,
        seed=args.seed,
        repo_id="local/synthetic",
    )
    print(
        f"wrote {args.episodes} episodes to {args.out}; task ids {description['task_ids']}"
    )


if __name__ == "__main__":
    main()
