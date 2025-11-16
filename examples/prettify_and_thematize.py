"""
Example script for running the prettify + thematize steps on an episode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mywhisper.podcasts import PodcastCatalog
from mywhisper.prettify import PrettifyConfig, TranscriptPrettifier
from mywhisper.thematize import EpisodeThematizer, ThematizeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate readable transcripts and themes.")
    parser.add_argument("episode_id", help="Episode id/guid recorded in the catalog.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override data directory (defaults to MYWHISPER_DATA_ROOT or ./data).",
    )
    parser.add_argument(
        "--skip-prettify",
        action="store_true",
        help="Assume the readable transcript already exists.",
    )
    parser.add_argument(
        "--skip-thematize",
        action="store_true",
        help="Skip thematization and only run prettify.",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3",
        help="Override the Ollama model used for thematization prompts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve() if args.data_root else None
    catalog = PodcastCatalog(data_root=data_root)
    episode = catalog.get_episode(args.episode_id)
    if not episode:
        raise SystemExit(f"Episode {args.episode_id} not found in catalog.")

    prettify_config = PrettifyConfig(data_root=data_root) if data_root else PrettifyConfig()
    thematize_kwargs = {"llm_model": args.llm_model}
    if data_root:
        thematize_kwargs["data_root"] = data_root
    thematize_config = ThematizeConfig(**thematize_kwargs)

    readable_path = prettify_config.readable_path(episode)
    if not args.skip_prettify:
        prettifier = TranscriptPrettifier(episode, config=prettify_config, catalog=catalog)
        readable_path = prettifier.prettify()
        print(f"Readable transcript written to {readable_path}")
    elif not readable_path.exists():
        raise SystemExit(f"Readable transcript not found at {readable_path}; remove --skip-prettify.")

    if args.skip_thematize:
        return

    thematizer = EpisodeThematizer(
        podcast=episode,
        config=thematize_config,
        catalog=catalog,
    )
    themes_path = thematizer.thematize(readable_path=readable_path)
    print(f"Thematic summary written to {themes_path}")


if __name__ == "__main__":
    main()

