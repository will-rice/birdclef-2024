"""Filter bad audio files from the dataset."""

import argparse
import multiprocessing
from pathlib import Path

import torchaudio
from tqdm import tqdm


def main() -> None:
    """Filter bad audio files from the dataset."""
    parser = argparse.ArgumentParser(
        description="Filter bad audio files from the dataset."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--processes", default=12, type=int)
    parser.add_argument("--maxtasksperchild", default=100, type=int)
    parser.add_argument("--extension", default="flac", type=str)
    args = parser.parse_args()

    paths = list(args.root.glob(f"**/*.{args.extension}"))
    with multiprocessing.Pool(
        processes=args.processes, maxtasksperchild=args.maxtasksperchild
    ) as pool:
        for _ in tqdm(pool.imap_unordered(filter_bad_audio, paths), total=len(paths)):
            pass


def filter_bad_audio(path: Path) -> None:
    """Filter bad audio files from the dataset."""
    try:
        torchaudio.load(str(path))
    except Exception as error:
        print(error)
        path.unlink()


if __name__ == "__main__":
    main()
