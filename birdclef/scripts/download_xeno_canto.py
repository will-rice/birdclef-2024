"""Download animal sounds from Xeno-Canto."""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List

import ffmpeg
import pandas as pd
import requests
from tqdm import tqdm


def main() -> None:
    """Download script."""
    parser = argparse.ArgumentParser(description="Download script.")
    parser.add_argument("save_root", type=Path)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--maxtasksperchild", type=int, default=100)
    parser.add_argument("--processes", type=int, default=32)
    parser.add_argument("--num_pages", type=int, default=-1)
    parser.add_argument(
        "--taxonomy",
        type=str,
        default="birds",
        choices=["birds", "frogs", "grasshoppers", "bats"],
    )
    args = parser.parse_args()

    downloader = Downloader(
        args.save_root,
        parallel=args.parallel,
        maxtasksperchild=args.maxtasksperchild,
        processes=args.processes,
        num_pages=args.num_pages,
        taxonomy=args.taxonomy,
    )
    downloader.download()


class Downloader:
    """Xeno-Canto downloader."""

    def __init__(
        self,
        save_root: Path,
        parallel: bool = False,
        maxtasksperchild: int = 100,
        processes: int = 32,
        num_pages: int = 1,
        taxonomy: str = "birds",
    ):
        self._save_root = save_root
        self._save_root.mkdir(exist_ok=True, parents=True)
        self._parallel = parallel
        self._maxtasksperchild = maxtasksperchild
        self._processes = processes
        self._num_pages = num_pages
        self._metadata: List[Dict[str, Any]] = []
        self._endpoint = f"https://xeno-canto.org/api/2/recordings?query=grp:{taxonomy}"

    def download(self) -> None:
        """Download bird sounds from Xeno-Canto."""
        if self._parallel:
            self.download_parallel()
        else:
            self.download_sequential()

        metadata_df = pd.DataFrame.from_records(self._metadata)
        print(metadata_df.head())
        metadata_df.to_csv(self._save_root / "metadata.csv", index=False)

    def download_parallel(self) -> None:
        """Download bird sounds from Xeno-Canto."""
        response = requests.get(self._endpoint).json()
        if self._num_pages == -1:
            num_pages = response["numPages"]
        else:
            num_pages = self._num_pages
        pages = list(range(1, num_pages + 1))
        with Pool(
            maxtasksperchild=self._maxtasksperchild, processes=self._processes
        ) as pool:
            for i in tqdm(
                pool.imap_unordered(self.download_page, pages), total=len(pages)
            ):
                _ = i

    def download_sequential(self) -> None:
        """Download bird sounds from Xeno-Canto."""
        response = requests.get(self._endpoint).json()
        if self._num_pages == -1:
            num_pages = response["numPages"]
        else:
            num_pages = self._num_pages
        for page in tqdm(range(1, num_pages + 1)):
            try:
                self.download_page(page)
            except Exception as e:
                print(e)
                continue

    def download_page(self, page: int) -> None:
        """Download page."""
        response = requests.get(f"{self._endpoint}&page={page}").json()
        for recording in response.get("recordings", []):
            self.download_recording(recording)
        time.sleep(1)

    def download_recording(self, recording: Dict[str, Any]) -> None:
        """Download audio."""
        url = recording["file"]
        species = recording["sp"]
        recording_id = recording["id"]

        save_parent = self._save_root / species
        save_parent.mkdir(exist_ok=True, parents=True)
        save_path = save_parent / f"XC{recording_id}.flac"

        if not save_path.exists():
            try:
                ffmpeg.input(url).output(
                    str(save_path), acodec="flac", ac=1, ar=32000
                ).global_args("-n").global_args("-loglevel", "error").run(
                    capture_stdout=True, capture_stderr=True
                )
            except ffmpeg.Error:
                # print(error.stderr)
                pass
            else:
                time.sleep(1)

        self._metadata.append(recording)


if __name__ == "__main__":
    main()
