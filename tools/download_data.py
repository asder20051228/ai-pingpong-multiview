"""Fetch the 17 official public validation videos used by the self-test."""
from pathlib import Path
import argparse
import json
import stat
import zipfile
from _download import download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'release/public-data.json')
    args = parser.parse_args()
    asset = json.loads(args.manifest.read_text(encoding='utf-8'))
    archive = download(asset, ROOT/'downloads')
    with zipfile.ZipFile(archive) as packed:
        for entry in packed.infolist():
            target = (ROOT/entry.filename).resolve()
            if not target.is_relative_to(ROOT/'participant') or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f'Unsafe data archive entry: {entry.filename}')
            if entry.is_dir():
                continue
            if not entry.filename.endswith('.mp4'):
                raise ValueError(f'Unexpected data entry: {entry.filename}')
        packed.extractall(ROOT)
    print('Public validation videos are ready (15 pingpang + 2 basketball).')


if __name__ == '__main__':
    main()
