"""Download, verify and join the complete Docker image from GitHub Releases."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
from _download import checked_name, download, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'release/docker-image.json')
    parser.add_argument('--output', type=Path, default=ROOT/'downloads')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--load', action='store_true', help='Load the verified archive into Docker')
    parser.add_argument('--keep-parts', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error('--jobs must be between 1 and 8')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = out/checked_name(manifest['archive_name'])
    if archive.exists():
        if archive.stat().st_size != manifest['archive_size'] or sha256(archive) != manifest['archive_sha256']:
            raise RuntimeError(f'Existing archive failed checksum; choose another output directory: {archive}')
    else:
        cache = out/'image-parts'
        cached = sum((cache/checked_name(p['name'])).stat().st_size for p in manifest['parts']
                     if (cache/checked_name(p['name'])).exists())
        needed = 2*manifest['archive_size']-cached+256*1024**2
        if shutil.disk_usage(out).free < needed:
            raise RuntimeError(f'Need about {needed/1024**3:.1f} GiB of free space in {out}')
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            paths = list(pool.map(lambda asset: download(asset, cache), manifest['parts']))
        assembled = archive.with_suffix(archive.suffix+'.assembling')
        digest = hashlib.sha256()
        written = 0
        with assembled.open('wb', buffering=4*1024**2) as result:
            for path in paths:
                with path.open('rb', buffering=4*1024**2) as source:
                    while block := source.read(4*1024**2):
                        result.write(block)
                        digest.update(block)
                        written += len(block)
        if written != manifest['archive_size'] or digest.hexdigest() != manifest['archive_sha256']:
            raise RuntimeError(f'Combined archive failed checksum: {assembled}')
        assembled.replace(archive)
        if not args.keep_parts:
            for path in paths:
                path.unlink()
    print(f'Complete verified image: {archive}', flush=True)
    if args.load:
        subprocess.run(['docker','load','-i',str(archive)], check=True)


if __name__ == '__main__':
    main()
