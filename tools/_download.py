"""Download public release files with resumable curl and SHA256 checks."""
from pathlib import Path
import hashlib
import shutil
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb', buffering=4*1024**2) as stream:
        while block := stream.read(4*1024**2):
            digest.update(block)
    return digest.hexdigest()


def checked_name(name):
    if Path(name).name != name or '/' in name or '\\' in name or name in ('.','..'):
        raise ValueError(f'Unsafe asset filename: {name!r}')
    return name


def download(asset, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/checked_name(asset['name'])
    if target.exists():
        if target.stat().st_size == asset['size'] and sha256(target) == asset['sha256']:
            print(f'Already verified: {target.name}', flush=True)
            return target
        raise RuntimeError(f'Existing file failed checksum; move it aside before retrying: {target}')
    curl = shutil.which('curl.exe') or shutil.which('curl')
    if not curl:
        raise RuntimeError('curl is required (included in Windows 10/11; install curl on Linux).')
    partial = target.with_suffix(target.suffix+'.download')
    subprocess.run([curl, '--disable', '--fail', '--location', '--silent', '--show-error',
                    '--retry', '4', '--retry-delay', '2', '--connect-timeout', '30',
                    '--continue-at', '-', '--output', str(partial), asset['url']], check=True)
    if partial.stat().st_size != asset['size'] or sha256(partial) != asset['sha256']:
        raise RuntimeError(f'Download checksum failed: {partial}')
    partial.replace(target)
    print(f'Downloaded and verified: {target.name}', flush=True)
    return target
