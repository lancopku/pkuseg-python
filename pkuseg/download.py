from __future__ import absolute_import, division, print_function, unicode_literals

import hashlib
import os
import re
import shutil
import sys
import tempfile
import zipfile

try:
    from requests.utils import urlparse
    from requests import get as urlopen
    requests_available = True
except ImportError:
    requests_available = False
    if sys.version_info[0] == 2:
        from urlparse import urlparse  # noqa f811
        from urllib2 import urlopen  # noqa f811
    else:
        from urllib.request import urlopen
        from urllib.parse import urlparse
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # defined below

HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

def download_model(url, model_dir, hash_prefix, progress=True):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        _download_url_to_file(url, cached_file, hash_prefix, progress=progress)
        unzip_file(cached_file, os.path.join(model_dir, filename.split('.')[0]))


def _download_url_to_file(url, dst, hash_prefix, progress):
    if requests_available:
        u = urlopen(url, stream=True, timeout=5)
        file_size = int(u.headers["Content-Length"])
        u = u.raw
    else:
        u = urlopen(url, timeout=5)
        meta = u.info()
        if hasattr(meta, 'getheaders'):
            file_size = int(meta.getheaders("Content-Length")[0])
        else:
            file_size = int(meta.get_all("Content-Length")[0])

    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


if tqdm is None:
    # fake tqdm if it's not installed
    class tqdm(object):

        def __init__(self, total, disable=False):
            self.total = total
            self.disable = disable
            self.n = 0

        def update(self, n):
            if self.disable:
                return

            self.n += n
            sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(self.total)))
            sys.stderr.flush()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.disable:
                return

            sys.stderr.write('\n')

def unzip_file(zip_name, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    file_zip = zipfile.ZipFile(zip_name, 'r')
    for file in file_zip.namelist():
        file_zip.extract(file, target_dir)
    file_zip.close()

if __name__ == '__main__':
    url = 'https://github.com/lancopku/pkuseg-python/releases/download/v0.0.14/mixed.zip'
    download_model(url, '.')
    
