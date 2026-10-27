"""Download the original UCI archive and extract only the documented files."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip'


def main():
    destination = ROOT / 'data' / 'raw'
    destination.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=60) as response:
        payload = response.read()
    archive = ZipFile(BytesIO(payload))
    records = {}
    for filename in ['hour.csv', 'day.csv', 'Readme.txt']:
        content = archive.read(filename)
        (destination / filename).write_bytes(content)
        records[filename] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    manifest = {'url': URL, 'doi': '10.24432/C5W894', 'license': 'CC BY 4.0',
                'archive_sha256': hashlib.sha256(payload).hexdigest(), 'files': records}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
