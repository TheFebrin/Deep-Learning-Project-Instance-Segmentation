import sys
from concurrent.futures import ThreadPoolExecutor

from models.download_model import download_models
from dataset.download_dataset import download_dataset


def main():
    print('== START DOWNLOADING REQUIRED FILES [will download about 2GB of data] ==')
    with ThreadPoolExecutor(max_workers=2) as e:
        e.submit(download_models)
        e.submit(download_dataset)
    return 0


if __name__ == '__main__':
    sys.exit(main())