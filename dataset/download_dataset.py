import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor


DATASET_LINK = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
GROUNDTRUTH_LINK = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'


def download_images():
    print('Downloading images (may take couple of minutes) ...')
    p = subprocess.Popen(
        f'curl {DATASET_LINK} > images.tar.gz',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = p.communicate()
    print(out, err)
    print('Finished downloading images!')

def download_annotations():
    print('Downloading annotations ...')
    p = subprocess.Popen(
        f'curl {GROUNDTRUTH_LINK} > annotations.tar.gz',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = p.communicate()
    print(out, err)
    print('Finished downloading annotations!')


def download_dataset():
    with ThreadPoolExecutor(max_workers=2) as e:
        e.submit(download_annotations)
        e.submit(download_images)

    print('Untar ...')
    p = subprocess.Popen(
        f'tar xvzf annotations.tar.gz',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = p.communicate()  # line needed to finish untar
    print('Untar annotations finished!: ')

    p = subprocess.Popen(
        f'tar xvzf images.tar.gz',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = p.communicate()  # line needed to finish untar
    print('Untar images finished!: ')

    return 0

if __name__ == '__main__':
    sys.exit(download_dataset())