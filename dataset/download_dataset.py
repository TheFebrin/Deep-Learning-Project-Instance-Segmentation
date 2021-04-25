import sys
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor


DATASET_LINK = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
GROUNDTRUTH_LINK = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'


def download_images(directory_path='.'):
    print('Downloading images.')
    command = f'curl {DATASET_LINK} > {directory_path}/images.tar.gz'
    print(f'Executing: {command}')
    p = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = p.communicate()
    print(err)
    print('Finished downloading images!')

def download_annotations(directory_path='.'):
    print('Downloading annotations.')
    command = f'curl {GROUNDTRUTH_LINK} > {directory_path}/annotations.tar.gz'
    print(f'Executing: {command}')
    p = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = p.communicate()
    print(err)
    print('Finished downloading annotations!')

def download_dataset(directory_path='.'):
    with ThreadPoolExecutor(max_workers=2) as e:
        e.submit(download_annotations, directory_path)
        e.submit(download_images, directory_path)

    print('Untar ...')
    p = subprocess.Popen(
        f'tar xvzf {directory_path}/annotations.tar.gz',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = p.communicate()  # line needed to finish untar
    print(err)
    print('Untar annotations finished!')

    p = subprocess.Popen(
        f'tar xvzf {directory_path}/images.tar.gz',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = p.communicate()  # line needed to finish untar
    print(err)
    print('Untar images finished!')

    return 0

if __name__ == '__main__':
    sys.exit(download_dataset())