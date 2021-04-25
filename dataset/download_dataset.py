import sys
import subprocess

DATASET_LINK = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
GROUNDTRUTH_LINK = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'

def download_dataset():
    download_dataset_result = subprocess.run([
        "curl", DATASET_LINK, " > ", "images.tar.gz"], capture_output=True)
    download_groundtruth_result = subprocess.run([
        "curl", GROUNDTRUTH_LINK, " > ", "annotations.tar.gz"], capture_output=True)
    return 0

if __name__ == '__main__':
    sys.exit(download_dataset())