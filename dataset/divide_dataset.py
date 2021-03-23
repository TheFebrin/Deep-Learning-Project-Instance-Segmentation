"""
Sample usage:
python divide_dataset.py --datapoints-path=../images --labels-path=../annotations/trimaps \
    --datapoints-extention=.jpg --labels-extention=.png --valid=True --train-ratio=0.7

Divides the dataset into:
    * train
    * test
    * valid

IMPORTANT: When reading the file names applies a filter
"""


import os
import sys
import argparse
import shutil
from typing import *
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(
        description='Divide the dataset into train, test and valid.'
    )
    parser.add_argument(
        '--datapoints-path', required=True, type=str, help='Datapoints path.'
    )
    parser.add_argument(
        '--labels-path', required=True, type=str, help='Labels path.'
    )
    parser.add_argument(
        '--datapoints-extention', required=True, type=str, help='Datapoints extention I.e. .jpg .png ...'
    )
    parser.add_argument(
        '--labels-extention', required=True, type=str, help='Labels extention I.e. .jpg .png ...'
    )
    parser.add_argument(
        '--valid', required=False, type=bool, default=False, help='Splits test into test and valid.'
    )
    parser.add_argument(
        '--train-ratio', required=False, type=float, default=0.7, help='Ratio of training data and the rest.'
    )
    return parser.parse_args()

def delete_existing_folders(dir_path: str):
    try:
        shutil.rmtree(f'{dir_path}/test')
        shutil.rmtree(f'{dir_path}/train')
        shutil.rmtree(f'{dir_path}/valid')
    except:
        print('Folders don\'t exist.')

def create_fresh_folders(dir_path: str):
    os.mkdir(f'{dir_path}/test')
    os.mkdir(f'{dir_path}/test/datapoints')
    os.mkdir(f'{dir_path}/test/labels')

    os.mkdir(f'{dir_path}/train')
    os.mkdir(f'{dir_path}/train/datapoints')
    os.mkdir(f'{dir_path}/train/labels')

    os.mkdir(f'{dir_path}/valid')
    os.mkdir(f'{dir_path}/valid/datapoints')
    os.mkdir(f'{dir_path}/valid/labels')


def copy_data_to_folders(
    X_train: List[str],
    X_test: List[str],
    X_valid: List[str],
    y_train: List[str],
    y_test: List[str],
    y_valid: List[str],
    datapoints_path: str,
    labels_path: str
):
    destination_path_to_datapoints: Mapping[str, List[str]] = {
        'train/datapoints/': X_train,
        'test/datapoints/': X_test,
        'valid/datapoints/': X_valid,
    }

    destination_path_to_labels: Mapping[str, List[str]] = {
        'train/labels/': y_train,
        'test/labels/': y_test,
        'valid/labels/': y_valid
    }

    # TODO(Dawid): Parallelize it
    for dst_path, datapoints in destination_path_to_datapoints.items():
        for d in datapoints:
            shutil.copyfile(f'{datapoints_path}/{d}', f'{dst_path}/{d}')

    for dst_path, labels in destination_path_to_labels.items():
        for l in labels:
            shutil.copyfile(f'{labels_path}/{l}', f'{dst_path}/{l}')

def divide_dataset_into_folders(
    datapoints_path: str,
    labels_path: str,
    datapoints_extention: str,
    labels_extention: str,
    split_to_valid: bool,
    train_ratio: float,
    datapoints_filter: Callable[[str], bool]=lambda x: x,  # function used to filter datapoints
    labels_filter: Callable[[str], bool]=lambda x: x,  # function used to filter labels
):
    try:
        datapoints_names: List[str] = os.listdir(datapoints_path)
    except:
        raise ValueError(f'Wrong datapoints path: {datapoints_path}')

    try:
        labels_names: List[str] = os.listdir(labels_path)
    except:
        raise ValueError(f'Wrong labels path: {labels_path}')


    datapoints_names = list(sorted(filter(datapoints_filter, datapoints_names)))
    labels_names = list(sorted(filter(labels_filter, labels_names)))

    error_message = f'Number of datapoints has to be equal to the number' \
        f'of labels: {len(datapoints_names)} != {len(labels_names)}'
    assert len(datapoints_names) == len(labels_names), error_message

    assert all([x.endswith(datapoints_extention) for x in datapoints_names])
    assert all([x.endswith(labels_extention) for x in labels_names])
    assert all([x.rstrip(datapoints_extention) == y.rstrip(labels_extention) \
        for x, y in zip(datapoints_names, labels_names)])

    X_train, X_test, y_train, y_test = train_test_split(
        datapoints_names, labels_names, test_size=(1 - train_ratio), random_state=420
    )

    X_valid, y_valid = [], []
    if split_to_valid:
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=69
        )

    copy_data_to_folders(
        X_train=X_train,
        X_test=X_test,
        X_valid=X_valid,
        y_train=y_train,
        y_test=y_test,
        y_valid=y_valid,
        datapoints_path=datapoints_path,
        labels_path=labels_path
    )

    print('Division into folders successful!')
    print('Sizes:')
    print(f'Train={len(X_train)} | Test={len(X_test)} | Valid={len(X_valid)}')

def main():
    args = parse_args()
    datapoints_path = args.datapoints_path
    labels_path = args.labels_path
    datapoints_extention = args.datapoints_extention
    labels_extention = args.labels_extention
    split_to_valid = args.valid
    train_ratio = args.train_ratio

    print(f'Datapoints path: {datapoints_path}')
    print(f'Labels path:     {labels_path}')

    # dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.getcwd()
    delete_existing_folders(dir_path=dir_path)
    create_fresh_folders(dir_path=dir_path)

    divide_dataset_into_folders(
        datapoints_path=datapoints_path,
        labels_path=labels_path,
        datapoints_extention=datapoints_extention,
        labels_extention=labels_extention,
        split_to_valid=split_to_valid,
        train_ratio=train_ratio,
        datapoints_filter=lambda x: not x.endswith('.mat'),
        labels_filter=lambda x: not x.startswith('.'),
    )
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
