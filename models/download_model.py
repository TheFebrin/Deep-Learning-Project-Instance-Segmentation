import os
import sys
import pathlib
import subprocess
import requests

from concurrent.futures import ThreadPoolExecutor

MASK_RCNN50_URL = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"
MASK_RCNN50_PATH = "mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"

TRAINED_MASK_RCNN50_URL = 'https://drive.google.com/u/0/uc?id=1urB9z3oshjcVX1KsDwdZbzf5P8nkTh4G&export=download'
TRAINED_MASK_RCNN50_PATH = "trained_mask_rcnn_r50_caffe_fpn_mstrain-poly_3x.pth"


def download_model(model_url=MASK_RCNN50_URL, model_filename=MASK_RCNN50_PATH, directory_path='.'):
    print(f'Downloading model: {MASK_RCNN50_PATH}')
    p = subprocess.Popen(
        f'wget -c {model_url} -O {os.path.join(directory_path, model_filename)}',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = p.communicate()
    print(err)
    print('Finished downloading model!')
    return 0

def download_file_from_google_drive(URL, id, destination):
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_model_from_google_drive(directory_path='.'):
    print(f'Downloading model from google drive: {TRAINED_MASK_RCNN50_PATH}')
    file_id = '1urB9z3oshjcVX1KsDwdZbzf5P8nkTh4G'
    download_file_from_google_drive(
        URL=TRAINED_MASK_RCNN50_URL,
        id=file_id,
        destination=os.path.join(directory_path, TRAINED_MASK_RCNN50_PATH)
    )
    print('Finished downloading model from google drive!')
    return 0

def download_models(directory_path='.'):
    with ThreadPoolExecutor(max_workers=2) as e:
        e.submit(download_model, model_url=MASK_RCNN50_URL, model_filename=MASK_RCNN50_PATH, directory_path=directory_path)
        e.submit(download_model_from_google_drive, directory_path=directory_path)


if __name__ == '__main__':
    sys.exit(download_models())



