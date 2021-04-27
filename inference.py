import os
import sys
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import numpy as np


def get_score_per_class(resuls):
    classes = (
        'Abyssinian',
        'american_bulldog',
        'american_pit_bull_terrier',
        'basset_hound',
        'beagle',
        'Bengal',
        'Birman',
        'Bombay',
        'boxer',
        'British_Shorthair',
        'chihuahua',
        'Egyptian_Mau',
        'english_cocker_spaniel',
        'english_setter',
        'german_shorthaired',
        'great_pyrenees',
        'havanese',
        'japanese_chin',
        'keeshond',
        'leonberger',
        'Maine_Coon',
        'miniature_pinscher',
        'newfoundland',
        'Persian',
        'pomeranian',
        'pug',
        'Ragdoll',
        'Russian_Blue',
        'saint_bernard',
        'samoyed',
        'scottish_terrier',
        'shiba_inu',
        'Siamese',
        'Sphynx',
        'staffordshire_bull_terrier',
        'wheaten_terrier',
        'yorkshire_terrier',
    )
    pairs = [(c, e[0][4]) for c, e in zip(classes, resuls[0]) if len(e)]
    pairs.sort(key=lambda e: e[1], reverse=True)
    return pairs


def main():
    if len(sys.argv) != 2:
        return 0
    img_path = sys.argv[1]
    model = init_detector(
        'Deep-Learning-Project-Instance-Segmentation/configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_pets.py',
        'Deep-Learning-Project-Instance-Segmentation/models/trained_mask_rcnn_r50_caffe_fpn_mstrain-poly_3x.pth',
        device='cuda:0')

    result = inference_detector(model, img_path)
    show_result_pyplot(model, img_path, result)

    scores_per_class = get_score_per_class(result)[:5]
    for cls, score in scores_per_class:
        print(f'{cls}: {np.round(score, 4)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
