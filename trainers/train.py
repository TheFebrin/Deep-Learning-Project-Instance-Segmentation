import numpy as np
import torch, torchvision
import comet_ml
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmdet
from PIL import Image
import matplotlib.pyplot as plt
from mmcv import Config
from mmdet.apis import set_random_seed
import subprocess



_base_ = './configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

cfg = Config.fromfile(_base_)


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

cfg.model.roi_head.bbox_head.num_classes = 37
cfg.model.roi_head.mask_head.num_classes = 37

cfg.data.train.img_prefix = 'images/'
cfg.data.train.classes = classes
cfg.data.train.ann_file = 'pets-train-coco-format.json'

cfg.data.val.img_prefix = 'images/'
cfg.data.val.classes = classes
cfg.data.val.ann_file = 'pets-test-coco-format.json'

cfg.data.test.img_prefix = 'images/'
cfg.data.test.classes = classes
cfg.data.test.ann_file = 'pets-test-coco-format.json'

cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

cfg.log_config.interval = 25
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='CometMLLoggerHook', api_key='your_api_key')
]

cfg.custom_imports = dict(imports=['mmdet.core.utils.comet_logger_hook'],
                          allow_failed_imports=False)

cfg.dump('mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_pets.py')

training_command = "python tools/train.py mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_pets.py"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()