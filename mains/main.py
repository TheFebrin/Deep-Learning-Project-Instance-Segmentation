from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import copy
import os.path as osp

import mmcv
from mmcv.runner import HOOKS, master_only
from mmcv.runner.hooks import LoggerHook
from mmcv import Config

import numpy as np
import comet_ml

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


@DATASETS.register_module()
class KittiTinyDataset(CustomDataset):
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        image_list = mmcv.list_from_file(self.ann_file)

        data_infos = []
        # convert annotations to middle format
        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}.jpeg'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]

            data_info = dict(filename=f'{image_id}.jpeg', width=width, height=height)

            # load annotations
            label_prefix = self.img_prefix.replace('image_2', 'label_2')
            lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt'))

            content = [line.strip().split(' ') for line in lines]
            bbox_names = [x[0] for x in content]
            bboxes = [[float(info) for info in x[4:8]] for x in content]

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos


@HOOKS.register_module()
class CometMLLoggerHook(LoggerHook):

    def __init__(self,
                 project_name=None,
                 hyper_params=None,
                 import_comet=False,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        """Class to log metrics to Comet ML.
        It requires `comet_ml` to be installed.
        Args:
            project_name (str, optional):
                Send your experiment to a specific project.
                Otherwise will be sent to Uncategorized Experiments.
                If project name does not already exists Comet.ml will create
                a new project.
            hyper_params (dict, optional): Logs a dictionary
                (or dictionary-like object) of multiple parameters.
            import_comet (bool optional): Whether to import comet_ml before run.
                WARNING: Comet ML have to be imported before sklearn and torch,
                or COMET_DISABLE_AUTO_LOGGING have to be set in the environment.
            interval (int): Logging interval (every k iterations).
            ignore_last (bool): Ignore the log of last iterations in each epoch
                if less than `interval`.
            reset_flag (bool): Whether to clear the output buffer after logging
            by_epoch (bool): Whether EpochBasedRunner is used.
        """
        super(CometMLLoggerHook, self).__init__(interval, ignore_last,
                                                reset_flag, by_epoch)
        if import_comet:
            self.import_comet()
        self.project_name = project_name
        self.hyper_params = hyper_params

    def import_comet(self):
        try:
            import comet_ml
        except ImportError:
            raise ImportError(
                'Please run "pip install comet_ml" to install Comet ML')
        self.comet_ml = comet_ml

    @master_only
    def before_run(self, runner):
        self.experiment = comet_ml.Experiment(
            api_key='ke6iV9Jj4ppTWxgnP4Oyy4CTh',
            project_name=self.project_name,
        )
        if self.hyper_params is not None:
            self.experiment.log_parameters(self.hyper_params)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            self.experiment.log_metric(
                name=tag,
                value=val,
                step=self.get_iter(runner),
                epoch=self.get_epoch(runner)
            )

    @master_only
    def after_run(self, runner):
        self.experiment.end()


if __name__ == '__main__':
    # Choose to use a config and initialize the detector
    config = 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    # initialize the detector
    model = init_detector(config, checkpoint, device='cuda:0')

    cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')
    # Modify dataset type and path
    cfg.dataset_type = 'KittiTinyDataset'
    cfg.data_root = 'kitti_tiny/'

    cfg.data.test.type = 'KittiTinyDataset'
    cfg.data.test.data_root = 'kitti_tiny/'
    cfg.data.test.ann_file = 'train.txt'
    cfg.data.test.img_prefix = 'training/image_2'

    cfg.data.train.type = 'KittiTinyDataset'
    cfg.data.train.data_root = 'kitti_tiny/'
    cfg.data.train.ann_file = 'train.txt'
    cfg.data.train.img_prefix = 'training/image_2'

    cfg.data.val.type = 'KittiTinyDataset'
    cfg.data.val.data_root = 'kitti_tiny/'
    cfg.data.val.ann_file = 'val.txt'
    cfg.data.val.img_prefix = 'training/image_2'

    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = 3
    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = './tutorial_exps'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 12
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 12

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    cfg.log_config = dict(
        interval=10,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='CometMLLoggerHook')
        ])

    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(
        cfg.model)
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)

    print('DONE ...')