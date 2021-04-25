import sys
import subprocess

MASK_RCNN50_URL = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"
MASK_RCNN50_FILENAME = "mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"

def download_model():
    print('Downloading model ...')
    p = subprocess.Popen(
        f'wget -c {MASK_RCNN50_URL} -O {MASK_RCNN50_FILENAME}',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = p.communicate()
    print(out, err)
    print('Finished downloading model!')

    return 0


if __name__ == '__main__':
    sys.exit(download_model())