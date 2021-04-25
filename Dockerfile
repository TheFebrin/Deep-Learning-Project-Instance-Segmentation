FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

# installing needed packages for pyhton
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3-pip python3-dev build-essential && \
    apt-get install -y git && \
    apt-get install -y wget && \
    apt-get install -y curl

# set as a workdir
WORKDIR .

# copy the whole repo
COPY . .

# installing our python script requirements
RUN pip3 install -r requirements.txt

RUN pip3 install comet_ml --upgrade --quiet

RUN pip3 install -U torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

## Install MMCV
RUN pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html

# Install MMDetection
RUN rm -rf mmdetection
RUN git clone https://github.com/open-mmlab/mmdetection.git
ENV FORCE_CUDA="1"
RUN cd mmdetection && pip3 install -e .

# Perform inference with a MMDet detector
RUN mkdir checkpoints
RUN wget -c https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth \
      -O checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth


# download dataset and models
CMD ["python3", "download.py"]

# check versions
CMD ["python3", "mains/check_versions.py"]

# executing the script
CMD ["python3", "mains/main.py"]