import subprocess

training_command = "!python tools/test.py mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_pets.py work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_pets/latest.pth --eval bbox segm"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()