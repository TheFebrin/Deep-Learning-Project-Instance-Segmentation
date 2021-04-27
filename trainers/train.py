import subprocess

training_command = "python tools/train.py mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_pets.py"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()