def main():
    import torch, torchvision
    print('Check Pytorch installation:')
    print(torch.__version__, torch.cuda.is_available())

    from mmcv.ops import get_compiling_cuda_version, get_compiler_version
    print('Check mmcv installation:')
    print(get_compiling_cuda_version())
    print(get_compiler_version())

    import mmdet
    print('Check MMDetection installation:')
    print(mmdet.__version__)


if __name__ == '__main__':
    main()
