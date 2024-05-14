# Guided Depth Super-resolution with Continuity-constrained Pixelwise Gradient Deformation

## Dependencies

```
python=3.9
pytorch=1.10
torchvision=0.11
cudatoolkit=11.3
segmentation-models-pytorch==0.2
scikit-image
h5py
tqdm
```

## Dataset

1. [NYU](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
2. [Lu & Middlebury](http://web.cecs.pdx.edu/~fliu/project/depth-enhance/)
3. [RGB-D-D](https://github.com/lingzhi96/RGB-D-D-Dataset)

```
test_data
└───NYU
│   └───gt
│   │   └───1001.npy
│   │   |   1002.npy
│   │   │   ...
│   │   │   1449.npy
│   └───rgb
│   │   └───1001.npy
│   │   |   1002.npy
│   │   │   ...
│   │   │   1449.npy
└───Lu
│   └───gt
│   └───rgb
└───Middlebury
│   └───gt
│   └───rgb
└───RGBDD
│   └───gt
│   └───rgb
```

## Pretrained Models

Our pretrained model checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1cE9-gkcHeiaZ7DkiIh42BoprYFrhMKhL).

## Training

```
python train.py --scale=8
```



## Testing

```
python test.py --scale=8
```

