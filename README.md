# SENSE: Self-Evolving Learning for Self-Supervised Monocular Depth Estimation

[Datasets](#datasets) // [Training](#training) // [Evaluation](#evaluation) // [Acknowledgements](#acknowledgements) // [Reference](#reference)

## Datasets

Please refer to [packnet-sfm
](https://github.com/TRI-ML/packnet-sfm) for downloading the KITTI dataset and put it in `data/datasets/<dataset-name>` (can be a symbolic link).

## Training

### Step 1

Specify the resolution (192x640 or 320x1024):
```
python scripts/train.py configs/train_kitti_[resolution]_stage_1.yaml
```

After training, you can obtain pseudo labels and confidence maps based on GMM:
```
python scripts/eval.py --checkpoint [checkpoint_path.ckpt] --config configs/eval_kitti_[resolution]_GMM.yaml
```
Config parameters in `eval_kitti_[resolution]_GMM.yaml`:

`save.folder`: specify the saving path

### Step 2

Specify the `dataset.train.pseudo_label_path` in `train_kitti_[resolution]_stage_2.yaml` and train the model:
```
python scripts/train.py configs/train_kitti_[resolution]_stage_2.yaml
```

### Step 3

Specify the `model.checkpoint_path` in `train_kitti_[resolution]_stage_3.yaml` and train the model:
```
python scripts/train.py configs/train_kitti_[resolution]_stage_3.yaml
```

### Step 4

Specify the `model.checkpoint_path` and the `dataset.train.pseudo_label_path` in `train_kitti_[resolution]_stage_4.yaml` and train the model:
```
python scripts/train.py configs/train_kitti_[resolution]_stage_4.yaml
```

## Evaluation

Specify the `datasets.augmentation.image_shape` in `eval_kitti.yaml` and eval the model:
```
python scripts/eval.py --checkpoint [checkpoint_path.ckpt] --config configs/eval_kitti.yaml
```

## Acknowledgements

This code borrow heavily from [packnet-sfm
](https://github.com/TRI-ML/packnet-sfm) and we thank the authors for sharing their code and models.

## Reference
```
@article{li2023sense,
  title={SENSE: Self-Evolving Learning for Self-Supervised Monocular Depth Estimation},
  author={Li, Guanbin and Huang, Ricong and Li, Haofeng and You, Zunzhi and Chen, Weikai},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```