from torch.utils.data import DataLoader

from packnet_sfm.dataloader.pt_data_loader.specialdatasets import StandardDataset
from packnet_sfm.dataloader.definitions.labels_file import labels_cityscape_seg
import packnet_sfm.dataloader.pt_data_loader.mytransforms as tf
from packnet_sfm.utils.horovod import print0
from packnet_sfm.utils.logging import pcolor

# class CityscapesDataset:
def cityscapes_train(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers):
    """A loader that loads images and ground truth for segmentation from the
    cityscapes training set.
    """
    labels = labels_cityscape_seg.getlabels()
    num_classes = len(labels_cityscape_seg.gettrainid2label())

    transforms = [
        tf.RandomHorizontalFlip(),
        tf.CreateScaledImage(),
        tf.Resize((resize_height, resize_width)),
        tf.RandomRescale(1.5),
        tf.RandomCrop((crop_height, crop_width)),
        tf.ConvertSegmentation(),
        tf.CreateColoraug(new_element=True),
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0),
        tf.RemoveOriginals(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'cityscapes_train_seg'),
        tf.AddKeyValue('purposes', ('segmentation', 'domain')),
        tf.AddKeyValue('num_classes', num_classes)
    ]

    dataset_name = 'Cityscapes'

    dataset = StandardDataset(
        dataset=dataset_name,
        trainvaltest_split='train',
        video_mode='mono',
        stereo_mode='mono',
        labels_mode='fromid',
        disable_const_items=True,
        labels=labels,
        keys_to_load=('color', 'segmentation'),
        data_transforms=transforms,
        video_frames=(0,)
    )

    print0(pcolor(f"  - Can use {len(dataset)} images from the cityscapes train set for segmentation training", 'yellow'))

    # return loader
    return dataset

def cityscapes_validation(resize_height, resize_width, batch_size, num_workers):
    """A loader that loads images and ground truth for segmentation from the
    cityscapes validation set
    """

    labels = labels_cityscape_seg.getlabels()
    num_classes = len(labels_cityscape_seg.gettrainid2label())

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize((resize_height, resize_width), image_types=('color', )),
        tf.ConvertSegmentation(),
        tf.CreateColoraug(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'cityscapes_val_seg'),
        tf.AddKeyValue('purposes', ('segmentation', )),
        tf.AddKeyValue('num_classes', num_classes)
    ]

    dataset = StandardDataset(
        dataset='Cityscapes',
        trainvaltest_split='validation',
        video_mode='mono',
        stereo_mode='mono',
        labels_mode='fromid',
        labels=labels,
        keys_to_load=['color', 'segmentation'],
        data_transforms=transforms,
        disable_const_items=True
    )

    print0(pcolor(f"  - Can use {len(dataset)} images from the cityscapes validation set for segmentation validation", 'yellow'))

    return dataset