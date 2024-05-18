from yacs.config import CfgNode as CN

cfg = CN()
cfg.lr = 1e-4
cfg.lr_backbone = 1e-5
cfg.batch_size = 2
cfg.weight_decay = 1e-4
cfg.epochs = 300
cfg.lr_drop = 200
cfg.clip_max_norm = 0.1  # gradient clipping max norm

# Model parameters
cfg.frozen_weights = None # Path to the pretrained model. If set, only the mask head will be trained
# the mask head is for the panoptic segmentation

# * Backbone
cfg.backbone = 'resnet50' # Name of the convolutional backbone to use
cfg.dilation = False # If true, we replace stride with dilation in the last convolutional block (DC5)
cfg.position_embedding = 'sine' # 'learned' # choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features"
cfg.num_feature_levels = 4 # number of feature levels

# * Transformer
cfg.enc_layers = 6 # Number of encoding layers in the transformer
cfg.dec_layers = 6 # Number of decoding layers in the transformer
cfg.dim_feedforward = 2048 # Intermediate size of the feedforward layers in the transformer blocks
cfg.hidden_dim = 8 # 256 # Size of the embeddings (dimension of the transformer)
cfg.dropout = 0.1 # Dropout applied in the transformer
cfg.nheads = 8 # Number of attention heads inside the transformer's attentions
cfg.num_queries = 192*640 # 100 Number of query slots
cfg.pre_norm = False
cfg.dec_n_points = 4
cfg.enc_n_points = 4

# * Segmentation
cfg.masks = False # Train segmentation head if the flag is provided

# Loss
cfg.aux_loss = True # Disables auxiliary decoding losses (loss at each layer)

# * Matcher
cfg.set_cost_class = 1 # Class coefficient in the matching cost
cfg.set_cost_bbox = 5. # "L1 box coefficient in the matching cost
cfg.set_cost_giou = 2. # giou box coefficient in the matching cost

# * Loss coefficients
cfg.mask_loss_coef = 1.
cfg.dice_loss_coef = 1.
cfg.bbox_loss_coef = 5.
cfg.giou_loss_coef = 2.
cfg.eos_coef = 0.1 # Relative classification weight of the no-object class

# dataset parameters
cfg.dataset_file = 'coco'
cfg.coco_path = None
cfg.coco_panoptic_path = None
cfg.remove_difficult = False

cfg.output_dir = '' # path where to save, empty for no saving
cfg.device = 'cuda' # device to use for training / testing
cfg.seed = 42
cfg.resume = '' # resume from checkpoint
cfg.start_epoch = 0 # metavar='N', start epoch
cfg.eval = False
cfg.num_workers = 2

# distributed training parameters
cfg.world_size = 1 # number of distributed processes
cfg.dist_url = 'env://' # url used to set up distributed training
