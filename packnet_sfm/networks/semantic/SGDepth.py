import torch
import torch.nn as nn

from . import networks


class SGDepthCommon(nn.Module):
    def __init__(self, layers, split_pos, pretrained=False):
        super().__init__()

        self.encoder = networks.ResnetEncoder(layers, pretrained)
        self.layers = layers  # This information is needed in the train loop for the sequential training

        # Number of channels for the skip connections and internal connections
        # of the decoder network, ordered from input to output
        self.shape_enc = tuple(reversed(self.encoder.num_ch_enc))
        self.shape_dec = (256, 128, 64, 32, 16)

        self.decoder = networks.PartialDecoder.gen_head(self.shape_dec, self.shape_enc, split_pos)

    def forward(self, x):
        # The encoder produces outputs in the order
        # (highest res, second highest res, …, lowest res)
        x = self.encoder(x)

        # The decoder expects it's inputs in the order they are
        # used. E.g. (lowest res, second lowest res, …, highest res)
        x = tuple(reversed(x))

        # Replace some elements in the x tuple by decoded
        # tensors and leave others as-is
        x = self.decoder(*x) # CHANGE ME BACK TO THIS

        return x

    #def get_last_shared_layer(self):
    #    return self.encoder.encoder.layer4


class SGDepthSeg(nn.Module):
    def __init__(self, common):
        super().__init__()

        self.decoder = networks.PartialDecoder.gen_tail(common.decoder)
        self.multires = networks.MultiResSegmentation(self.decoder.chs_x()[-1:])
        self.nl = nn.Softmax2d()

    def forward(self, *x):
        x = self.decoder(*x)
        semantics_feature = x
        x = self.multires(*x[-1:])
        x_lin = x[-1]

        return x_lin, semantics_feature


class SGDepth(nn.Module):
    KEY_FRAME_CUR = ('color_aug', 0, 0)

    def __init__(self, split_pos=1, layers=50, weights_init='pretrained', resolutions_depth=1, num_layers_pose=18):

        super().__init__()

        # sgdepth allowed for five possible split positions.
        # The PartialDecoder developed as part of sgdepth
        # is a bit more flexible and allows splits to be
        # placed in between sgdepths splits.
        # As this class is meant to maximize compatibility
        # with sgdepth the line below translates between
        # the split position definitions.
        split_pos = max((2 * split_pos) - 1, 0)

        # The Depth and the Segmentation Network have a common (=shared)
        # Encoder ("Feature Extractor")
        self.common = SGDepthCommon(
            layers, split_pos, weights_init == 'pretrained'
        )

        # While Depth and Seg Network have a shared Encoder,
        # each one has it's own Decoder
        self.seg = SGDepthSeg(self.common)

    def forward(self, batch):
        x = batch[self.KEY_FRAME_CUR]

        # Feed the stitched-together input tensor through
        # the common network part and generate two output
        # tuples that look exactly the same in the forward
        # pass, but scale gradients differently in the backward pass.
        x_seg = self.common(x)

        outputs = list(dict() for _ in range(1))

        # All the way back in the loaders each dataset is assigned one or more 'purposes'.
        # For datasets with the 'depth' purpose set the outputs[DATASET_IDX] dict will be
        # populated with depth outputs.
        # Datasets with the 'segmentation' purpose are handled accordingly.
        # If the pose outputs are populated is dependant upon the presence of
        # ('color_aug', -1, 0)/('color_aug', 1, 0) keys in the Dataset.
        x = x_seg
        encoder_features = x
        x, decoder_features = self.seg(*x)

        outputs[0]['segmentation_logits', 0] = x

        return tuple(outputs), decoder_features
