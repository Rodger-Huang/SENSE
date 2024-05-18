import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import filters
import cv2
from sklearn.mixture import GaussianMixture

from packnet_sfm.utils.image import match_scales, match_feature_scales
from packnet_sfm.geometry.camera import Camera
from packnet_sfm.geometry.camera_utils import view_synthesis
from packnet_sfm.utils.depth import calc_smoothness, inv2depth, viz_inv_depth
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling
from packnet_sfm.utils.util import get_colormap

########################################################################################################################

def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim

########################################################################################################################

class MultiViewPhotometricLoss(LossBase):
    """
    Self-Supervised multiview photometric loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them

    Parameters
    ----------
    num_scales : int
        Number of inverse depth map scales to consider
    ssim_loss_weight : float
        Weight for the SSIM loss
    occ_reg_weight : float
        Weight for the occlusion regularization loss
    smooth_loss_weight : float
        Weight for the smoothness loss
    C1, C2 : float
        SSIM parameters
    photometric_reduce_op : str
        Method to reduce the photometric loss
    disp_norm : bool
        True if inverse depth is normalized for
    clip_loss : float
        Threshold for photometric loss clipping
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    padding_mode : str
        Padding mode for view synthesis
    automask_loss : bool
        True if automasking is enabled for the photometric loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, num_scales=4, ssim_loss_weight=0.85, occ_reg_weight=0.1, smooth_loss_weight=0.1, 
                 slic_loss_weight=0.1, pseudo_loss_weight=1e-3,
                 C1=1e-4, C2=9e-4, photometric_reduce_op='mean', disp_norm=True, clip_loss=0.5,
                 progressive_scaling=0.0, padding_mode='zeros',
                 automask_loss=False, stage=1, **kwargs):
        super().__init__()
        self.n = num_scales
        self.progressive_scaling = progressive_scaling
        self.ssim_loss_weight = ssim_loss_weight
        self.occ_reg_weight = occ_reg_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.slic_loss_weight = slic_loss_weight
        self.pseudo_loss_weight = pseudo_loss_weight
        self.C1 = C1
        self.C2 = C2
        self.photometric_reduce_op = photometric_reduce_op
        self.disp_norm = disp_norm
        self.clip_loss = clip_loss
        self.padding_mode = padding_mode
        self.automask_loss = automask_loss
        self.progressive_scaling = ProgressiveScaling(
            progressive_scaling, self.n)
        self.stage = stage

        # fit a two-component GMM to the loss
        self.gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

        # labels for adversarial training
        self.source_label = 0
        self.target_label = 1

        # Asserts
        if self.automask_loss:
            assert self.photometric_reduce_op == 'min', \
                'For automasking only the min photometric_reduce_op is supported.'
                
########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'num_scales': self.n,
        }

########################################################################################################################

    def warp_ref_image(self, inv_depths, ref_image, K, ref_K, pose, n=3):
        """
        Warps a reference image to produce a reconstruction of the original one.

        Parameters
        ----------
        inv_depths : torch.Tensor 4*[B,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        """
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        # Generate cameras for all scales
        cams, ref_cams = [], []
        for i in range(n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(Camera(K=K.float()).scaled(scale_factor).to(device))
            ref_cams.append(Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device))
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(n)]
        ref_images = match_scales(ref_image, inv_depths, n)

        ref_warped = []
        for i in range(n):
            ref_warped_list = view_synthesis(
                ref_images[i], depths[i], ref_cams[i], cams[i],
                padding_mode=self.padding_mode)
            ref_warped.append(ref_warped_list)

        # Return warped reference image
        return ref_warped

    def warp_feature_image(self, inv_depths, ref_image, K, ref_K, pose):
        """
        Warps a reference image to produce a reconstruction of the original one.

        Parameters
        ----------
        inv_depths : torch.Tensor 4*[B,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        """
        inv_depths = [inv_depths] * self.n
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        # Generate cameras for all scales
        cams, ref_cams = [], []
        for i in range(self.n):
            # NOTE 这里需要保证inv_depths大小和输入图像大小一致如(640, 192)
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = float(W) / DW
            cams.append(Camera(K=K.float()).scaled(scale_factor).to(device))
            ref_cams.append(Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device))
        # View synthesis
        inv_depths = match_feature_scales(inv_depths, ref_image, self.n, align_corners=False)
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = [ref_image] * self.n

        ref_warped = []
        # occ_mask = []
        for i in range(self.n):
            # ref_warped_list, occ_mask_list = view_synthesis(
            ref_warped_list = view_synthesis(
                ref_images[i], depths[i], ref_cams[i], cams[i],
                padding_mode=self.padding_mode)
            ref_warped.append(ref_warped_list)

        return ref_warped

########################################################################################################################

    def SSIM(self, x, y, kernel_size=3):
        """
        Calculates the SSIM (Structural SIMilarity) loss

        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        kernel_size : int
            Convolutional parameter

        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM loss
        """
        ssim_value = SSIM(x, y, C1=self.C1, C2=self.C2, kernel_size=kernel_size)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def calc_photometric_loss(self, t_est, images, n=3):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Photometric loss
        """
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i]) for i in range(n)]
        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.SSIM(t_est[i], images[i], kernel_size=3)
                         for i in range(n)]             
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                                (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                                for i in range(n)]
        else:
            photometric_loss = l1_loss
        # Clip loss
        if self.clip_loss > 0.0:
            for i in range(n):
                mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                photometric_loss[i] = torch.clamp(
                    photometric_loss[i], max=float(mean + self.clip_loss * std))
        # Return total photometric loss
        return photometric_loss

    def reduce_photometric_loss(self, photometric_losses, pseudo_mask=None):
        """
        Combine the photometric loss from all context images

        Parameters
        ----------
        photometric_losses : list of torch.Tensor [B,3,H,W]
            Pixel-wise photometric losses from the entire context

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Reduced photometric loss
        """
        # Reduce function
        def reduce_function(losses, pseudo_mask=None):
            if self.photometric_reduce_op == 'mean':
                return sum([l.mean() for l in losses]) / len(losses)
            elif self.photometric_reduce_op == 'min':
                if pseudo_mask is None:
                    return torch.cat(losses, 1).min(1, True)[0].mean()
                else:
                    output = torch.cat(losses, 1).min(1, True)[0] * pseudo_mask
                    return output.mean()
            else:
                raise NotImplementedError(
                    'Unknown photometric_reduce_op: {}'.format(self.photometric_reduce_op))
        # Reduce photometric loss
        photometric_loss = sum([reduce_function(photometric_losses[i], pseudo_mask)
                                for i in range(self.n)]) / (self.n)
        # Store and return reduced photometric loss
        self.add_metric('photometric_loss', photometric_loss)
        return photometric_loss

########################################################################################################################

    def calc_smoothness_loss(self, inv_depths, images):
        """
        Calculates the smoothness loss for inverse depth maps.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted inverse depth maps for all scales
        images : list of torch.Tensor [B,3,H,W]
            Original images for all scales

        Returns
        -------
        smoothness_loss : torch.Tensor [1]
            Smoothness loss
        """
        # Calculate smoothness gradients
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images, self.n)
        # Calculate smoothness loss
        smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                                smoothness_y[i].abs().mean()) / 2 ** i
                               for i in range(self.n)]) / self.n
        # Apply smoothness loss weight
        smoothness_loss = self.smooth_loss_weight * smoothness_loss
        # Store and return smoothness loss
        self.add_metric('smoothness_loss', smoothness_loss)
        return smoothness_loss

########################################################################################################################

    def calc_slic_loss(self, segments, inv_depths):
        mask = torch.zeros(segments[0].shape[:]).cuda().repeat(4, 1, 1, 1)
        b = segments[0].shape[0]
        inv_depths = torch.stack(inv_depths)
        inv_depths_seg = torch.zeros(segments[0].shape[:]).cuda().repeat(4, 1, 1, 1)
        # for i in range(self.n):
        for j in range(b):
            for segVal in torch.unique(segments[0][j]):
                    # inv_depths_seg[i][j][segments[i][j] == segVal] = torch.mean(inv_depths[i][j][0][segments[i][j] == segVal])
                inv_depths_seg[:,j,segments[0][j] == segVal] = torch.mean(inv_depths[:,j,0,segments[0][j] == segVal], dim=1).unsqueeze(1).repeat(1, torch.sum(segments[0][j] == segVal))
        
        # mask = torch.stack(inv_depths).squeeze(2) - inv_depths_seg
        mask = inv_depths.squeeze(2) - inv_depths_seg
        slic_loss = torch.norm(mask)
        # Apply slic loss weight
        slic_loss = self.slic_loss_weight * slic_loss
        # Store and return slic loss
        self.add_metric('slic_loss', slic_loss)
        return slic_loss

########################################################################################################################

    def _process_batch_seg(self, dataset, output, batch_idx, domain_name):
        if ('segmentation_logits', 0) not in output:
            return 0

        preds = output['segmentation_logits', 0]
        targets = dataset['segmentation', 0, 0][:, 0, :, :].long()
        losses_seg = dict()

        SEG_CLASS_WEIGHTS = (
            2.8149201869965, 6.9850029945374, 3.7890393733978, 9.9428062438965,
            9.7702074050903, 9.5110931396484, 10.311357498169, 10.026463508606,
            4.6323022842407, 9.5608062744141, 7.8698215484619, 9.5168733596802,
            10.373730659485, 6.6616044044495, 10.260489463806, 10.287888526917,
            10.289801597595, 10.405355453491, 10.138095855713, 0
        )
        self.weights = torch.tensor(SEG_CLASS_WEIGHTS).to(preds.get_device())

        losses_seg["loss_seg"] = F.cross_entropy(
            preds, targets, self.weights, ignore_index=255
        )
        # losses_seg["loss_seg"] = F.cross_entropy(
        #     preds, targets, ignore_index=255
        # )

        return losses_seg["loss_seg"]

########################################################################################################################

    def bce_loss(self, y_pred, y_label):
        y_truth_tensor = torch.FloatTensor(y_pred.size())
        y_truth_tensor.fill_(y_label)
        y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
        return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

########################################################################################################################
    
    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def compute_perceptional_loss(self, tgt_f, src_f):
        loss = self.robust_l1(tgt_f, src_f).mean(1, True)
        return loss

########################################################################################################################

    def berhu_loss(self, depth, gt_depth):
        residual = (gt_depth - depth).abs()
        max_res = residual.max()
        condition = 0.2 * max_res
        L2_loss = ((residual**2 + condition**2) / (2 * condition))
        L1_loss = residual
        loss = torch.where(residual > condition, L2_loss, L1_loss).mean(1, True)
        return loss

########################################################################################################################

    def forward(self, image, context, inv_depths, K, ref_K, poses, progress=0.0, batch=None):
        """
        Calculates training photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B, 3, H, W]
            Original image
        context : list of torch.Tensor [B, 3, H, W]
            Context containing a list of reference images
        inv_depths : list of torch.Tensor 4*[B, 1, H, W]
            Predicted depth maps for the original image, in all scales
        K : torch.Tensor [B, 3, 3]
            Original camera intrinsics
        ref_K : torch.Tensor [B, 3, 3]
            Reference camera intrinsics
        poses : list of Pose
            Camera transformation between original and context
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        # If using progressive scaling
        self.n = self.progressive_scaling(progress)

        loss = 0
        if self.stage != 2:
            # Loop over all reference images
            photometric_losses = [[] for _ in range(self.n)]
            images = match_scales(image, inv_depths, self.n) # 4*[B, 3, H, W]

            for j, (ref_image, pose) in enumerate(zip(context, poses)):
                # Calculate warped images
                ref_warped0 = self.warp_ref_image(inv_depths, ref_image, K, ref_K, pose[0], self.n)

                # Calculate and store image loss
                photometric_loss = self.calc_photometric_loss(ref_warped0, images, self.n)

                for i in range(self.n):
                    photometric_losses[i].append(photometric_loss[i])

                # If using automask
                if self.automask_loss:
                    # Calculate and store unwarped image loss
                    ref_images = match_scales(ref_image, inv_depths, self.n)
                    unwarped_image_loss = self.calc_photometric_loss(ref_images, images, self.n)
                    for i in range(self.n):
                        photometric_losses[i].append(unwarped_image_loss[i])

            # Calculate reduced photometric loss
            loss = self.reduce_photometric_loss(photometric_losses, batch['GMM_prob'] if self.stage != 1 and self.stage != 3 else None)
        
        pseudo_loss = torch.zeros(1)
        if self.stage != 1 and self.stage != 3 and self.pseudo_loss_weight > 0.0:
            pseudo_inv_depth = batch['pseudo_inv_depth']
            
            pseudo_loss = []
            for i in range(self.n):
                pseudo_Berhu = self.berhu_loss(inv_depths[i], pseudo_inv_depth) * batch['GMM_prob']
                pseudo_loss.append(pseudo_Berhu.mean())
            pseudo_loss = self.pseudo_loss_weight * sum(pseudo_loss) / (self.n)
            loss += pseudo_loss

        if self.stage != 2:
            # Include smoothness loss if requested
            if self.smooth_loss_weight > 0.0:
                loss += self.calc_smoothness_loss(inv_depths, images)
        
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'source_loss': pseudo_loss.unsqueeze(0),
            'metrics': self.metrics,
        }

########################################################################################################################

    def GMM(self, image, context, inv_depths, 
            K, ref_K, poses):
        """
        Calculates training photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B, 3, H, W]
            Original image
        context : list of torch.Tensor [B, 3, H, W]
            Context containing a list of reference images
        inv_depths : list of torch.Tensor 4*[B, 1, H, W]
            Predicted depth maps for the original image, in all scales
        K : torch.Tensor [B, 3, 3]
            Original camera intrinsics
        ref_K : torch.Tensor [B, 3, 3]
            Reference camera intrinsics
        poses : list of Pose
            Camera transformation between original and context
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        pseudo_inv_depth = inv_depths
        photometric_losses_pseudo = [[] for _ in range(1)]

        for j, (ref_image, pose) in enumerate(zip(context, poses)):
            ref_warped_pseudo = self.warp_ref_image([pseudo_inv_depth], ref_image, K, ref_K, pose[0], 1)

            # warp_image = ref_warped_pseudo[0][0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255
            # cv2.imwrite('data/warp.png', warp_image)

            photometric_loss_pseudo = self.calc_photometric_loss(ref_warped_pseudo, [image], 1)
            photometric_losses_pseudo[0].append(photometric_loss_pseudo[0])

        loss_pseudo = torch.cat(photometric_losses_pseudo[0], 1).min(1, True)[0]

        return loss_pseudo
