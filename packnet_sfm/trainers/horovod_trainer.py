import os
import torch
import torch.nn as nn
import horovod.torch as hvd
from packnet_sfm.trainers.base_trainer import BaseTrainer, sample_to_cuda, _batch_to_device
from packnet_sfm.utils.config import prep_logger_and_checkpoint
from packnet_sfm.utils.logging import print_config
from packnet_sfm.utils.logging import AvgMeter

import numpy as np
import warnings
from sklearn.mixture import GaussianMixture

class Evaluator(object):
    # CONF MATRIX
    #     0  1  2  (PRED)
    #  0 |TP FN FN|
    #  1 |FP TP FN|
    #  2 |FP FP TP|
    # (GT)
    # -> rows (axis=1) are FN
    # -> columns (axis=0) are FP
    @staticmethod
    def iou(conf):  # TP / (TP + FN + FP)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            iu = np.diag(conf) / (conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf))
        meaniu = np.nanmean(iu)
        result = {'iou': dict(zip(range(len(iu)), iu)), 'meaniou': meaniu}
        return result

    @staticmethod
    def accuracy(conf):  # TP / (TP + FN) aka 'Recall'
        # Add 'add' in order to avoid division by zero and consequently NaNs in iu
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            totalacc = np.diag(conf).sum() / (conf.sum())
            acc = np.diag(conf) / (conf.sum(axis=1))
        meanacc = np.nanmean(acc)
        result = {'totalacc': totalacc, 'meanacc': meanacc, 'acc': acc}
        return result

    @staticmethod
    def precision(conf):  # TP / (TP + FP)
        # Add 'add' in order to avoid division by zero and consequently NaNs in iu
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            prec = np.diag(conf) / (conf.sum(axis=0))
        meanprec = np.nanmean(prec)
        result = {'meanprec': meanprec, 'prec': prec}
        return result

    @staticmethod
    def freqwacc(conf):
        # Add 'add' in order to avoid division by zero and consequently NaNs in iu
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            iu = np.diag(conf) / (conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf))
            freq = conf.sum(axis=1) / (conf.sum())
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        result = {'freqwacc': fwavacc}
        return result

    @staticmethod
    def depththresh(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        result = {'delta1': a1, 'delta2': a2, 'delta3': a3}
        return result

    @staticmethod
    def deptherror(gt, pred):
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())
        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        result = {'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse, 'rmse_log': rmse_log}
        return result

class SegmentationRunningScore(object):
    def __init__(self, n_classes=20):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask_true = (label_true >= 0) & (label_true < n_class)
        mask_pred = (label_pred >= 0) & (label_pred < n_class)
        mask = mask_pred & mask_true
        label_true = label_true[mask].astype(np.int)
        label_pred = label_pred[mask].astype(np.int)
        hist = np.bincount(n_class * label_true + label_pred,
                           minlength=n_class*n_class).reshape(n_class, n_class).astype(np.float)
        return hist

    def update(self, label_trues, label_preds):
        # label_preds = label_preds.exp()
        # label_preds = label_preds.argmax(1).cpu().numpy() # filter out the best projected class for each pixel
        # label_trues = label_trues.numpy() # convert to numpy array

        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes) # update confusion matrix

    def get_scores(self, listofparams=None):
        """Returns the evaluation params specified in the list"""
        possibleparams = {
            'iou': Evaluator.iou,
            'acc': Evaluator.accuracy,
            'freqwacc': Evaluator.freqwacc,
            'prec': Evaluator.precision
        }
        if listofparams is None:
            listofparams = possibleparams

        result = {}
        for param in listofparams:
            if param in possibleparams.keys():
                result.update(possibleparams[param](self.confusion_matrix))
        return result

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class HorovodTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        hvd.init()
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
        torch.cuda.set_device(hvd.local_rank())
        torch.backends.cudnn.benchmark = True

        self.avg_loss = AvgMeter(50)
        self.avg_source_loss = AvgMeter(50)
        self.dtype = kwargs.get("dtype", None)  # just for test for now
        
        # fit a two-component GMM to the loss
        self.gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    @property
    def proc_rank(self):
        return hvd.rank()

    @property
    def world_size(self):
        return hvd.size()

    def fit(self, module):
        # Prepare module for training
        module.trainer = self
        # Update and print module configuration
        prep_logger_and_checkpoint(module)
        print_config(module.config)

        # Send module to GPU
        module = module.to('cuda')
        # Configure optimizer and scheduler
        module.configure_optimizers()

        # Create distributed optimizer
        compression = hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(module.optimizer,
            named_parameters=module.named_parameters(), compression=compression, backward_passes_per_step=2)
        scheduler = module.scheduler

        # Get train and val dataloaders
        train_dataloader = module.train_dataloader()
        val_dataloaders = module.val_dataloader()

        # Log to wandb
        if module.logger:
            module.logger.watch(module, log='all', log_freq=100)

        # Epoch loop
        for epoch in range(module.current_epoch, self.max_epochs):
            # Train
            self.train(train_dataloader, module, optimizer)
            # Validation
            validation_output = self.validate(val_dataloaders, module)
            # Check and save model
            self.check_and_save(module, validation_output)
            # Update current epoch
            module.current_epoch += 1
            # Take a scheduler step
            scheduler.step()

    def train(self, dataloader, module, optimizer):
        # Set module to train
        module.train()
        
        # Shuffle dataloader sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)

        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            dataloader, module.config.datasets.train)
        # Start training loop
        outputs = []
        # For all batches
        for i, batch in progress_bar:
            # Reset optimizer
            optimizer.zero_grad()

            # Send samples to GPU and take a training step
            batch = sample_to_cuda(batch)

            output = module.training_step(batch, i)

            # Backprop through loss and take an optimizer step
            output['loss'].backward()

            optimizer.step()

            # Append output to list of outputs
            output['loss'] = output['loss'].detach()
            output['source_loss'] = output['source_loss'].detach()
            outputs.append(output)
            # Update progress bar if in rank 0
            if self.is_rank_0:
                progress_bar.set_description(
                    'Epoch {} | Avg.Loss {:.4f} Sem.Loss {:.4f} '.format(
                        module.current_epoch, self.avg_loss(output['loss'].item()), self.avg_source_loss(output['source_loss'].item())))
        # Return outputs for epoch end
        return module.training_epoch_end(outputs)

    def validate(self, dataloaders, module):
        with torch.no_grad():
            # Set module to eval
            module.eval()
            # Start validation loop
            all_outputs = []
            # For all validation datasets
            for n, dataloader in enumerate(dataloaders):
                # Prepare progress bar for that dataset
                progress_bar = self.val_progress_bar(
                    dataloader, module.config.datasets.validation, n)
                outputs = []
                # For all batches
                for i, batch in progress_bar:
                    # Send batch to GPU and take a validation step
                    batch = sample_to_cuda(batch)
                    output = module.validation_step(batch, i, n)
                    # Append output to list of outputs
                    outputs.append(output)
                # Append dataset outputs to list of all outputs
                all_outputs.append(outputs)
            # Return all outputs for epoch end
            return module.validation_epoch_end(all_outputs)

    def _run_validation(self, source_validation_dataloader, module):
        print(f'Validation scores for epoch {module.current_epoch}:')

        segmentation_scores = dict()
        module.eval()
        # torch.no_grad() = disable gradient calculation
        with torch.no_grad():
            for batch in source_validation_dataloader:
                domain = batch['domain'][0]
                num_classes = batch['num_classes'][0].item()

                if domain not in segmentation_scores:
                    segmentation_scores[domain] = SegmentationRunningScore(num_classes)

                batch_gpu = _batch_to_device((batch,))
                outputs = module.semantic_net(batch_gpu)[0]  # forward the data through the network

                segs_gt = batch['segmentation', 0, 0].squeeze(1).long() # shape [1, 1024, 2048]
                segs_pred = outputs[0]['segmentation_logits', 0] # shape [1, 20, 192, 640] one for every class
                import torch.nn.functional as functional
                segs_pred = functional.interpolate(segs_pred, segs_gt[0, :, :].shape, mode='nearest') # upscale predictions

                for i in range(segs_pred.shape[0]):
                    seg_gt = segs_gt[i].unsqueeze(0)
                    seg_pred = segs_pred[i].unsqueeze(0)

                    seg_pred = seg_pred.exp().cpu() # exp preds and shift to CPU
                    seg_pred = seg_pred.numpy() # transform preds to np array
                    seg_pred = seg_pred.argmax(1) # get the highest score for classes per pixel
                    seg_gt = seg_gt.numpy() # transform gt to np array

                    segmentation_scores[domain].update(seg_gt, seg_pred)

        for domain, score in segmentation_scores.items():
            metrics = score.get_scores()

            print(f'  - {domain}:')

            for metric in sorted(metrics):
                value = metrics[metric]

                if metric in ('iou', 'acc', 'prec'):
                    # ignore non-scalars
                    continue

                print(f'    - {metric}: {value:.4f}')

    def test(self, module):
        # Send module to GPU
        module = module.to('cuda', dtype=self.dtype)
        # Get test dataloaders
        test_dataloaders = module.test_dataloader()
        # Run evaluation
        self.evaluate(test_dataloaders, module)

    @torch.no_grad()
    def evaluate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start evaluation loop
        all_outputs = []
        # For all test datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.test, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a test step
                batch = sample_to_cuda(batch, self.dtype)
                output = module.test_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)

        # Return all outputs for epoch end
        return module.test_epoch_end(all_outputs)
