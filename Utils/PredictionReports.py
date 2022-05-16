import argparse
from typing import Dict, Optional, Union
import os
from pytorch_lightning.loggers.base import rank_zero_experiment
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from pytorch_lightning.loggers import LightningLoggerBase
from sksurv.metrics import cumulative_dynamic_auc
import torch
from torch import nn
from sksurv.metrics import concordance_index_censored
import torchmetrics
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.utilities.cloud_io import get_filesystem
import logging
from pytorch_lightning.loggers import TensorBoardLogger

class PredictionReports(TensorBoardLogger):
    def __init__(self, config,
                 save_dir: str,
                 name: Optional[str] = "default",
                 version: Optional[Union[int, str]] = None,
                 log_graph: bool = False,
                 default_hp_metric: bool = True,
                 prefix: str = "",
                 sub_dir: Optional[str] = None,
                 **kwargs, ):

        super().__init__(save_dir, name, version, log_graph, default_hp_metric, prefix, sub_dir, **kwargs)
        self.config = config

    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        pass

    @property
    def version(self):
        description = ''
        for i, param in enumerate(self.config['CRITERIA'].keys()):
            clinical_criteria = str(self.config['CRITERIA'][param])
            if i > 0:
                description = description + '_'
            description = description + param + '+' + '+'.join(clinical_criteria)
        # Return the experiment version, int or str.
        modules = self.config['DATA']['module']
        description = description + '_' + 'modalities' + '+' + '+'.join(modules)

        return description

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.experiment.add_scalar(k, v, step)

    def log_image(self, img, text, current_epoch=None):
        img_batch = img.view(img.shape[0] * img.shape[1], *[1, img.shape[2], img.shape[3]])
        grid = torchvision.utils.make_grid(img_batch)
        self.experiment.add_image(text, grid, current_epoch)
        return grid

    def log_text(self) -> None:
        configurations = 'The img_dim is ' + str(self.config['MODEL']['img_sizes']) + ' and the modules included are ' + str(
            self.config['DATA']['module'])
        self.experiment.add_text('configurations:', configurations)

    def regression_matrix(self, prediction, label, prefix):
        r_out = {}
        if 'cindex' in self.config['REPORT']['matrix']:
            cindex = c_index(prediction, label)
            r_out[prefix + 'cindex'] = cindex[0]
        if 'r2' in self.config['REPORT']['matrix']:
            r2 = r2_index(prediction, label)
            r_out[prefix + 'r2'] = r2
        return r_out

    def classification_matrix(self, prediction, label, prefix):
        c_out = {}
        if 'ROC' in self.config['REPORT']['matrix']:
            auroc = torchmetrics.AUROC()
            accuracy = auroc(prediction, label.int())
            c_out[prefix + 'roc'] = accuracy
        if 'Specificity' in self.config['REPORT']['matrix']:
            specificity = torchmetrics.Specificity()
            spec = specificity(prediction.to('cpu'), label.int().to('cpu'))
            c_out[prefix + 'specificity'] = spec
        return c_out

    def generate_cumulative_dynamic_auc(self, y_train, prediction, label, current_epoch) -> None:
        # this function has issues
        risk_score = 1 / prediction
        va_times = np.arange(int(label.cpu().min()) + 1, label.cpu().max(), 1)

        dtypes = np.dtype([('event', np.bool_), ('time', np.float)])
        construct_test = np.ndarray(shape=(len(label),), dtype=dtypes)
        for i in range(len(label)):
            construct_test[i] = (True, label[i].cpu().numpy())
        # construct_test = {'death': torch.ones(y_test.shape, dtype=torch.bool), 'time': y_test}
        construct_train = np.ndarray(shape=(len(y_train),), dtype=dtypes)
        for i in range(len(y_train)):
            construct_train[i] = (True, y_train.to_numpy()[i])

        cph_auc, cph_mean_auc = cumulative_dynamic_auc(
            construct_train, construct_test, risk_score.cpu(), va_times
        )
        fig = plt.figure()
        plt.plot(va_times, cph_auc, marker="o")
        plt.axhline(cph_mean_auc, linestyle="--")
        plt.xlabel("days from enrollment")
        plt.ylabel("time-dependent AUC")
        plt.grid(True)
        plt.show()
        self.experiment.add_figure("AUC", fig, current_epoch)

    def plot_AUROC(self, prediction, label, prefix, current_epoch=None) -> None:
        roc = torchmetrics.ROC()
        fpr, tpr, _ = roc(prediction, label)
        fig = plt.figure()
        # lw = 2
        # plt.plot(fpr.cpu(), tpr.cpu(), color='darkorange', lw=lw)
        plt.plot(fpr.cpu(), tpr.cpu(), color='darkorange')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.show()
        self.experiment.add_figure(prefix + "AUC", fig, current_epoch)

    def worst_case_show(self, validation_step_outputs, prefix):
        out = {}
        worst_AE = 0
        for i, data in enumerate(validation_step_outputs):
            loss = data['MAE']
            idx = torch.argmax(loss)
            if loss[idx] > worst_AE:
                worst_img = data['img'][idx]
                worst_AE = loss[idx]
        out[prefix + 'worst_AE'] = worst_AE
        return out

    def report_epoch(self, prediction, label, train_label, validation_step_outputs,
                     current_epoch, prefix) -> None:
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            regression_out = self.regression_matrix(prediction, label, prefix)
            self.log_metrics(regression_out, current_epoch)
            if 'AUC' in self.config['REPORT']['matrix']:
                self.generate_cumulative_dynamic_auc(train_label, prediction, label, current_epoch)
            if 'WorstCase' in self.config['REPORT']['matrix']:
                worst_record = self.worst_case_show(validation_step_outputs, prefix)
                self.log_metrics({prefix+'worst_AE': worst_record[prefix+'worst_AE']},current_epoch)
                text = 'validate_worst_case_img'
                self.log_image(worst_record[prefix+'worst_img'], text, current_epoch)

        if self.config['MODEL']['Prediction_type'] == 'Classification':
            classification_out = self.classification_matrix(prediction.squeeze(), label, prefix)
            self.log_metrics(classification_out, current_epoch)
            if 'RUC' in self.config['REPORT']['matrix']:
                self.plot_AUROC(prediction.squeeze(), label, prefix, current_epoch)


def r2_index(prediction, label):
    loss = nn.MSELoss()
    MSE = loss(prediction, label)
    SSres = MSE * label.shape[0]
    SStotal = torch.sum(torch.square(label - torch.mean(label)))
    r2 = 1 - SSres / SStotal
    return r2


