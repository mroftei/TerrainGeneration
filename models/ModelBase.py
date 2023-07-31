import torch
import torchmetrics
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

class ModelBase(pl.LightningModule):
    def __init__(self, classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        metrics = {
            'F1': torchmetrics.classification.MulticlassF1Score(num_classes=len(classes), compute_on_step=False),
        }
        metrics = torchmetrics.MetricCollection(metrics)
        self.val_metrics = metrics.clone(f"val/")
        self.test_metrics = metrics.clone(f"test/")
        self.cm_metric = torchmetrics.classification.MulticlassConfusionMatrix(len(classes), normalize='true')

        self.classes = classes
        self.outputs_list = []

    def on_train_start(self):
        if self.global_step==0: 
            init_logs = {k: 0 for k in self.val_metrics.keys()}
            init_logs.update({k: 0 for k in self.test_metrics.keys()})
            self.logger.log_hyperparams(self.hparams, init_logs)

    def on_test_start(self):
        # Initialize outputs list
        self.outputs_list = torch.zeros((len(self.trainer.datamodule.ds_test.indices), len(self.classes)))

    def training_step(self, batch, batch_nb):
        data, target, _ = batch
        output = self.forward(data)
        loss = self.loss(output, target)
        if self.global_step!= 0: self.logger.log_metrics({'train/loss': loss, 'epoch': self.current_epoch}, self.global_step)
        self.log("loss", loss, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_nb):
        data, target, snr = batch
        output = self.forward(data)
        self.val_metrics.update(output, target)
        self.cm_metric.update(output, target)

        # batch_size = len(snr)
        # batch_idx = batch_nb*batch_size
        # self.outputs_list[batch_idx:batch_idx+batch_size] = output.detach().cpu()

    def on_validation_epoch_end(self):
        metrics_dict = self.val_metrics.compute()
        self.val_metrics.reset()
        if self.global_step!= 0: self.logger.log_metrics(metrics_dict, self.global_step)
        self.log("val/F1", metrics_dict['val/F1'], prog_bar=True, logger=False)
        
        # Confusion Matrix
        mpl.use("Agg")
        fig = plt.figure(figsize=(13, 13))
        cm = self.cm_metric.compute().numpy(force=True)
        self.cm_metric.reset()
        ax = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)
        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(self.classes, rotation=90)
        ax.yaxis.set_ticklabels(self.classes , rotation=0)
        plt.tight_layout()
        self.logger.experiment.add_figure("val/cm", fig, global_step=self.global_step)

        # self.logger.experiment.log({x: metrics_dict[x] for x in metrics_dict if x != 'val/CM'}, commit=True)

        # test_snr = self.trainer.datamodule.ds_val.dataset.tensors[2][self.trainer.datamodule.ds_val.indices][:len(self.outputs_list)]
        # test_true = self.trainer.datamodule.ds_val.dataset.tensors[1][self.trainer.datamodule.ds_val.indices][:len(self.outputs_list)]

        # SNRs = torch.unique(test_snr).numpy(force=True)
        # F1s = []
        # for snr in SNRs:
        #     ind = test_snr == snr
        #     F1s.append(torchmetrics.functional.classification.multiclass_f1_score(self.outputs_list[ind].cpu(), test_true[ind], len(self.classes)).numpy(force=True))
        # self.outputs_list = self.outputs_list.zero_()

        # fig = plt.figure(figsize=(8, 4))
        # ax = fig.subplots()
        # plt.plot(SNRs, F1s, linestyle='-', marker='o')
        # ax.set_title('SNR F1')
        # plt.xlabel('SNR')
        # plt.ylabel('F1')
        # plt.ylim(0,1)
        # plt.grid(True)
        # plt.tight_layout()
        # self.logger.experiment.add_figure("val/snr_f1", fig, global_step=self.global_step)
        
    def test_step(self, batch, batch_nb):
        data, target, snr = batch
        output = self.forward(data)
        self.test_metrics.update(output, target)
        self.cm_metric.update(output, target)
        
        batch_size = len(snr)
        batch_idx = batch_nb*batch_size
        self.outputs_list[batch_idx:batch_idx+batch_size] = output.detach().cpu()

    def on_test_epoch_end(self):
        metrics_dict = self.test_metrics.compute()
        self.test_metrics.reset()
        if self.global_step!= 0: self.logger.log_metrics(metrics_dict, self.global_step)
        
        # Confusion Matrix
        mpl.use("Agg")
        fig = plt.figure(figsize=(13, 13))
        cm = self.cm_metric.compute().numpy(force=True)
        self.cm_metric.reset()
        ax = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)
        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(self.classes, rotation=90)
        ax.yaxis.set_ticklabels(self.classes , rotation=0)
        plt.tight_layout()
        self.logger.experiment.add_figure("test/cm", fig, global_step=self.global_step)
        
        test_snr = self.trainer.datamodule.ds_test.dataset.tensors[2][self.trainer.datamodule.ds_test.indices][:len(self.outputs_list)]
        test_true = self.trainer.datamodule.ds_test.dataset.tensors[1][self.trainer.datamodule.ds_test.indices][:len(self.outputs_list)]

        test_snr = torch.round(test_snr)
        SNRs = torch.unique(test_snr).numpy(force=True)
        F1s = []
        for snr in SNRs:
            ind = test_snr == snr
            F1s.append(torchmetrics.functional.classification.multiclass_f1_score(self.outputs_list[ind].cpu(), test_true[ind], len(self.classes)).numpy(force=True))
        self.outputs_list = self.outputs_list.zero_()

        fig = plt.figure(figsize=(8, 4))
        ax = fig.subplots()
        plt.plot(SNRs, F1s, linestyle='-', marker='o')
        ax.set_title('SNR F1')
        plt.xlabel('SNR')
        plt.ylabel('F1')
        plt.ylim(0,1)
        plt.grid(True)
        plt.tight_layout()
        self.logger.experiment.add_figure("test/snr_f1", fig, global_step=self.global_step)
