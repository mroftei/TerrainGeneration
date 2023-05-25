import torch
import torchmetrics
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

class ModelBase(pl.LightningModule):
    def __init__(self, classes):
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
        # self.example_input_array = torch.zeros(1,2,1024)

    def on_train_start(self):
        if self.global_step==0: 
            init_logs = {k: 0 for k in self.val_metrics.keys()}
            self.logger.log_hyperparams(self.hparams, init_logs)

    def training_step(self, batch, batch_nb):
        data, target, _ = batch
        output = self.forward(data)
        loss = self.loss(output, target)
        if self.global_step!= 0: self.logger.log_metrics({'train/loss': loss, 'epoch': self.current_epoch}, self.global_step)
        return loss

    def validation_step(self, batch, batch_nb):
        data, target, snr = batch
        output = self.forward(data)
        self.val_metrics.update(output, target)
        self.cm_metric.update(output, target)

    def on_validation_epoch_end(self):
        metrics_dict = self.val_metrics.compute()
        self.val_metrics.reset()
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
        self.logger.experiment.add_figure("val/cm", fig, global_step=self.global_step)

        # self.logger.experiment.log({x: metrics_dict[x] for x in metrics_dict if x != 'val/CM'}, commit=True)
        
    def test_step(self, batch, batch_nb):
        data, target, snr = batch
        output = self.forward(data)
        self.test_metrics.update(output, target)
        if self.cm_metric: self.cm_metric.update(output, target)
        return {"test_out": output, "test_true": target, "test_snr": snr}

    def on_test_epoch_end(self):
        metrics_dict = self.test_metrics.compute()
        self.test_metrics.reset()
        if self.global_step!= 0: self.logger.log_metrics(metrics_dict, self.global_step)
        
        if self.cm_metric:
            mpl.use("Agg")
            fig = plt.figure(figsize=(13, 13))
            cm = self.cm_metric.compute().numpy(force=True)
            self.cm_metric.reset()
            ax = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)
            # labels, title and ticks
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(self.classes , rotation=90)
            ax.yaxis.set_ticklabels(self.classes , rotation=0)
            plt.tight_layout()
            self.logger.experiment.add_figure("test/cm", fig, global_step=self.global_step)
        
        # test_true = torch.cat([x['test_true'] for x in outputs])
        # test_out = torch.cat([x['test_out'] for x in outputs])
        # test_snr = torch.cat([x['test_snr'] for x in outputs])
        
        # # SNR table        
        # SNRs = torch.unique(test_snr).numpy(force=True)
        # F1s = []
        # for snr in SNRs:
        #     ind = test_snr == snr
        #     F1s.append(torchmetrics.functional.f1_score(test_out[ind], test_true[ind]).numpy(force=True))

        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.subplots()
        # plt.plot(SNRs, F1s, linestyle='-', marker='o')
        # ax.set_title('SNR Accuracy')
        # plt.xlabel('SNR')
        # plt.ylabel('Accuracy')
        # plt.ylim(0,1)
        # plt.grid(True)
        # self.logger.experiment.add_figure("test/snr_f1", fig, global_step=self.global_step)
