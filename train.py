import sys
import torch
import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder, StochasticWeightAveraging

from data.RML2018DataModule import RML2018DataModule
from data.CSPB2018DataModule import CSPB2018DataModule
from data.DisAMRDataModule import DisAMRDataModule
from models import *

def train(model, dm, name, epochs=40, precision="32", debug=False):
    logger = TensorBoardLogger("./logs", name, version=(None if debug else os.environ["SLURM_JOB_ID"]), log_graph=False, default_hp_metric=False)
    callbacks = [
        ModelCheckpoint(monitor='val/F1', mode='max', save_top_k=1, save_last=True),
        # LearningRateMonitor(logging_interval='step'),
        # StochasticWeightAveraging(swa_lrs=1e-2),
    ]
    profiler = None
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(logger.log_dir), 
    #     record_shapes=True, 
    #     profile_memory=True, 
    #     with_stack=True, 
    #     with_flops=True
    # )


    trainer = Trainer(
        fast_dev_run=False, 
        logger=logger, 
        callbacks=callbacks,
        accelerator='auto', 
        devices='auto', # 'auto'
        num_nodes=1,
        # strategy='ddp',
        sync_batchnorm=True,
        deterministic='warn',
        precision=precision,
        enable_progress_bar=debug,
        max_epochs=epochs,
        # val_check_interval=1.0,
        default_root_dir=logger.log_dir,
        gradient_clip_val=0.5,
        profiler=profiler
    )
    trainer.fit(model, datamodule=dm)

    # Test best model on test set
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(model, datamodule=dm, verbose=True)
    
    logger.finalize('success')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='CNN1')
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--nrx", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=float, default=42)
    parser.add_argument("--precision", type=str, default="32", choices=["64", "32", "16", "bf16"])
    args = parser.parse_args()

    ## Initialize torch
    seed_everything(args.seed)
    torch.set_float32_matmul_precision('high') # Allow TF32 on >=Ampere GPUs

    # dm = RML2018DataModule('/work/ds2/data/k.witham/RML2018.01/GOLD_XYZ_OSC.0001_1024.hdf5', args.bs, use_hard = True)
    # dm = CSPB2018DataModule("/work/ds2/data/k.witham/CSPB.ML.2018", args.bs, download=False)
    dm = DisAMRDataModule("/work/ds2/data/k.witham/CSPB ML Noise Free/cspb_no_noise.hdf5", args.bs, n_rx=args.nrx)

    model_args = {
        # Required
        'input_samples': dm.frame_size, 
        'input_channels': args.nrx, 
        'classes': dm.classes,
        'learning_rate': args.lr,

        # Hyperparameters to log
        'epochs': args.epochs,
        'batch_size': args.bs,
        'precision': args.precision,
    }

    # Create model by finding class matching args.model and initializing with parameters in model_args
    model = getattr(sys.modules['models'], args.model)(**model_args)
    
    session_name = f"{args.model}-{dm.__class__.__name__}"
    train(model, dm, session_name, epochs=args.epochs, precision=args.precision, debug=args.debug)
