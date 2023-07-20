import torch
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from data.RML2018DataModule import RML2018DataModule
from data.CSPB2018DataModule import CSPB2018DataModule
from data.CSPB2018DataModule_v2 import CSPB2018DataModule_v2
from argparse import ArgumentParser
import os
from models import *

def train(model, dm, name, epochs=40, precision="32", debug=False):
    # logger = pl.loggers.TensorBoardLogger("./logs", name, log_graph=True)
    logger = pl.loggers.TensorBoardLogger("./logs", name, version=(None if debug else os.environ["SLURM_JOB_ID"]), log_graph=False)
    checkpointer = pl.callbacks.ModelCheckpoint(dirpath=f"./artifacts/{logger.version}", save_top_k=1)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    torch.set_float32_matmul_precision('high')
    trainer = Trainer(
        fast_dev_run=False, 
        logger=logger, 
        callbacks=[checkpointer, lr_monitor],
        accelerator='auto', 
        devices='auto', # 'auto'
        num_nodes=1,
        # strategy='ddp',
        sync_batchnorm=True,
        deterministic=True,
        precision=precision,
        enable_progress_bar=debug,
        max_epochs=epochs,
        # val_check_interval=1.0,
        default_root_dir=logger.log_dir
    )
    trainer.fit(model, datamodule=dm)

    # Test best model on test set
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(model, datamodule=dm, verbose=True)
    
    logger.finalize('success')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='DeepFIR')
    parser.add_argument("--use_1d", action='store_true')
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=float, default=42)
    parser.add_argument("--precision", type=str, default="32", choices=["64", "32", "16", "bf16"])
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # dm = RML2018DataModule('/work/ds2/data/k.witham/RML2018.01/GOLD_XYZ_OSC.0001_1024.hdf5', args.bs, use_hard = True)
    # dm = CSPB2018DataModule("/work/ds2/data/k.witham/CSPB.ML.2018", args.bs, download=False)
    dm = CSPB2018DataModule_v2("/work/ds2/data/k.witham/CSPB ML Noise Free/cspb_no_noise_6chan.hdf5", args.bs, n_rx=6)

    # args.use_1d=True
    match args.model:
        case 'CNNBlocks':
            model = CNNBlocks(classes=dm.classes, input_samples=dm.frame_size, use_1d=args.use_1d)
        case 'ResNet':
            model = ResNet(classes=dm.classes, input_samples=dm.frame_size, use_1d=args.use_1d)
        case 'Cldnn':
            model = Cldnn2(classes=dm.classes, input_samples=dm.frame_size, use_1d=args.use_1d)
        case 'DeepFIR':
            model = DeepFIR(classes=dm.classes, input_samples=dm.frame_size, input_channels=6)
    
    session_name = f"{args.model}{'_1d' if args.use_1d else ''}-{dm.__class__.__name__}"
    train(model, dm, session_name, epochs=args.epochs, precision=args.precision, debug=args.debug)
