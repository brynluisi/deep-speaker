import argparse
import json
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torchaudio
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, Timer, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn

from deepspeaker.configs.config import Config
from deepspeaker.data.preprocess import Preprocessor
from deepspeaker.data.utils import get_input_target_from_path
from deepspeaker.models.loss_functions import ESRLoss, MelSpectrogramLoss, MyMSELoss, LossWrapper
from deepspeaker.models.my_logging import create_wandb_audio_table
from deepspeaker.models.networks import AudioDataModule, RNNModel, WaveNetModel

# Random seed setting
torch.manual_seed(16)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

ROOT_DIR = Path(__file__).parent.parent


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


prsr = argparse.ArgumentParser(
    description="""This script implements training for neural network amplifier/distortion effects modelling. This is
    intended to recreate the training of models of the ht1 amplifier and big muff distortion pedal, but can easily be 
    adapted to use any dataset"""
)

prsr.add_argument("--data_dir", help='Location of the "Data" directory')
prsr.add_argument("--data_name", help="Name of directory containing audio files")
prsr.add_argument("--project_name", type=str)
prsr.add_argument("--run_name", type=str)

"""
Arguments for training loop
"""
prsr.add_argument(
    "--epochs",
    "-eps",
    type=int,
    help="Max number of training epochs to run",
)
prsr.add_argument("--learn_rate", "-lr", type=float, help="Initial learning rate")
prsr.add_argument("--loss_function", "-lf", help="Loss Function")
prsr.add_argument("--batch_size", "-bs", type=int, help="Training mini-batch size")
prsr.add_argument(
    "--sequence_length",
    "-slen",
    type=int,
    help="Training audio segment length in samples",
)

"""
Arguments for RNN Model
"""
prsr.add_argument(
    "--truncated_bptt_steps",
    "-bptt",
    type=Union[int, str2bool],
    help="Number of truncated backprops",
)
prsr.add_argument("--num_layers", "-nl", type=int, help="Number of recurrent layers")
prsr.add_argument(
    "--hidden_size",
    "-hs",
    type=int,
    help="Recurrent unit hidden state size",
)
prsr.add_argument("--cell_type", "-ct", help="Recurrent unit cell type (lstm, gru)")
prsr.add_argument("--bidirectional", "-bi", type=str2bool)
prsr.add_argument("--skip", type=str2bool, help="Skip connection")

"""
Arguments for WaveNet model
"""

prsr.add_argument("--dilation_depth", type=int, help="Dilation depth")
prsr.add_argument("--num_channels", type=int, help="Number of channels")
prsr.add_argument("--kernel_size", type=int, help="Kernel size")
prsr.add_argument("--num_blocks", type=int, help="Number of blocks")

# This must be set with command-line arguments for non-default values and it cannot be set
# within the config file (obviously)
prsr.add_argument(
    "--config_path",
    "-cl",
    default=f"{ROOT_DIR}/configs",
    help='Location of the "Configs" directory',
)
# This must be set with command-line arguments for non-default values and it cannot be set
# within the config file (obviously)
prsr.add_argument(
    "--config_name",
    "-l",
    help="File path, to a JSON config file, arguments listed in the config file will replace the defaults",
    default="lstm_local.yaml",
)

def main():
    args = prsr.parse_args()
    CONFIG = Config(args)
    CONFIG.save_config()

    # Initialize wandb for logging
    config_dict = vars(CONFIG)
    json.dump(config_dict, open(f"{CONFIG.model_save_path}/config.json", "w"))
    if config_dict.get("run_name"):
        run = wandb.init(
            project=CONFIG.project_name,
            dir=CONFIG.model_save_path,
            name=CONFIG.run_name,
        )
        wandb_logger = WandbLogger(project=CONFIG.project_name, name=CONFIG.run_name)
    else:
        run = wandb.init(project=CONFIG.project_name, dir=CONFIG.model_save_path)
        wandb_logger = WandbLogger(project=CONFIG.project_name)
    wandb_logger.experiment.config.update(config_dict)

    # Preprocess data and write to data directory
    preprocessor = Preprocessor(
        raw_data_dir=CONFIG.raw_data_dir, processed_data_dir=CONFIG.processed_data_dir
    )
    preprocessor.write_audio()

    # Create datamodules and dataloaders
    train_datamodule = AudioDataModule(
        f"{CONFIG.processed_data_dir}/train",
        CONFIG.sequence_length,
        CONFIG.batch_size,
        CONFIG.model,
    )

    # Need to get audio info so sequence length is the number of frames
    test_input_path, test_target_path = get_input_target_from_path(
        f"{CONFIG.processed_data_dir}/test"
    )
    test_info = torchaudio.info(test_input_path)

    test_datamodule = AudioDataModule(
        f"{CONFIG.processed_data_dir}/test",
        sequence_length=test_info.num_frames,
        batch_size=1,
        model_type=CONFIG.model,
    )

    if CONFIG.loss_function == "MSE":
        criterion = MyMSELoss()
    elif CONFIG.loss_function == "ESR":
        criterion = ESRLoss()
    elif CONFIG.loss_function == "Mel":
        criterion = MelSpectrogramLoss(CONFIG.device)
    elif CONFIG.loss_function == "MSE_Mel":
        criterion = LossWrapper({"MSE": 0.5, "Mel": 0.5}, CONFIG.device)
    elif CONFIG.loss_function == "ESR_Mel":
        criterion = LossWrapper({"ESR": 0.5, "Mel": 0.5}, CONFIG.device)
    else:
        raise ValueError("--loss_function must equal 'MSE', 'ESR', or 'Mel'")

    if CONFIG.model == "RNNModel":
        model = RNNModel(
            CONFIG.input_channels,
            CONFIG.output_channels,
            CONFIG.batch_size,
            CONFIG.hidden_size,
            CONFIG.sequence_length,
            CONFIG.cell_type,
            CONFIG.num_layers,
            CONFIG.bidirectional,
            criterion,
            CONFIG.learn_rate,
            CONFIG.model_save_path,
            CONFIG.truncated_bptt_steps,
            CONFIG.skip,
        )
    elif CONFIG.model == "WaveNetModel":
        model = WaveNetModel(
            criterion,
            CONFIG.learn_rate,
            CONFIG.model_save_path,
            CONFIG.input_channels,
            CONFIG.output_channels,
            CONFIG.num_channels,
            CONFIG.dilation_depth,
            CONFIG.num_blocks,
            CONFIG.kernel_size,
        )
    else:
        raise ValueError("--model must equal 'RNNModel' or 'WaveNetModel'")

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log="all")

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    timer = Timer(duration="00:12:00:00")
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=CONFIG.model_save_path,
        filename="best_model",
    )

    # Define model trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        default_root_dir=CONFIG.model_save_path,
        max_epochs=CONFIG.epochs,
        log_every_n_steps=1,
        accelerator=CONFIG.device,
        devices=1,
        check_val_every_n_epoch=1,
        callbacks=[lr_monitor, timer, checkpoint_callback],
    )

    # train model
    trainer.fit(model=model, datamodule=train_datamodule)

    model.to_onnx(f"{CONFIG.model_save_path}/model.onnx", export_params=True)
    base_path = str(Path(CONFIG.model_save_path).parent)
    try:
        torch.save(model.state_dict(), f"{CONFIG.model_save_path}/model.pt")
        wandb.save(f"{CONFIG.model_save_path}/model.onnx", base_path=base_path)
    except:
        print("Could not save model.pt and model.onnx")

    trainer.test(model, datamodule=test_datamodule)

    # # load model from checkpoint
    # model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    #
    # # predict on test set
    # trainer.predict(model, datamodule=test_datamodule)

    train_time = timer.time_elapsed("train")
    test_time = timer.time_elapsed("test")

    # Log training and testing times
    run.log({"train_time": train_time})
    run.log({"test_time": test_time})

    # Log audio table for test data
    table = create_wandb_audio_table(
        test_input_path, test_target_path, CONFIG, trainer.callback_metrics["test_loss"]
    )

    run.log({"Test Audio Output": table})


if __name__ == "__main__":
    main()
