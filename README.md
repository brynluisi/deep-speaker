# deep-speaker

Deep Learning models for speaker emulation. Dataset should be stereo audio where input data is original audio tracks 
and ground truth is stereo audio where input data has been played through a speaker/amp and recorded. Model will 
accurately create a digital twin of the speaker which can be used to finetune postprocessing DSP to improve speaker 
quality. 

## Installation

Navigate to project root: `deep-speaker`

```
conda create --name ds python=3.9 
conda activate ds 
pip install -e .
```

## Configuration

`/deep-speaker/src/deepspeaker/configs/` contains several config files that allow models to be run locally or on gpu 
and use either an RNN or WaveNet-style model. 

Edit config file keys: 

```
data_dir: 
data_name: 
save_location: 
project_name:
```

## Training

Run `train_model.py` and pass `config_name` as a command line argument.

```
python ./src/deepspeaker/models/train_model.py --config_name=lstm_gpu.yaml
```

## Wandb

Right now you will get this warning:

```
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
```

You can run training without wandb logging, but I recommend creating an account and
linking it to this run.

## Project status

Development on this project has ended. 