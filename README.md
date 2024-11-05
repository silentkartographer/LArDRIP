![alt text](https://github.com/silentkartographer/LArDRIP/blob/main/logo.png?raw=true)

# LArDRIP
This package is a work-in-progress for generating inferred signals in dead regions of a DUNE-ND-like liquid argon time projection chamber (LArTPC).

It is a generative Sparse 3D Convolutional Neural Network. 

## View Rendered Notebook Outputs

[View the plotting notebook with outputs on nbviewer](https://nbviewer.org/github/silentkartographer/LArDRIP/blob/main/plotting/plotting.ipynb)

Examples and preprint coming soon...

## Instructions
1. Data Preparation
Place the simulation .npz files for muon-like tracks in the /data directory. A link to download these files is provided in a text file located in /data. Instructions on generating these simulation files will be added here in the future.

2. Running the Model
To run the model, use the following command:

```bash python3 /ME_model/model_twoinactive_new.py ```

If you need to resume training from a specific epoch (for instance, after an interruption), use the following command:

```bash python3 /ME_model/model_twoinactive_new.py --resume --checkpoint_path /path/to/checkpoint/checkpoint_epoch_{number}.pth --wandb_run_id {wandb_id} ```

{number}: The epoch from which to resume training (the model will start from this epoch + 1).
{wandb_id}: The Weights and Biases ID for tracking the training session.

