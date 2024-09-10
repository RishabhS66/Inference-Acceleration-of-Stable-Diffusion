import os
from pruning_experiment import *
from pruning_profile_exp import *

# !mkdir data
# !mkdir data/weights
# !wget  https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json -O ./data/tokenizer_vocab.json
# !wget  https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt -O ./data/tokenizer_merges.txt
# !wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -O ./data/weights/v1-5-pruned-emaonly.ckpt
# !pip install pytorch-lightning


if __name__ == '__main__':
    ''' Wandb initialization and running experiments'''

    wandb.login()
    wandb.init(
        project="quantization_pruning",
        entity="hpmlcolumbia"
    )

    run_pruning_exp()
    # run_pruning_exp(structured_pruning=True)
    # run run_pruning_profile_exp()

    wandb.finish()
