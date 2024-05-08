# Quantizing and Pruning Diffusion Models

```
Team Members: Pranjal Srivastava (ps3392), Rishabh Srivastava (rs4489)
```

## Project Overview

Diffusion models, particularly in image generation and representation learning, have demonstrated remarkable capabilities in capturing complex data distributions. However, since these models use an iterative refinement process and have multiple components, they need a lot of resources during inference. Thus, deploying these models in resource-constrained environments, such as edge devices and mobile platforms, poses challenges due to their computational intensity.

Thus, the primary objective of this project is to investigate the impact of acceleration techniques like **quantization** and **pruning** on the inference of latent diffusion models.

## Code Structure

    README.md
    code
    ├── data
    ├── stable_diffusion
    ├── quantisation
    ├── utility.py
    ├── main.py
    ├── pruning_experiment.py

The `data` folder contains the CLIP tokenizer vocabulary files, weights of the pre-trained Stable Diffusion model, and sample images used to calculate the FID score.

The `stable_diffusion` folder contains all the necessary code to create the Stable Diffusion model from the pre-trained model's weights.

The `quantisation` folder contains the modules for quantisation.

The `utility.py` file has some helper functions, and the `pruning_experiment.py` file has the code for the pruning experiments. The `main.py` file runs the pruning experiments and logs the results in Weights & Biases. 

## Dependencies

- PyTorch
- PyTorch Lightning
- Numpy
- Matplotlib
- tqdm
- Transformers
- Wandb
- Torchmetrics

## Installation and Running the Experiments
To download the pre-trained model's weights, run the following command:
```bash
mkdir data/weights
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -O ./data/weights/v1-5-pruned-emaonly.ckpt
```

To run the experiments, pruning experiments execute the command below:
```bash
python main.py
```

To run the quantisation experiments, execute the command below:
```bash
python quant_experiment_script.py config.yaml
```
## Quantisation Description
The quantisation module simulates quantisation based on parameters specified by the config.yaml.
We implement different strategies for quantisation, which can be accessed via different parameters.

The ufile `uniform_symmetric_quantiser.py` has the quantiser which can quantise inputs to a symmetric range as per the specified `n_bits`.

The quantisation module implements two models that serve as wrappers for base diffusion model. They are:

- QuantModel: Supports n-bits symmetric quantisation with possible scale strategies of `mse` and `max`
- TimeStepCalibratedQuantModel: Implements time-step aware quantisation for activations, supports possible scale strategies of `mse` and `max`. Parametrised by number of timesteps and k, a constant denoting the number o fintervals in the timesteps.

We also implement a custom Calibrator, which calibrates the quantisation modules as per the policies:

- ACT_SCALE_POLICY: How to scale the activations for quantisations. Possible values are `max` for absolute max and `mse` for Lp norm based scaling.
- ACT_UPDATE_POLICY: How to update the step parameter of the quantiser. Possible values are `maximum`, for maximum of values through the run and `momentum` for a weighted exponential average of the scales.

Refer to config.yaml for a more detailed description of possible parameters.

To load a quantised model, we can do the following:
```bash
from stable_diffusion import *
from stable_diffusion.model_loader import load_from_standard_weights
from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer("<path_to_vocab_file>",
                          merges_file="<path_to_merges_file>")
clip = CLIP()
encoder = VAE_Encoder()
decoder = VAE_Decoder()
diff = Diffusion()

state_dict = load_from_standard_weights('<path_to_weights>')
clip.load_state_dict(state_dict['clip'], strict = True)
encoder.load_state_dict(state_dict['encoder'], strict = True)
decoder.load_state_dict(state_dict['decoder'], strict = True)
diff.load_state_dict(state_dict['diffusion'], strict = True)

quantised_diff = QuantModel(diff, weight_quant_params={'n_bits': 8}, act_quant_params={'n_bits': 8})

OR

quantised_diff = TimeStepCalibratedQuantModel(diff, timesteps = 40, k = 5, weight_quant_params={'n_bits': 8}, act_quant_params={'n_bits': 8}, quant_filters = filters)
```

## Experimental Results and Observations

### Pruning
[//]: # "WandB Experiment 1 link: [Experiment 1](https://wandb.ai/hpmlcolumbia/quantization_pruning/reports/Quantization-and-Pruning--Vmlldzo3ODE1MDQ5?accessToken=5m0vlrzjcw6gyayrputy8legp1buvphuvc5esm4v6vttq9710xux9biaqx5zz5fa)"

[//]: # "WandB Experiment 1 link: [Experiment 1](https://wandb.ai/hpmlcolumbia/quantization_pruning/reports/Pruning-Experiments--Vmlldzo3ODIzMTU4?accessToken=taan0iakgdmmv6rx0herulahv1o17ik83lhz6ewdzvkgiz0y8iwdnokpcwr9br5e)"

[//]: # "WandB Experiment 2 link: [Experiment 2](https://wandb.ai/hpmlcolumbia/quantization_pruning/reports/Experiment-3-CLIP-Scores-for-Pruned-Models--Vmlldzo3ODI2NDY0?accessToken=5m2g4dq157s98aqs16mshug68igg3khb0a70cjlovydbgpmgzrvzolzgxknyxdpn)"

[//]: # "WandB Experiment 3 link: [Experiment 3](https://wandb.ai/hpmlcolumbia/quantization_pruning/reports/Pruning-Experiments-Linear-Conv-Layers-of-UNet--Vmlldzo3ODM1MjM3?accessToken=7ik20yrk4lcvah1fehubxnbk28fid9s3jxjz18qd5vrlha7xniu8pi4zzawkurya)"

L1-unstructured pruning was carried out on all the linear and convolutional layers of the UNet architecture of the Diffusion model. The results can be seen in the report below:

WandB Experiment Report link for Pruning: [Pruning Experiments](https://wandb.ai/hpmlcolumbia/quantization_pruning/reports/Quantitative-Analysis-of-Pruned-Models--Vmlldzo3ODQxMjAx?accessToken=zotsiub1f124mwqrsu346hgyqpti1iiz8fnejg8kp3xuvq9pbeq0uvwe8v984zm5)

The experiments show that pruning till 30-35% give us satisfactory results, but further pruning degrades the performance heavily.


### Quantisation

Different methods for quantisation were studied along with their generated images.

We also present a new method for wuantisation called timestep aware quantisation.
Following were the interesting takeaways:

- If we just quantize the weights, we can quantise the entire network without loss of much capabilities.

- Quantising activations is harder, we skip the first and the last layer during quantising activations as well as weights.

- Scaling  based on MSE works better than scaling based on MAX value.

- Our proposed approach, Time step aware quantisation improves the performance significantly over vanilla quantisation techniques.

More details are available at the wandb page. Following are some representative images:

<figure>
    <img src="assets/base.png"
         alt="Albuquerque, New Mexico">
    <figcaption>Base diffusion image</figcaption>
</figure>

<figure>
    <img src="assets/qw+act.png"
         alt="Albuquerque, New Mexico">
    <figcaption>Quantising weights and activations</figcaption>
</figure>

<figure>
    <img src="assets/qw+act+mse.png"
         alt="Albuquerque, New Mexico">
    <figcaption>Quantising weights and activations with mse</figcaption>
</figure>

<figure>
    <img src="assets/tqw.png"
         alt="Albuquerque, New Mexico">
    <figcaption>Time step aware</figcaption>
</figure>
