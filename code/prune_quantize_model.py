from stable_diffusion import *
from stable_diffusion.model_loader import load_from_standard_weights
from transformers import CLIPTokenizer
from quantisation.quant_modules import TimeStepCalibratedQuantModel, QuantModel, Calibrator
from quantisation.quant_modules.utils import init_quantised_diff
import torch
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from functools import partial
import torchvision.transforms.functional as F
import numpy as np
import yaml
import sys
import wandb
from dotenv import load_dotenv
import torch.nn.utils.prune as prune
from torch import nn
import os


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    clip_score = clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


def calculate_fid_score(fake_images, real_images):
    fake_images = torch.from_numpy(fake_images).permute(0, 3, 1, 2) / 255.0
    real_images = torch.from_numpy(real_images).permute(0, 3, 1, 2) / 255.0
    fake_images = F.center_crop(fake_images, (256, 256))
    real_images = F.center_crop(real_images, (256, 256))

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return round(float(fid.compute()), 4)


def test_clip_fid_score(models, tokenizer, config):
    with open(config["TEST_PROMPTS_PATH"]) as f:
        prompts = f.readlines()
        prompts = [line.rstrip() for line in prompts]
    prompts = prompts[:config["TEST_PARAMS"]["NUM_TEST_SAMPLES"]]
    device = torch.device("cuda" if torch.cuda.is_available() and config['USE_CUDA'] else "cpu")

    uncond_prompt = config['GENERATION_PARAMS']['UNCOND_PROMPT']  # Also known as negative prompt
    do_cfg = config['GENERATION_PARAMS']['DO_CFG']
    cfg_scale = config['GENERATION_PARAMS']['CFG_SCALE']
    strength = config['GENERATION_PARAMS']['STRENGTH']
    num_inference_steps = config['GENERATION_PARAMS']['NUM_INFERENCE_STEPS']
    seed = config['SEED'] if 'SEED' in config else None
    sampler = config['GENERATION_PARAMS']['SAMPLER_NAME']
    quantized_images = []

    models['diffusion'].set_use_quant(use_act_quant=config['QUANTISATION_PARAMS']['USE_ACT_QUANT'],
                                      use_weight_quant=config['QUANTISATION_PARAMS']['USE_WEIGHT_QUANT'])
    for prompt in prompts:
        quantized_image = generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=None,
            strength=strength,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=models,
            device=device,
            idle_device="cpu",
            tokenizer=tokenizer,
        )
        quantized_images.append(quantized_image)

    models['diffusion'].set_use_quant(use_act_quant=False, use_weight_quant=False)
    real_images = []
    for prompt in prompts:
        real_image = generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=None,
            strength=strength,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=models,
            device=device,
            idle_device="cpu",
            tokenizer=tokenizer,
        )
        real_images.append(real_image)

    clip_score = calculate_clip_score(np.array(quantized_images), prompts)
    fid_score = calculate_fid_score(np.array(quantized_images), np.array(real_images))

    if config["LOG_TO_WANDB"]:
        table = wandb.Table(columns=["Prompt", "Quantised Image", "Unquantised Image"])

        for prompt, q_img, unq_img in zip(prompts, quantized_images, real_images):
            table.add_data(prompt,
                           wandb.Image(q_img),
                           wandb.Image(unq_img))

        wandb.log({'Pruned Model - 0.2, Quant Model - Max': table})
        wandb.log({
            'CLIP Score': clip_score,
            'FID Score': fid_score
        })

    return clip_score, fid_score


def calibrate(config, models, tokenizer):
    with open(config['CALIBRATION_PROMPTS_PATH'], 'r') as f:
        prompts = f.readlines()
        prompts = [line.rstrip() for line in prompts]

    calibrator = Calibrator(models,
                            tokenizer,
                            act_scale_policy=config['CALIBRATION_PARAMS']['ACT_SCALE_POLICY'],
                            act_update_policy=config['CALIBRATION_PARAMS']['ACT_UPDATE_POLICY'])

    calibrator.calibrate(prompts[:config['CALIBRATION_PARAMS']['NUM_CALIBRATION_SAMPLES']], config)


def run(config, structured_pruning=False):
    print('Loading models...')
    tokenizer = CLIPTokenizer(config['CLIP_VOCAB_PATH'],
                              merges_file=config['CLIP_MERGES_PATH'])
    clip = CLIP()
    encoder = VAE_Encoder()
    decoder = VAE_Decoder()
    diffuser = Diffusion()

    state_dict = load_from_standard_weights(config['PRETRAINED_WEIGHTS_PATH'])
    clip.load_state_dict(state_dict['clip'], strict=True)
    encoder.load_state_dict(state_dict['encoder'], strict=True)
    decoder.load_state_dict(state_dict['decoder'], strict=True)
    diffuser.load_state_dict(state_dict['diffusion'], strict=True)

    p = 0.2

    for module_name, module in diffuser.named_modules():
        if 'unet' in module_name and (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
            # If the module is a convolutional or linear layer, prune it
            if structured_pruning:
                prune.ln_structured(module, name='weight', amount=p, n=2, dim=0)
            else:
                prune.l1_unstructured(module, name='weight', amount=p)
            prune.remove(module, 'weight')
            if module.bias is not None and not structured_pruning:
                prune.l1_unstructured(module, name='bias', amount=p)
                prune.remove(module, 'bias')
    print('Model Pruned!')

    if config['MODEL'] == 'QuantModel':
        print('Initialising Quant Model...')
        quantised_diffuser = QuantModel(diffuser,
                                        weight_quant_params=config['QUANTISATION_PARAMS']['WEIGHT_QUANT_PARAMS'],
                                        act_quant_params=config['QUANTISATION_PARAMS']['WEIGHT_QUANT_PARAMS'],
                                        quant_filters=config['QUANTISATION_PARAMS']['QUANT_FILTERS'])
    elif config['MODEL'] == 'TimeStepCalibratedQuantModel':
        assert 'K' in config[
            'QUANTISATION_PARAMS'], "Please specify a value for K for time step calibrated quantisation"
        # assert 'TIMESTEPS' in config['QUANTISATION_PARAMS'], "Please specify a value for TIMESTEPS for time step calibrated quantisation"

        quantised_diffuser = TimeStepCalibratedQuantModel(diffuser,
                                                          timesteps=config['GENERATION_PARAMS']['NUM_INFERENCE_STEPS'],
                                                          k=config['QUANTISATION_PARAMS']['K'],
                                                          weight_quant_params=config['QUANTISATION_PARAMS'][
                                                              'WEIGHT_QUANT_PARAMS'],
                                                          act_quant_params=config['QUANTISATION_PARAMS'][
                                                              'ACT_QUANT_PARAMS'],
                                                          quant_filters=config['QUANTISATION_PARAMS']['QUANT_FILTERS'])
    models = {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": quantised_diffuser,
    }

    if config['CALIBRATION_PARAMS']['USE_CALIBRATION']:
        print('Calibrating...')
        assert 'CALIBRATION_PROMPTS_PATH' in config, "Please specify a path to calibration prompts"
        assert 'TEST_PROMPTS_PATH' in config, "Please specify a path to test prompts"
        assert config['QUANTISATION_PARAMS']['USE_ACT_QUANT'], "Please enable act quantisation for calibration"
        assert config['QUANTISATION_PARAMS']['USE_WEIGHT_QUANT'], "Please enable weight quantisation for calibration"
        init_quantised_diff(quantised_diffuser)
        calibrate(config, models, tokenizer)
    print("Evaluating Quantised model...")
    clip_score, fid_score = test_clip_fid_score(models, tokenizer, config)
    print(f'Clip Score: {clip_score}, FID Score: {fid_score}')


if __name__ == '__main__':
    load_dotenv()
    config_path = sys.argv[1]
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    if config["LOG_TO_WANDB"]:
        n_bits = config["QUANTISATION_PARAMS"]["WEIGHT_QUANT_PARAMS"]["n_bits"]
        name = f'{n_bits}_TIME_CAL_' if config['MODEL'] == 'TimeStepCalibratedQuantModel' else f'{n_bits}_SIMPLE_'
        if config["QUANTISATION_PARAMS"]["USE_ACT_QUANT"]:
            name = name + 'a'

        if config["QUANTISATION_PARAMS"]["USE_WEIGHT_QUANT"]:
            name = name + 'q'

        if config["CALIBRATION_PARAMS"]["USE_CALIBRATION"]:
            name = name + "_" + config["CALIBRATION_PARAMS"]["ACT_SCALE_POLICY"] + "_" + config["CALIBRATION_PARAMS"][
                "ACT_UPDATE_POLICY"]

        if len(config["QUANTISATION_PARAMS"]["QUANT_FILTERS"]) > 0:
            name += "_filt"

        if config["CALIBRATION_PARAMS"]["NUM_CALIBRATION_SAMPLES"] > 15:
            name += f'({config["CALIBRATION_PARAMS"]["NUM_CALIBRATION_SAMPLES"]} steps)'

        os.environ["WANDB_API_KEY"] = "918907cbd509a54b20e48c35485f867aab3e59df"
        os.environ["WANDB_HOST"] = "rishabhs"
        wandb.login()

        wandb.init(
            project="quantization_pruning",
            entity="hpmlcolumbia",
            config={
                "model": config["MODEL"],
                "seed": config['SEED'],
                "quant_params": config["QUANTISATION_PARAMS"],
                "generation_params": config["GENERATION_PARAMS"],
                "calibration_params": config["CALIBRATION_PARAMS"],
                "test_params": config["TEST_PARAMS"]
            },
            name=name
        )

    run(config)
    if config["LOG_TO_WANDB"]:
        wandb.finish()
