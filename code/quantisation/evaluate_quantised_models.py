import sys
import gc
sys.path.insert(0, '../')

import torch
from stable_diffusion import *
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from functools import partial
import torchvision.transforms.functional as F
import numpy as np
import wandb

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    clip_score = clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def calculate_fid_score(fake_images, real_images):
  fake_images = torch.from_numpy(fake_images).permute(0, 3, 1, 2)/255.0
  real_images = torch.from_numpy(real_images).permute(0, 3, 1, 2)/255.0
  fake_images = F.center_crop(fake_images, (256, 256))
  real_images = F.center_crop(real_images, (256, 256))

  fid = FrechetInceptionDistance(normalize=True)
  fid.update(real_images, real=True)
  fid.update(fake_images, real=False)
  return round(float(fid.compute()), 4)

def evaluate(models, tokenizer, config):
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
  real_images = []
  diffusion_mses = np.zeros(config['GENERATION_PARAMS']['NUM_INFERENCE_STEPS'])

  for prompt in prompts:
    models['diffusion'].set_use_quant(use_act_quant = config['QUANTISATION_PARAMS']['USE_ACT_QUANT'], 
                                      use_weight_quant = config['QUANTISATION_PARAMS']['USE_WEIGHT_QUANT'])
    quantized_diff_outputs = []
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
                diff_output_tracer = quantized_diff_outputs
            )
    quantized_images.append(quantized_image)
    
    unquantized_diff_output = []
    
    models['diffusion'].set_use_quant(use_act_quant = False, use_weight_quant = False)
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
                diff_output_tracer = unquantized_diff_output
            )
    real_images.append(real_image)

    for i in range(len(diffusion_mses)):
      mse = torch.mean(torch.square(unquantized_diff_output[i] - quantized_diff_outputs[i])).item()
      diffusion_mses[i] += mse

    del unquantized_diff_output
    del quantized_diff_outputs
    gc.collect()
  
  clip_score = calculate_clip_score(np.array(quantized_images), prompts)
  fid_score = calculate_fid_score(np.array(quantized_images), np.array(real_images))
  diffusion_mses /= config["TEST_PARAMS"]["NUM_TEST_SAMPLES"]

  if config["LOG_TO_WANDB"]:
    data = [[t, mse] for t,mse in enumerate(diffusion_mses)]
    table = wandb.Table(data=data, columns=["timestep", "MSE"])
    wandb.log(
    {
        "MSE for diffusion outputs": wandb.plot.line(
            table, "timestep", "MSE", title="MSE difference (Quantised v Unquantised)"
        )
    }
    )

    table = wandb.Table(columns = ["Experiment", "Prompt", "Quantised Image", "Unquantised Image"])

    for prompt, q_img, unq_img in zip(prompts, quantized_images, real_images):
      table.add_data(
                    config["EXP_NAME"],
                    prompt, 
                    wandb.Image(q_img),
                    wandb.Image(unq_img))
    
    wandb.log({'Comparison table': table})
    wandb.log({
        'CLIP Score': clip_score,
        'FID Score': fid_score
      })
  
  return diffusion_mses, clip_score, fid_score