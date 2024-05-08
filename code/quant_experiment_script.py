import wandb
from utility import *
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

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
def calculate_clip_score(images, prompts):
    clip_score = clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def calculate_fid_score(fake_images, real_images):
  fake_images = torch.from_numpy(fake_images).permute(0, 3, 1, 2)/255.0
  fake_images = F.center_crop(fake_images, (256, 256))

  fid = FrechetInceptionDistance(normalize=True)
  fid.update(real_images, real=True)
  fid.update(fake_images, real=False)
  return round(float(fid.compute()), 4)

def test_clip_fid_score(models, tokenizer, config):
  with open(config["TEST_PROMPTS_PATH"]) as f:
    prompts = f.readlines()
    prompts = [line.rstrip() for line in prompts]
  device = torch.device("cuda" if torch.cuda.is_available() and config['USE_CUDA'] else "cpu")
  
  uncond_prompt = config['GENERATION_PARAMS']['UNCOND_PROMPT']  # Also known as negative prompt
  do_cfg = config['GENERATION_PARAMS']['DO_CFG']
  cfg_scale = config['GENERATION_PARAMS']['CFG_SCALE']
  strength = config['GENERATION_PARAMS']['STRENGTH']
  num_inference_steps = config['GENERATION_PARAMS']['NUM_INFERENCE_STEPS']
  seed = config['SEED'] if 'SEED' in config else None
  sampler = config['GENERATION_PARAMS']['SAMPLER_NAME']
  quantized_images = []
  quantized_images_fid = []
  

  models['diffusion'].set_use_quant(use_act_quant = config['QUANTISATION_PARAMS']['USE_ACT_QUANT'], use_weight_quant = config['QUANTISATION_PARAMS']['USE_WEIGHT_QUANT'])

  table = wandb.Table(columns=["Prompt", "Image Generated"])
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
    table.add_data(prompt, wandb.Image(quantized_image))

  wandb.log({"Images Generated from Quantized Model": table})

  prompts_fid = [
      "cassette player",
      "chainsaw",
      "chainsaw",
      "church",
      "gas pump",
      "gas pump",
      "gas pump",
      "parachute",
      "parachute",
      "tench",
  ]
  real_images = get_real_images()
  for prompt in prompts_fid:
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
    quantized_images_fid.append(quantized_image)

  
  clip_score = calculate_clip_score(np.array(quantized_images), prompts)
  fid_score = calculate_fid_score(np.array(quantized_images_fid), real_images)
  return clip_score, fid_score

def calibrate(config, models, tokenizer):
  with open(config['CALIBRATION_PROMPTS_PATH'], 'r') as f:
    prompts = f.readlines()
    prompts = [line.rstrip() for line in prompts]
  
  calibrator = Calibrator(models, 
                          tokenizer, 
                          act_scale_policy=config['CALIBRATION_PARAMS']['ACT_SCALE_POLICY'], 
                          act_update_policy=config['CALIBRATION_PARAMS']['ACT_UPDATE_POLICY'])
  
  calibrator.calibrate(prompts[:config['CALIBRATION_PARAMS']['NUM_CALIBRATION_SAMPLES']])


def run(config):
  print('Loading models...')
  tokenizer = CLIPTokenizer(config['CLIP_VOCAB_PATH'],
                            merges_file=config['CLIP_MERGES_PATH'])
  clip = CLIP()
  encoder = VAE_Encoder()
  decoder = VAE_Decoder()
  diffuser = Diffusion()

  state_dict = load_from_standard_weights(config['PRETRAINED_WEIGHTS_PATH'])
  clip.load_state_dict(state_dict['clip'], strict = True)
  encoder.load_state_dict(state_dict['encoder'], strict = True)
  decoder.load_state_dict(state_dict['decoder'], strict = True)
  diffuser.load_state_dict(state_dict['diffusion'], strict = True)

  if config['MODEL'] == 'QuantModel':
    print('Initializing Quant Model')
    quantised_diffuser = QuantModel(diffuser, 
                                    weight_quant_params= config['QUANTISATION_PARAMS']['WEIGHT_QUANT_PARAMS'], 
                                    act_quant_params = config['QUANTISATION_PARAMS']['WEIGHT_QUANT_PARAMS'], 
                                    quant_filters = config['QUANTISATION_PARAMS']['QUANT_FILTERS'])
  elif config['MODEL'] == 'TimeStepCalibratedQuantModel':
    assert 'K' in config['QUANTISATION_PARAMS'], "Please specify a value for K for time step calibrated quantisation"
    assert 'TIMESTEPS' in config['QUANTISATION_PARAMS'], "Please specify a value for TIMESTEPS for time step calibrated quantisation"

    quantised_diffuser = TimeStepCalibratedQuantModel(diffuser, 
                                                      timesteps = config['QUANTISATION_PARAMS']['TIMESTEPS'], 
                                                      k = config['QUANTISATION_PARAMS']['K'], 
                                                      weight_quant_params=config['QUANTISATION_PARAMS']['WEIGHT_QUANT_PARAMS'], 
                                                      act_quant_params=config['QUANTISATION_PARAMS']['ACT_QUANT_PARAMS'], 
                                                      quant_filters = config['QUANTISATION_PARAMS']['QUANT_FILTERS'])
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

  clip_score, fid_score = test_clip_fid_score(models, tokenizer, config)
  print(f'Clip Score: {clip_score}, FID Score: {fid_score}')
  table = wandb.Table(columns=["Experiment Description", "CLIP Score", "FID Score"])
  experiment_type = f'Quantization of Base Diffusion Model - type {config["MODEL"]}'
  table.add_data(experiment_type, clip_score, fid_score)
  wandb.log({"Results after Quantization": table})

if __name__ == '__main__':
  config_path = sys.argv[1]
  config = yaml.load(open(config_path), Loader=yaml.FullLoader)

  os.environ["WANDB_API_KEY"] = "918907cbd509a54b20e48c35485f867aab3e59df"
  os.environ["WANDB_HOST"] = "rishabhs"

  wandb.login()
  wandb.init(
      project="quantization_pruning",
      entity="hpmlcolumbia"
  )

  run(config)

  wandb.finish()
