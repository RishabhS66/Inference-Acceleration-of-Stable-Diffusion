from stable_diffusion import *
from stable_diffusion.model_loader import load_from_standard_weights
from transformers import CLIPTokenizer
from quantisation.quant_modules import TimeStepCalibratedQuantModel, QuantModel, Calibrator
from quantisation.quant_modules.utils import init_quantised_diff
from quantisation.evaluate_quantised_models import evaluate


import yaml
import sys
import wandb
import os
from dotenv import load_dotenv

def construct_exp_name(config):
  n_bits = config["QUANTISATION_PARAMS"]["WEIGHT_QUANT_PARAMS"]["n_bits"]
  name = f'{n_bits}_T_' if config['MODEL'] == 'TimeStepCalibratedQuantModel' else f'{n_bits}_S_'
  if config["QUANTISATION_PARAMS"]["USE_ACT_QUANT"]:
    name = name + 'a'

  if config["QUANTISATION_PARAMS"]["USE_WEIGHT_QUANT"]:
    name = name+'w'

  if config["CALIBRATION_PARAMS"]["USE_CALIBRATION"]:
    name = name + "_" + config["CALIBRATION_PARAMS"]["ACT_SCALE_POLICY"] + "_" + config["CALIBRATION_PARAMS"]["ACT_UPDATE_POLICY"]
    
  if len(config["QUANTISATION_PARAMS"]["QUANT_FILTERS"]) > 0:
    name += "_filt"
    
  if config["CALIBRATION_PARAMS"]["NUM_CALIBRATION_SAMPLES"] > 15:
    name+= f'({config["CALIBRATION_PARAMS"]["NUM_CALIBRATION_SAMPLES"]} steps)'
  
  return name

def calibrate(config, models, tokenizer):
  with open(config['CALIBRATION_PROMPTS_PATH'], 'r') as f:
    prompts = f.readlines()
    prompts = [line.rstrip() for line in prompts]
  
  calibrator = Calibrator(models, 
                          tokenizer, 
                          act_scale_policy=config['CALIBRATION_PARAMS']['ACT_SCALE_POLICY'], 
                          act_update_policy=config['CALIBRATION_PARAMS']['ACT_UPDATE_POLICY'])
  
  calibrator.calibrate(prompts[:config['CALIBRATION_PARAMS']['NUM_CALIBRATION_SAMPLES']], config)


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
    print('Initialising Quant Model...')
    quantised_diffuser = QuantModel(diffuser, 
                                    weight_quant_params= config['QUANTISATION_PARAMS']['WEIGHT_QUANT_PARAMS'], 
                                    act_quant_params = config['QUANTISATION_PARAMS']['WEIGHT_QUANT_PARAMS'], 
                                    quant_filters = config['QUANTISATION_PARAMS']['QUANT_FILTERS'])
  elif config['MODEL'] == 'TimeStepCalibratedQuantModel':
    assert 'K' in config['QUANTISATION_PARAMS'], "Please specify a value for K for time step calibrated quantisation"
    # assert 'TIMESTEPS' in config['QUANTISATION_PARAMS'], "Please specify a value for TIMESTEPS for time step calibrated quantisation"

    quantised_diffuser = TimeStepCalibratedQuantModel(diffuser, 
                                                      timesteps = config['GENERATION_PARAMS']['NUM_INFERENCE_STEPS'], 
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
  print("Evaluating Quantised model...")
  diff_mses, clip_score, fid_score = evaluate(models, tokenizer, config)
  print(f'Diffusion output MSEs: {diff_mses}')
  print(f'Clip Score: {clip_score}, FID Score: {fid_score}')

if __name__ == '__main__':
  # import pdb; pdb.set_trace()
  load_dotenv()
  config_path = sys.argv[1]
  config = yaml.load(open(config_path), Loader=yaml.FullLoader)

  if config["LOG_TO_WANDB"]:
    if "EXP_NAME" in config:
      exp_name = config["EXP_NAME"]
    else: 
      exp_name = construct_exp_name(config)
      config["EXP_NAME"] = exp_name
    
    wandb.login()

    wandb.init(
          project="stable-diff-quantisation",
          entity="pranjal_sri",
          config = {
              "model": config["MODEL"],
              "seed": config['SEED'],
              "quant_params": config["QUANTISATION_PARAMS"],
              "generation_params": config["GENERATION_PARAMS"],
              "calibration_params": config["CALIBRATION_PARAMS"],
              "test_params": config["TEST_PARAMS"]
          },
          name = exp_name
      )
  
  
  run(config)
  if config["LOG_TO_WANDB"]:
    wandb.finish()
