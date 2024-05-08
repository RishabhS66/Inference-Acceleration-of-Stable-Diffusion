import gc
import torch

from .quant_wrappers import QuantWrapper, TimeStepCalibratedQuantWrapper
import sys
sys.path.insert(0, '../../')
from stable_diffusion.pipeline import generate

class Calibrator:
  def __init__(self, models, tokeniser, 

               act_update_policy = 'momentum',
               act_scale_policy = 'max'):
    self.models = models
    self.quantised_diff = models['diffusion']
    self.quantised_diff.set_use_quant(use_weight_quant = True, use_act_quant = True)
    self.tokenizer = tokeniser
    self.device = 'cuda'
    self.idle_device = 'cpu'
    self.act_update_policy = act_update_policy
    self.act_scale_policy = act_scale_policy



  def calibrate(self, prompts, config):
    for n,m in self.quantised_diff.named_modules():
      if isinstance(m, (QuantWrapper, TimeStepCalibratedQuantWrapper)):
        m.set_act_scale_method(self.act_scale_policy)
        m.set_act_update_policy(self.act_update_policy)
        m.reset_act_quantiser_inited()

    uncond_prompt = config['GENERATION_PARAMS']['UNCOND_PROMPT']  # Also known as negative prompt
    do_cfg = config['GENERATION_PARAMS']['DO_CFG']
    cfg_scale = config['GENERATION_PARAMS']['CFG_SCALE']
    strength = config['GENERATION_PARAMS']['STRENGTH']
    num_inference_steps = config['GENERATION_PARAMS']['NUM_INFERENCE_STEPS']
    seed = config['SEED'] if 'SEED' in config else None

    models = self.models
    sampler = config['GENERATION_PARAMS']['SAMPLER_NAME']

    for prompt in prompts:
      generate(
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
          device='cuda',
          idle_device="cpu",
          tokenizer=self.tokenizer,
      )
      gc.collect()
      torch.cuda.empty_cache()

    for n,m in self.quantised_diff.named_modules():
      if isinstance(m, (QuantWrapper, TimeStepCalibratedQuantWrapper)):
        m.set_act_update_policy(None)

