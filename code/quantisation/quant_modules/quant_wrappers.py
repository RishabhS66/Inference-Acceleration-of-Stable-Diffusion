import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .uniform_symmetric_quantiser import UniformSymmetricQuantiser

class QuantWrapper(nn.Module):
  def __init__(self, org_module, weight_quant_params = {}, act_quant_params = {}, act_quant_update_policy = None):
    super(QuantWrapper, self).__init__()
    # Always use tensor wise quantisation for activations
    act_quant_params['channel_wise'] = False


    self.act_quantiser = UniformSymmetricQuantiser(**act_quant_params)
    self.weight_quantiser = UniformSymmetricQuantiser(**weight_quant_params)
    self.org_module = org_module
    self.use_weight_quant = False
    self.use_act_quant = False
    self.act_quant_update_policy = act_quant_update_policy

    if isinstance(org_module, nn.Conv2d):
      self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
      self.fwd_func = F.conv2d
    elif isinstance(org_module, nn.Conv1d):
      self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
      self.fwd_func = F.conv1d
    else:
      self.fwd_kwargs = dict()
      self.fwd_func = F.linear

    self.weight = nn.Parameter(org_module.weight.data.clone())
    self.org_weight = org_module.weight.data.clone()
    if org_module.bias is not None:
      self.bias = nn.Parameter(org_module.bias.data.clone())
      self.org_bias = org_module.bias.data.clone()
    else:
      self.bias = None
      self.org_bias = None

  def set_act_update_policy(self, act_quant_update_policy):
    self.act_quant_update_policy = act_quant_update_policy

  def set_act_scale_method(self, scale_method):
    self.act_quantiser.set_scale_method(scale_method)

  def reset_act_quantiser_inited(self):
    self.act_quantiser.inited = False

  def forward(self, x):
    # import pdb; pdb.set_trace()
    weight = self.weight
    bias = self.bias
    if self.use_weight_quant:
      weight = self.weight_quantiser(weight)

    if self.use_act_quant:
      if self.act_quant_update_policy is not None:
        if self.act_quant_update_policy == 'momentum':
          self.act_quantiser.act_momentum_update(x)
        elif self.act_quant_update_policy == 'maximum':
          self.act_quantiser.act_momentum_update(x, -1)
        else:
          raise NotImplementedError

      x = self.act_quantiser(x)


    x = self.fwd_func(x, weight, bias, **self.fwd_kwargs)
    return x

  def set_use_quant(self, use_weight_quant = False, use_act_quant = False):
    self.use_weight_quant = use_weight_quant
    self.use_act_quant = use_act_quant


class TimeStepCalibratedQuantWrapper(nn.Module):
  def __init__(self, org_module, timesteps, k = 5, weight_quant_params = {}, act_quant_params = {}, act_quant_update_policy = None):
    super().__init__()
    # Always use tensor wise quantisation for activations
    act_quant_params['channel_wise'] = False

    self.weight_quantiser = UniformSymmetricQuantiser(**weight_quant_params)

    self.act_quantisers = nn.ModuleList()
    self.timesteps_per_act_quantiser = math.ceil(timesteps/k)
    self.n_quantisers = math.ceil(timesteps/self.timesteps_per_act_quantiser)
    for i in range(self.n_quantisers):
      self.act_quantisers.append(UniformSymmetricQuantiser(**act_quant_params))

    self.org_module = org_module
    self.use_weight_quant = False
    self.use_act_quant = False
    self.act_quant_update_policy = act_quant_update_policy

    if isinstance(org_module, nn.Conv2d):
      self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
      self.fwd_func = F.conv2d
    elif isinstance(org_module, nn.Conv1d):
      self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
      self.fwd_func = F.conv1d
    else:
      self.fwd_kwargs = dict()
      self.fwd_func = F.linear

    self.weight = nn.Parameter(org_module.weight.data.clone())
    self.org_weight = org_module.weight.data.clone()
    if org_module.bias is not None:
      self.bias = nn.Parameter(org_module.bias.data.clone())
      self.org_bias = org_module.bias.data.clone()
    else:
      self.bias = None
      self.org_bias = None

    self.timestep_for_calibration = None

  def set_current_timestep(self, timestep):
    self.timestep_for_calibration = timestep

  def set_act_update_policy(self, act_quant_update_policy):
    self.act_quant_update_policy = act_quant_update_policy

  def set_act_scale_method(self, scale_method):
    for act_quantiser in self.act_quantisers:
      act_quantiser.set_scale_method(scale_method)

  def reset_act_quantiser_inited(self):
    for act_quantiser in self.act_quantisers:
      act_quantiser.inited = False

  def forward(self, x):
    weight = self.weight
    bias = self.bias
    if self.use_weight_quant:
      weight = self.weight_quantiser(weight)

    if self.use_act_quant:
      assert self.timestep_for_calibration is not None, 'Please set the timestep using .set_current_timestep() for using timestep calibrated quantisation'
      act_quantiser = self.act_quantisers[self.timestep_for_calibration//self.timesteps_per_act_quantiser]
      if self.act_quant_update_policy is not None:
        if self.act_quant_update_policy == 'momentum':
          act_quantiser.act_momentum_update(x)
        elif self.act_quant_update_policy == 'maximum':
          act_quantiser.act_momentum_update(x, -1)
        else:
          raise NotImplementedError

      x = act_quantiser(x)

    x = self.fwd_func(x, weight, bias, **self.fwd_kwargs)
    return x

  def set_use_quant(self, use_weight_quant = False, use_act_quant = False):
    self.use_weight_quant = use_weight_quant
    self.use_act_quant = use_act_quant

