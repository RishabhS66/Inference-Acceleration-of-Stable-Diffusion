

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_wrappers import QuantWrapper, TimeStepCalibratedQuantWrapper

class QuantModel(nn.Module):

    def __init__(self, model: nn.Module,
                 weight_quant_params: dict = {},
                 act_quant_params: dict = {},
                 quant_filters = [], **kwargs,):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        if hasattr(model, 'in_channels'):
          self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        # self.specials = get_specials(act_quant_params['leaf_param'])
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params,
                                   quant_filters, **kwargs)
        # self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module, weight_quant_params, act_quant_params, quant_filters, parent_name = '', **kwargs):
      for n, m in module.named_children():
        name = parent_name +'.'+ n
        filter_flag = False
        for filter in quant_filters:
          if filter in name:
            filter_flag = True
            break
        if filter_flag:
          continue

        if isinstance(m, (nn.Conv2d, nn.Linear)):
          setattr(module, n, QuantWrapper(
                    m, weight_quant_params, act_quant_params, **kwargs))
        else:
          self.quant_module_refactor(m, weight_quant_params, act_quant_params, quant_filters, parent_name = name, **kwargs)



    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def set_use_quant(self, use_weight_quant = False, use_act_quant = False):
      # import pdb; pdb.set_trace()
      for n,m in self.model.named_modules():
        if isinstance(m, QuantWrapper):
          m.set_use_quant(use_weight_quant, use_act_quant)


class TimeStepCalibratedQuantModel(nn.Module):

    def __init__(self, model: nn.Module,
                 timesteps, k,
                 weight_quant_params: dict = {},
                 act_quant_params: dict = {},
                 quant_filters = [], **kwargs,):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.k = k

        self.sm_abit = kwargs.get('sm_abit', 8)
        if hasattr(model, 'in_channels'):
          self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        # self.specials = get_specials(act_quant_params['leaf_param'])
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params,
                                   quant_filters, **kwargs)
        # self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module, weight_quant_params, act_quant_params, quant_filters, parent_name = '', **kwargs):
      for n, m in module.named_children():
        name = parent_name +'.'+ n
        filter_flag = False
        for filter in quant_filters:
          if filter in name:
            filter_flag = True
            break
        if filter_flag:
          continue

        if isinstance(m, (nn.Conv2d, nn.Linear)):
          setattr(module, n, TimeStepCalibratedQuantWrapper(
                    m, self.timesteps, self.k, weight_quant_params, act_quant_params, **kwargs))

        else:
          self.quant_module_refactor(m, weight_quant_params, act_quant_params, quant_filters, parent_name = name, **kwargs)



    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def set_use_quant(self, use_weight_quant = False, use_act_quant = False):
      # import pdb; pdb.set_trace()
      for n,m in self.model.named_modules():
        if isinstance(m, TimeStepCalibratedQuantWrapper):
          m.set_use_quant(use_weight_quant, use_act_quant)


    def set_current_timestep(self, timestep):
      for n, m in self.model.named_modules():
        if isinstance(m, TimeStepCalibratedQuantWrapper):
          m.set_current_timestep(timestep)