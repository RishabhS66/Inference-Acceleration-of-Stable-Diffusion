import torch

def init_quantised_diff(quantised_diff, state_dict = None):
  quantised_diff.set_use_quant(use_weight_quant = True, use_act_quant = True)
  if hasattr(quantised_diff, 'set_current_timestep'):
    quantised_diff.set_current_timestep(0)

  ip = torch.randn((1, 4, 512//8, 512//8))
  context = torch.randn((1, 77, 768))
  time = torch.randn((1, 320))
  with torch.no_grad():
    quantised_diff(ip, context, time)

  if state_dict is not None:
    quantised_diff.load_state_dict(state_dict, strict = False)