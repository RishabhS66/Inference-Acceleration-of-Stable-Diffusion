import torch
import torch.nn as nn
import warnings

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

class UniformSymmetricQuantiser(nn.Module):
  def __init__(self, n_bits, channel_wise=True, scale_method='max', activation_mom = 0.95):
    super(UniformSymmetricQuantiser, self).__init__()
    assert 2 <= n_bits <= 8, 'bitwidth not supported'
    self.n_bits = n_bits
    self.n_levels = 2 ** (self.n_bits - 1)
    self.channel_wise = channel_wise
    self.scale_method = scale_method
    self.act_momentum = activation_mom
    self.register_buffer('delta', None)
    self.register_buffer('x_absmax', None)
    self.inited = False

  def set_scale_method(self, scale_method):
    self.scale_method = scale_method

  def forward(self, x):
    if not self.inited:
      self.delta, self.x_absmax = self.init_quantization_scale(x, self.channel_wise)
      self.inited = True

    x_quant = round_ste(x/self.delta)
    x_quant = torch.clamp(x_quant, -(self.n_levels-1), self.n_levels-1)
    return x_quant*self.delta

  def dequantize(self, x):
    assert self.inited, 'quantization not initialized'
    x_dequant = (x) * self.delta
    return x_dequant

  def _quantize_for_mse(self, x, x_absmax_l):
    delta = x_absmax_l / (self.n_levels - 1)
    x_q = round_ste(x/delta)
    x_q = torch.clamp(x_q, -(self.n_levels-1), self.n_levels-1)
    return x_q*delta

  def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = None):
    # import pdb; pdb.set_trace()
    if act_range_momentum is None:
      act_range_momentum = self.act_momentum

    x_d = x.data
    delta, x_max = self.init_quantization_scale(x_d, False)
    if not self.inited:
      self.x_absmax = x_max
      self.inited = True
    else:
      if act_range_momentum < 0.0:
        self.x_absmax = torch.maximum(self.x_absmax, x_max)
      else:
        self.x_absmax = self.x_absmax * act_range_momentum + x_max * (1 - act_range_momentum)
    delta = self.x_absmax / (self.n_levels - 1)
    delta = torch.clamp(delta, min=1e-8)
    self.delta = torch.tensor(delta).type_as(x)

  def init_quantization_scale(self, x: torch.Tensor, channel_wise = False):
    delta = None
    if channel_wise:
      x_clone = x.clone().detach()
      n_channels = x_clone.shape[0]
      delta = torch.zeros(n_channels).type_as(x)
      x_absmax = torch.zeros(n_channels).type_as(x)
      for c in range(n_channels):
        delta[c], x_absmax[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)

      if len(x.shape) == 4:
        delta = delta.view(-1, 1, 1, 1)
        x_absmax = x_absmax.view(-1, 1, 1, 1)
      elif len(x.shape) == 3:
        delta = delta.view(-1, 1, 1)
        x_absmax = x_absmax.view(-1, 1, 1)
      else:
        delta = delta.view(-1, 1)
        x_absmax = x_absmax.view(-1, 1)

    else:
      if 'max' in self.scale_method:
        x_absmax = torch.max(torch.abs(x))
        if 'scale' in self.scale_method:
          x_absmax = x_absmax * (self.n_bits + 2) / 8

        delta = x_absmax / (self.n_levels-1)
        if delta < 1e-8:
          warnings.warn('Quantization range close to zero')
          delta = 1e-8

      elif self.scale_method == 'mse':
        x_absmax = torch.max(torch.abs(x))
        x_absmax_best = x_absmax
        best_score = 1e+10
        for i in range(30):
          x_absmax_l = x_absmax * (1.0 - (i * 0.01))
          x_q = self._quantize_for_mse(x, x_absmax_l)
          # L_p norm minimization as described in LAPQ
          # https://arxiv.org/abs/1911.07190
          score = lp_loss(x, x_q, p=2.4, reduction='all')
          if score < best_score:
            best_score = score
            delta = x_absmax_l / (self.n_levels - 1)
            x_absmax_best = x_absmax_l

        x_absmax = x_absmax_best
      else:
        raise NotImplementedError

    return delta.type_as(x), x_absmax.type_as(x)

