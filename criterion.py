import torch
import torch.nn as nn


def cosine_sim(im, s):#（batch,feature_dim）,(batch,feature_dim)
  '''cosine similarity between all the image and sentence pairs
  '''
  inner_prod = im.mm(s.t())#torch.mm(a, b)是a矩阵乘b。 而torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等

  im_norm = torch.sqrt((im**2).sum(1).view(-1, 1) + 1e-18)
  s_norm = torch.sqrt((s**2).sum(1).view(1, -1) + 1e-18)
  sim = inner_prod / (im_norm * s_norm)
  return sim

class ContrastiveLoss(nn.Module):
  '''compute contrastive loss
  '''
  def __init__(self, margin=0, max_violation=False, direction='bi', topk=1):
    '''Args:
      direction: i2t for negative sentence, t2i for negative image, bi for both
    '''
    super(ContrastiveLoss, self).__init__()
    self.margin = margin
    self.max_violation = max_violation
    self.direction = direction
    self.topk = topk

  def forward(self, scores, margin=None, average_batch=True):
    '''
    Args:
      scores: image-sentence score matrix, (batch, batch)
        the same row of im and s are positive pairs, different rows are negative pairs
    '''

    if margin is None:
      margin = self.margin

    batch_size = scores.size(0)
    diagonal = scores.diag().view(batch_size, 1) # positive pairs  size--> (batch, 1)

    # mask to clear diagonals which are positive pairs
    pos_masks = torch.eye(batch_size).bool().to(scores.device)

    batch_topk = min(batch_size, self.topk)# topk = 1
    if self.direction == 'i2t' or self.direction == 'bi':
      d1 = diagonal.expand_as(scores) # same collumn for im2s (negative sentence) .expand_as把一个tensor变成和函数括号内一样形状的tensor   Size-->(batch, batch)
      # compare every diagonal score to scores in its collumn
      # caption retrieval
      cost_s = (margin + scores - d1).clamp(min=0) # (batch, batch)
      cost_s = cost_s.masked_fill(pos_masks, 0)
      if self.max_violation:#hard negative
        cost_s, _ = torch.topk(cost_s, batch_topk, dim=1)# (batch, 1)  dim=1是按行取值
        cost_s = cost_s / batch_topk
        if average_batch:
          cost_s = cost_s / batch_size
      else:
        if average_batch:
          cost_s = cost_s / (batch_size * (batch_size - 1))
      cost_s = torch.sum(cost_s)

    if self.direction == 't2i' or self.direction == 'bi':
      d2 = diagonal.t().expand_as(scores) # same row for s2im (negative image)
      # compare every diagonal score to scores in its row
      cost_im = (margin + scores - d2).clamp(min=0)
      cost_im = cost_im.masked_fill(pos_masks, 0)
      if self.max_violation:
        cost_im, _ = torch.topk(cost_im, batch_topk, dim=0)
        cost_im = cost_im / batch_topk
        if average_batch:
          cost_im = cost_im / batch_size
      else:
        if average_batch:
          cost_im = cost_im / (batch_size * (batch_size - 1))
      cost_im = torch.sum(cost_im)

    if self.direction == 'i2t':
      return cost_s
    elif self.direction == 't2i':
      return cost_im
    else:
      return cost_s + cost_im
