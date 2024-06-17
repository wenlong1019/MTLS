import torch
import torch.nn as nn
import torch.nn.functional as F


def dist_similarity(textual_output, visual_output):
    visual_output = F.log_softmax(visual_output, dim=-1)
    textual_output = F.softmax(textual_output, dim=-1)
    similarity = F.kl_div(visual_output, textual_output, reduction='sum')
    return similarity


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        assert temperature > 0
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cosine_similarity(x, y) / self.temperature


def spat_similarity(query, positive, temperature=0.2, reduction='mean'):
    # Normalize to unit vectors
    query, positive = normalize(query, positive)
    m_eye = torch.eye(query.shape[0])
    # 1.self
    logits_s = query @ transpose(query)
    logits_1 = logits_s[(1 - m_eye).bool()].reshape([len(logits_s), -1])
    # 2.positive
    logits_2 = query @ transpose(positive)
    # Positive keys are the entries on the diagonal
    logits = torch.cat([logits_2, logits_1], dim=1)
    labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def cross_entropy_loss(label_scores, batch):
    mask_matrix = torch.zeros([len(batch.seq_lengths), batch.seq_lengths[0]]).bool()
    targets = list(batch.targets)
    for b in range(len(batch.seq_lengths)):
        mask_matrix[b, 1:batch.seq_lengths[b]] = True
        if len(targets[b]) != batch.seq_lengths[b] - 1:
            targets[b] = targets[b][:batch.seq_lengths[b] - 1]
    scores = label_scores[mask_matrix]
    gold_targets = torch.cat(targets).cuda().detach().long()
    loss = F.cross_entropy(scores, gold_targets)
    return loss


def text_cls_loss(label_scores, batch):
    scores = label_scores[:, 0, :]
    gold_targets = torch.tensor(batch.targets).cuda().detach().long()
    loss = F.cross_entropy(scores, gold_targets)
    return loss


def focalLoss(label_scores, batch):
    alpha = 0.25
    gamma = 2
    pred = label_scores[:, 0, :]
    target = torch.tensor(batch.targets).cuda().detach().long()
    # 计算交叉熵损失
    ce_loss = F.cross_entropy(pred, target, reduction='none')

    # 计算难度权重
    pt = torch.exp(-ce_loss)
    at = alpha * target + (1 - alpha) * (1 - target)
    weight = at * pt ** gamma

    # 计算 Focal Loss
    focal_loss = weight * ce_loss
    return focal_loss.mean()
