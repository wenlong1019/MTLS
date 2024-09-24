import torch
import torch.nn.functional as F


def dist_similarity(textual_output, visual_output):
    visual_output = F.log_softmax(visual_output, dim=-1)
    textual_output = F.softmax(textual_output, dim=-1)
    # Calculate the KL divergence between the visual_output and textual_output
    similarity = F.kl_div(visual_output, textual_output, reduction='sum')
    return similarity


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
        # Set the mask matrix to True from the 1st to the actual length of the target sequence
        mask_matrix[b, 1:batch.seq_lengths[b]] = True
        # If the length of the target sequence is not equal to the actual length of the target sequence - 1,
        # set the target sequence to the first n-1 elements
        if len(targets[b]) != batch.seq_lengths[b] - 1:
            targets[b] = targets[b][:batch.seq_lengths[b] - 1]
    scores = label_scores[mask_matrix]
    gold_targets = torch.cat(targets).cuda().detach().long()
    # Calculate the cross-entropy loss using the scores and gold targets
    loss = F.cross_entropy(scores, gold_targets)
    return loss
