import torch
import torch.nn as nn
import torch.nn.functional as F


def KL_loss(x_logits, y_logits):
    x1 = F.log_softmax(x_logits, dim=-1)
    y1 = F.softmax(y_logits, dim=-1)
    x2 = F.softmax(x_logits, dim=-1)
    y2 = F.log_softmax(y_logits, dim=-1)
    kl = nn.KLDivLoss(reduction='batchmean')
    return 0.5 * (kl(x1, y1) + kl(y2, x2))


def contrastive_loss(inputs, labels, tau=1):
    '''
    inputs: [batch_size, hidden_dim]
    labels: [batch_size]
    '''
    inputs = F.normalize(inputs, dim=-1)  # 模为1，故内积就是相似度
    # print(torch.matmul(inputs, inputs.T) / tau)
    inputs = torch.exp(torch.matmul(inputs, inputs.T) / tau)
    # mask构建，label相同的True
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
    diagonal_mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    mask = torch.logical_and(mask, ~diagonal_mask)
    # print(mask)
    positive_similarity = torch.sum(
        inputs * mask, dim=-1) + 1e-8  # 千万注意，这里加上1e-6，否则loss为nan
    loss = -torch.log(positive_similarity / torch.sum(inputs, dim=-1))
    # 没有正样本的一行不计算loss
    positive_mask = torch.sum(mask, dim=-1) > 0
    loss = torch.where(positive_mask, loss, torch.zeros_like(loss))
    # loss = torch.mean(loss[positive_mask])  # 一开始可能会出现nan，迭代几次就好了,差
    loss = torch.mean(loss)
    return loss


def sup_contrastive_loss(inputs, labels, tau=0.1):
    """
    :param inputs: torch.Tensor, shape [batch_size, projection_dim]
    :param labels: torch.Tensor, shape [batch_size]
    :return: torch.Tensor, scalar
    """
    device = torch.device("cuda") if inputs.is_cuda else torch.device("cpu")

    inputs = F.normalize(inputs, dim=-1)

    dot_product_tempered = torch.mm(inputs, inputs.T) / tau
    # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
    exp_dot_tempered = (
        torch.exp(dot_product_tempered -
                  torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
    )

    mask_similar_class = (labels.unsqueeze(1).repeat(
        1, labels.shape[0]) == labels).to(device)
    mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
    mask_combined = mask_similar_class * mask_anchor_out
    cardinality_per_samples = torch.sum(mask_combined, dim=1) + 1e-8
    # print(cardinality_per_samples)

    log_prob = -torch.log(exp_dot_tempered /
                          (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
    supervised_contrastive_loss_per_sample = torch.sum(
        log_prob * mask_combined, dim=1) / cardinality_per_samples
    supervised_contrastive_loss = torch.mean(
        supervised_contrastive_loss_per_sample)
    return supervised_contrastive_loss

# def kmean_contrastive_loss(inputs, labels, km: KMean, tau=1):
#     '''
#     inputs: [batch_size, hidden_dim]
#     labels: [batch_size]
#     '''
#     inputs = F.normalize(inputs, dim=-1)  # 模为1，故内积就是相似度
#     initializer = [0 for _ in range(km.sub_labels)]
#     for k, v in km.label2sub.items():
#         assert len(km.initializer[k]) == len(v)
#         for l, i in zip(v, km.initializer[k]):
#             initializer[l] = i
#     types = F.normalize(torch.tensor(initializer).cuda(), dim=-1)
#     # print(torch.matmul(inputs, types.T) / tau)
#     inputs = torch.exp(torch.matmul(inputs, types.T) / tau)
#     mask = F.one_hot(labels, num_classes=km.sub_labels)
#     # print(mask)
#     positive_similarity = torch.sum(inputs * mask, dim=-1) + 1e-8  # 千万注意，这里加上1e-6，否则loss为nan
#     loss = -torch.log(positive_similarity / torch.sum(inputs, dim=-1))
#     # 没有正样本的一行不计算loss
#     positive_mask = torch.sum(mask, dim=-1) > 0
#     loss = torch.where(positive_mask, loss, torch.zeros_like(loss))
#     # loss = torch.mean(loss[positive_mask])  # 一开始可能会出现nan，迭代几次就好了,差
#     loss = torch.mean(loss)
#     return loss


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,
                          ignore_index=self.ignore_index)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                           ignore_index=self.ignore_index)
