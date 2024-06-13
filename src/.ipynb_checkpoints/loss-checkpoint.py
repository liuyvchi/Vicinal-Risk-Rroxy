import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn, Tensor
from typing import Tuple
import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.stats
from torch.distributions.normal import Normal


def gaussian_kernal(dists, confidence, labelM, mask_self, sigma=0.3):
    #高斯核函数 np.exp(-dists ** 2 / (2 * sigma ** 2))
    normal_d = Normal(0, sigma)
    # sigma = (1-confidence).unsqueeze(1).clamp(min=0.1, max=0.9)
    # sigma = (1-confidence.mean())*0.3
    
    dists = (dists - mask_self).clamp(min=0)

   
    device = dists.device

    #  _, index_sort = dists.sort(dim=-1)
    # weight_return = torch.zeros(dists.shape).to(device).clamp(min=1e-6)
    # for i in range(len(dists)):
    #     num_neighbors = labelM[i].sum()
    #     kernel_dists = torch.linspace(0,1,num_neighbors).to(device)
    #     density = normal_d.log_prob(kernel_dists).exp()
    #     weight_return[i][index_sort[i]][:num_neighbors] = density

    scale = normal_d.log_prob(dists).exp().clamp(min=1e-6)

    density = (1- dists)

    # weight_return = density*scale
    weight_return = density

    return weight_return 

def compute_vicinalRisk_L2(s1, s2):
    risk = (s1**2)*s2 + s1*(s2**2) - 4*s1*s2 + s1 + s2
    score = (1 - risk).sum()/len(risk)
    return score

def mixup_data(x, y, iter_num=1, alpha=0.2, use_cuda=True):
    seed = iter_num
    torch.manual_seed(seed)
    batch_size = x.size()[0]
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = torch.tensor([np.random.beta(4, 2) for i in range(batch_size)]).cuda()
    else:
        lam = torch.tensor([1 for i in range(batch_size)]).cuda()

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    try:
        mixed_x =  x * lam[:, None, None, None] + x[index, :] * (1 - lam[:, None, None, None])
    except:
        print(lam)
        print(index.size(), lam.size(), x.size())
        assert(0)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam , index


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return loss.mean()


def find_score(far, vr, target=1e-4):
    # far is an array in descending order, find the index of far whose element is closest to target,
    # and return vr[index]
    l = 0
    u = far.size - 1
    e = -1
    while u - l > 1:
        mid = (l + u) // 2
        # print far[mid]
        if far[mid] == target:
            if target != 0:
                return vr[mid]
            else:
                e = mid
                break
        elif far[mid] < target:
            u = mid
        else:
            l = mid
    if target == 0:
        i = e
        while i >= 0:
            if far[i] != 0:
                break
            i -= 1
        if i >= 0:
            return vr[i + 1]
        else:
            return vr[u]
    # Actually, either array[l] or both[u] is not equal to target, so choose a closer one.

    if target != 0 and far[l] / target >= 8:  # cannot find points that's close enough to target.
        return 0.0
    nearest_point = (target - far[l]) / (far[u] - far[l]) * (vr[u] - vr[l]) + vr[l]
    return nearest_point
    # if far[l] - target > target - far[u]:
    #     return vr[u]
    # else:
    #     return vr[l]


def compute_roc(score, label, num_thresholds=1000, show_sample_hist=False):
    pos_dist = score[label == 1]
    neg_dist = score[label == 0]

    num_pos_samples = pos_dist.size
    num_neg_samples = neg_dist.size
    data_max = np.max(score)
    data_min = np.min(score)
    unit = (data_max - data_min) * 1.0 / num_thresholds
    threshold = data_min + (data_max - data_min) * np.array(range(1, num_thresholds + 1)) / num_thresholds
    new_interval = threshold - unit / 2.0 + 2e-6
    new_interval = np.append(new_interval, np.array(new_interval[-1] + unit))
    P = np.triu(np.ones(num_thresholds))

    pos_hist, dummy = np.histogram(pos_dist, new_interval)
    neg_hist, dummy2 = np.histogram(neg_dist, new_interval)
    pos_mat = pos_hist[:, np.newaxis]
    neg_mat = neg_hist[:, np.newaxis]

    assert pos_hist.size == neg_hist.size == num_thresholds
    far = np.dot(P, neg_mat) / num_neg_samples
    far = np.squeeze(far)
    vr = np.dot(P, pos_mat) / num_pos_samples
    vr = np.squeeze(vr)
    if show_sample_hist is False:
        return far, vr, threshold
    else:
        return far, vr, threshold, pos_hist, neg_hist
    
def test_tprATfpr(score, label):
    num_trials = 5
    vr = np.zeros([num_trials, 3], dtype=np.float32)
    far_array, vr_array, threshold = compute_roc(score.flat, label.flat, 5000)
    for i in range(num_trials):
        vr[i, 0] = find_score(far_array, vr_array, 0.2)
        vr[i, 1] = find_score(far_array, vr_array, 1e-1)
        vr[i, 2] = find_score(far_array, vr_array, 1e-2)
    mean_vr = vr.mean(axis=0)
    return mean_vr


def roc_auc(score, label):
    return roc_auc_score(label, score)

def convert_label_to_AUsim(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

def convert_label_to_overlap(logit: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:

    confidenceMul_matrix = logit @ logit.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    confidenceMul_matrix = confidenceMul_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)

    positive_simi = confidenceMul_matrix[positive_matrix]
    negative_simi = confidenceMul_matrix[negative_matrix]

    return positive_simi, negative_simi

def convert_label_to_overlapMatrix(logit: Tensor, logit2: Tensor, label: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

    confidenceMul_matrix = logit @ logit2.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
    label_matrix = label_matrix.long()

    return confidenceMul_matrix, label_matrix


def convert_label_to_cdistMatrix(logit: Tensor, logit2: Tensor, label: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

    cdist_matrix = torch.cdist(logit, logit2, p=1)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
    label_matrix = label_matrix.long()

    return cdist_matrix, label_matrix

def convert_label_to_KLdistMatrix(logit: Tensor, logit2: Tensor, label: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    logit = logit.unsqueeze(0)
    logit2 = logit2.unsqueeze(1)

    KLdist_matrix = F.kl_div(torch.log(logit2), logit, reduction='none')
    print(KLdist_matrix.shape)
    assert(0)

    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
    label_matrix = label_matrix.long()

    return KLdist_matrix, label_matrix

def convert_label_to_confMatrix(softmax1: Tensor, softmax2: Tensor, label: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    
    # _, idx = softmax1.max(dim=1)
    # g = idx.view(1, -1).expand(len(softmax1), -1)
    # # 使用gather函数获取softmax2中对应位置的值
    # c = torch.gather(softmax2, 1, g)
    # vicinalProb_matrix1 = c.T

    # find the index of the maximum value along the second dimension of A
    max_indices = torch.argmax(softmax1, dim=1)

    # extract the corresponding columns of B for each row of A using gathered index
    C = softmax2.index_select(1, max_indices)

    vicinalProb_matrix1 = C.T

    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
    label_matrix = label_matrix.long()

    return vicinalProb_matrix1, label_matrix

def convert_label_to_EI(logit: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:

    EI_matrix = logit.max(dim=-1)[0].unsqueeze(1) @ logit.transpose(1, 0).max(dim=0)[0].unsqueeze(0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    EI_matrix = EI_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)

    return EI_matrix[positive_matrix], EI_matrix[negative_matrix]

def convert_label_to_cosimi(logit: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:

    cosimi_matrix = torch.nn.CosineSimilarity()(logit, logit)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    cosimi_matrix = cosimi_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)

    return cosimi_matrix[positive_matrix], cosimi_matrix[negative_matrix]


def convert_label_to_intensDis(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    intensDis_matrix = normed_feature.max(dim=-1)[0].unsqueeze(dim=1) - normed_feature.max(dim=-1)[0].unsqueeze(dim=0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    intensDis_matrix = intensDis_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return intensDis_matrix[positive_matrix], intensDis_matrix[negative_matrix]

class CircleLoss(nn.Module):
    def __init__(self) -> None:
        super(CircleLoss, self).__init__()
        self.m = 0.25
        self.gamma = 256
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class AU_Loss(nn.Module):
    def __init__(self) -> None:
        super(AU_Loss, self).__init__()
        self.m = 0.25
        self.gamma = 16
        self.soft_plus = nn.Softplus()


    def forward(self, sp: Tensor, sn: Tensor, sp_au, sn_au) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - (1 - sp_au) * (sp - delta_p) * self.gamma
        logit_n = sn_au * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss

def ACLoss(att_map1, att_map2, grid_l, output):
    flip_grid_large = grid_l.expand(output.size(0), -1, -1, -1)
    flip_grid_large = Variable(flip_grid_large, requires_grad = False)
    flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
    att_map2_flip = F.grid_sample(att_map2, flip_grid_large, mode = 'bilinear', padding_mode = 'border', align_corners=True)
    flip_loss_l = F.mse_loss(att_map1, att_map2_flip)
    return flip_loss_l