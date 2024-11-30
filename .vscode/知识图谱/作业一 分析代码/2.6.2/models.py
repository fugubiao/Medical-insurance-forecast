# encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(self, ent_num, rel_num, device, dim=100, norm=1, margin=2.0, alpha=0.01):
        super(TransE, self).__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = device
        self.dim = dim
        self.norm = norm # 使用L1范数还是L2范数
        self.margin = margin
        self.alpha = alpha

        # 初始化实体和关系表示向量
        self.ent_embeddings = nn.Embedding(self.ent_num, self.dim)
        torch.nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, 1)

        self.rel_embeddings = nn.Embedding(self.rel_num, self.dim)
        torch.nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, 2, 1)

        # 损失函数
        self.criterion = nn.MarginRankingLoss(margin=self.margin)

    def get_ent_resps(self, ent_idx): #[batch]
        return self.ent_embeddings(ent_idx) # [batch, emb]

    # 越小越好
    def scoring(self, h_idx, r_idx, t_idx):
        h_embs = self.ent_embeddings(h_idx) # [batch, emb]
        r_embs = self.rel_embeddings(r_idx) # [batch, emb]
        t_embs = self.ent_embeddings(t_idx) # [batch, emb]
        scores = h_embs + r_embs - t_embs

        norms = (torch.mean(h_embs.norm(p=self.norm, dim=1) - 1.0)
                 + torch.mean(r_embs ** 2) +
                 torch.mean(t_embs.norm(p=self.norm, dim=1) - 1.0)) / 3

        return scores.norm(p=self.norm, dim=1), norms

    # 计算损失
    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.float, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def forward(self, ph_idx, pr_idx, pt_idx, nh_idx, nr_idx, nt_idx):
        pos_scores, pos_norms = self.scoring(ph_idx, pr_idx, pt_idx)
        neg_scores, neg_norms = self.scoring(nh_idx, nr_idx, nt_idx)

        tmp_loss = self.loss(pos_scores, neg_scores)
        tmp_loss += self.alpha * pos_norms
        tmp_loss += self.alpha * neg_norms

        return tmp_loss, pos_scores, neg_scores


class RESCAL(nn.Module):
    def __init__(self, ent_num, rel_num, device, dim=100, norm=1, alpha=0.001):
        super(RESCAL, self).__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = device
        self.dim = dim
        self.norm = norm  # 使用L1范数还是L2范数
        self.alpha = alpha

        # 初始化实体向量
        self.ent_embeddings = nn.Embedding(self.ent_num, self.dim)
        torch.nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, 1)

        # 初始化关系矩阵
        self.rel_embeddings = nn.Embedding(self.rel_num, self.dim * self.dim)
        torch.nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, 2, 1)

        # 损失函数
        self.criterion = nn.MSELoss()

    def get_ent_resps(self, ent_idx): #[batch]
        return self.ent_embeddings(ent_idx) # [batch, emb]

    # 越大越好，正例接近1，负例接近0
    def scoring(self, h_idx, r_idx, t_idx):
        h_embs = self.ent_embeddings(h_idx)  # [batch, emb]
        t_embs = self.ent_embeddings(t_idx)  # [batch, emb]
        r_mats = self.rel_embeddings(r_idx)  # [batch, emb * emb]

        norms = (torch.mean(h_embs ** 2) + torch.mean(t_embs ** 2) + torch.mean(r_mats ** 2)) / 3

        r_mats = r_mats.view(-1, self.dim, self.dim)
        t_embs = t_embs.view(-1, self.dim, 1)

        tr_embs = torch.matmul(r_mats, t_embs)
        tr_embs = tr_embs.view(-1, self.dim)

        return torch.sum(h_embs * tr_embs, -1), norms

    def forward(self, h_idx, r_idx, t_idx, labels):
        scores, norms = self.scoring(h_idx, r_idx, t_idx)

        tmp_loss = self.criterion(scores, labels.float())
        tmp_loss += self.alpha * norms

        return tmp_loss, scores

