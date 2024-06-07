import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
import trains.singleTask.model.ddpm as ddpm
import functools
from .rcan import Group
from random import sample

__all__ = ['IMDER']

# 定义均方误差损失函数
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

# 定义 IMDER 模型
class IMDER(nn.Module):
    def __init__(self, args, unet_config_l, unet_config_v, unet_config_a, ckpt_path_l, ckpt_path_v, ckpt_path_a):
        super(IMDER, self).__init__()
        # 1. BERT 文本编码器初始化
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        # 2. 特征维度和注意力头数
        # 模型需要处理不同模态的特征，每个模态的特征维度和注意力头数在配置文件中设置
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        # 3. 层数和dropout设置
        # 不同的dropout层防止过拟合，以增强模型的泛化能力
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        # 4. 损失函数和扩散模型
        # 扩散模型用于处理缺失模态，通过生成缺失模态的样本来增强数据。
        self.MSE = MSE()

        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        # 初始化DDPM模型
        self.ddpm_l = ddpm.DDPM(unet_config=unet_config_l, timesteps=1000)
        self.ddpm_v = ddpm.DDPM(unet_config=unet_config_v, timesteps=1000)
        self.ddpm_a = ddpm.DDPM(unet_config=unet_config_a, timesteps=1000)

        # 加载预训练模型权重
        self.ddpm_l.init_from_ckpt(ckpt_path_l)
        self.ddpm_v.init_from_ckpt(ckpt_path_v)
        self.ddpm_a.init_from_ckpt(ckpt_path_a)

        # 5. 特征拼接和重建网络
        # 模态融合和重建
        # 定义用于拼接和重建模态特征的卷积层和残差块。这些层在恢复缺失模态的过程中起重要作用。
        self.cat_lv = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        self.cat_la = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        self.cat_va = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        self.rec_l = nn.Sequential(
            nn.Conv1d(self.d_l, self.d_l * 2, 1),
            Group(num_channels=self.d_l * 2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_l * 2, self.d_l, 1)
        )

        self.rec_v = nn.Sequential(
            nn.Conv1d(self.d_v, self.d_v * 2, 1),
            Group(num_channels=self.d_v * 2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_v * 2, self.d_v, 1)
        )

        self.rec_a = nn.Sequential(
            nn.Conv1d(self.d_a, self.d_a * 2, 1),
            Group(num_channels=self.d_a * 2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_a * 2, self.d_a, 1)
        )

        # 6. 时间卷积层
        # 时间卷积用于处理序列特征。
        # 将原始特征投影到目标特征空间，增强特征表达。
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 7. 跨模态注意力
        # 跨模态注意力机制用于捕捉模态间的依赖关系。
        # 定义不同模态之间的跨模态注意力层，促进模态间的信息融合。
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 8. 自注意力
        # 自注意力机制用于捕捉模态内部的依赖关系。
        # 定义每种模态的自注意力层，增强模态内部特征的表达。
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # 9. 投影层和输出层
        # 最终的特征投影和输出层。
        # 对融合后的特征进行投影，并生成最终的输出结果（分类或回归）。
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        # 根据不同的类型返回一个 TransformerEncoder 实例。
        # embed_dim：嵌入维度，根据不同的类型（如 l, a, v 等）选择相应的维度。
        # attn_dropout：注意力的 dropout 比例，也根据不同的类型选择相应的值。
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, num_modal=None):
        with torch.no_grad():
            if self.use_bert:
                text = self.text_model(text)
        # 将原始特征投影到目标特征空间，以统一特征维度。
        # 通过卷积层将文本、音频和视频特征投影到目标维度，并进行转置和 dropout 操作。
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        # Project the textual/visual/audio features
        with torch.no_grad():
            proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
            proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
            proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
            gt_l, gt_v, gt_a = proj_x_l, proj_x_v, proj_x_a

        # 随机选择可用的模态，以模拟不同模态缺失的情况。
        # 随机选择 num_modal 个可用模态，生成 ava_modal_idx 列表。
        # 论文对应：模拟真实世界中模态缺失的情况，通过随机选择模态来训练模型，提高其应对模态缺失的能力。
        modal_idx = [0, 1, 2]
        ava_modal_idx = sample(modal_idx, num_modal)

        if num_modal == 1:
            if ava_modal_idx[0] == 0:
                conditions = proj_x_l
                loss_score_a = self.ddpm_a.training_step(conditions, proj_x_a)
                loss_score_v = self.ddpm_v.training_step(conditions, proj_x_v)
                loss_score_l = torch.tensor(0)
                proj_x_a = self.ddpm_a.sample(conditions, proj_x_a.shape)
                proj_x_v = self.ddpm_v.sample(conditions, proj_x_v.shape)
                proj_x_a = self.rec_a(proj_x_a)
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.MSE(proj_x_a, gt_a) + self.MSE(proj_x_v, gt_v)
            elif ava_modal_idx[0] == 1:
                conditions = proj_x_v
                loss_score_l = self.ddpm_l.training_step(conditions, proj_x_l)
                loss_score_a = self.ddpm_a.training_step(conditions, proj_x_a)
                loss_score_v = torch.tensor(0)
                proj_x_l = self.ddpm_l.sample(conditions, proj_x_l.shape)
                proj_x_a = self.ddpm_a.sample(conditions, proj_x_a.shape)
                proj_x_l = self.rec_l(proj_x_l)
                proj_x_a = self.rec_a(proj_x_a)
                loss_rec = self.MSE(proj_x_l, gt_l) + self.MSE(proj_x_a, gt_a)
            else:
                conditions = proj_x_a
                loss_score_l = self.ddpm_l.training_step(conditions, proj_x_l)
                loss_score_v = self.ddpm_v.training_step(conditions, proj_x_v)
                loss_score_a = torch.tensor(0)
                proj_x_l = self.ddpm_l.sample(conditions, proj_x_l.shape)
                proj_x_v = self.ddpm_v.sample(conditions, proj_x_v.shape)
                proj_x_l = self.rec_l(proj_x_l)
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.MSE(proj_x_l, gt_l) + self.MSE(proj_x_v, gt_v)
        elif num_modal == 2:
            if set(modal_idx) - set(ava_modal_idx) == {0}:
                conditions = self.cat_va(torch.cat([proj_x_v, proj_x_a], dim=1))
                loss_score_l = self.ddpm_l.training_step(conditions, proj_x_l)
                loss_score_v, loss_score_a = torch.tensor(0), torch.tensor(0)
                proj_x_l = self.ddpm_l.sample(conditions, proj_x_l.shape)
                proj_x_l = self.rec_l(proj_x_l)
                loss_rec = self.MSE(proj_x_l, gt_l)
            elif set(modal_idx) - set(ava_modal_idx) == {1}:
                conditions = self.cat_la(torch.cat([proj_x_l, proj_x_a], dim=1))
                loss_score_v = self.ddpm_v.training_step(conditions, proj_x_v)
                loss_score_l, loss_score_a = torch.tensor(0), torch.tensor(0)
                proj_x_v = self.ddpm_v.sample(conditions, proj_x_v.shape)
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.MSE(proj_x_v, gt_v)
            else:
                conditions = self.cat_lv(torch.cat([proj_x_l, proj_x_v], dim=1))
                loss_score_a = self.ddpm_a.training_step(conditions, proj_x_a)
                loss_score_l, loss_score_v = torch.tensor(0), torch.tensor(0)
                proj_x_a = self.ddpm_a.sample(conditions, proj_x_a.shape)
                proj_x_a = self.rec_a(proj_x_a)
                loss_rec = self.MSE(proj_x_a, gt_a)
        else:
            loss_score_l, loss_score_v, loss_score_a = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            loss_rec = torch.tensor(0)

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]
        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]
        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        # A residual block
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs
        # 生成最终的情感识别结果。
        output = self.out_layer(last_hs_proj)

        res = {
            'Feature_t': last_h_l,
            'Feature_a': last_h_a,
            'Feature_v': last_h_v,
            'Feature_f': last_hs,
            'loss_score_l': loss_score_l,
            'loss_score_v': loss_score_v,
            'loss_score_a': loss_score_a,
            'loss_rec': loss_rec,
            'ava_modal_idx': ava_modal_idx,
            'M': output
        }
        return res
