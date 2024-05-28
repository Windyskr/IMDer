import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from .scoremodel import ScoreNet, loss_fn, Euler_Maruyama_sampler
import functools
from .rcan import Group
from random import sample

__all__ = ['IMDER']


# 定义均方误差损失函数
# 在模型训练过程中，MSE 用于计算重建的模态和原始模态之间的误差，
# 从而指导模型学习更准确地重建缺失模态。
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


# Set up the SDE (SDE is used to define Diffusion Process)
device = 'cuda'


# 边际概率标准差函数和扩散系数函数
# 计算扩散过程中给定时间步 t 的边际概率标准差。
# 这用于确定从初始状态到当前时间步的标准差。
def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    t = torch.as_tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


# 计算扩散过程中的扩散系数，用于生成新的数据点。
def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.as_tensor(sigma ** t, device=device)


# 定义 IMDER 模型
class IMDER(nn.Module):
    def __init__(self, args):
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
        # 初始化MSE损失函数和扩散模型的参数，并设置 ScoreNet 用于每种模态的评分。
        # ScoreNet 负责评估和生成模态样本。
        self.MSE = MSE()

        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        output_dim = args.num_classes if args.train_mode == "classification" else 1
        # 扩散过程中的标准差参数，用于控制噪声的扩散程度。
        sigma = 25.0
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)  # used for sample
        # 用于估计数据分布的评分网络，通过学习数据的边际概率标准差来生成样本。
        self.score_l = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)
        self.score_v = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)
        self.score_a = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)
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
        modal_idx = [0, 1, 2]  # (0:text, 1:vision, 2:audio)
        ava_modal_idx = sample(modal_idx, num_modal)  # sample available modality
        if num_modal == 1:  # one modality is available
            if ava_modal_idx[0] == 0:  # has text
                # proj_x_l 是文本模态的特征投影，作为条件输入给扩散模型。
                conditions = proj_x_l
                # 通过 loss_fn 函数计算评分损失，用于指导 ScoreNet 学习如何生成缺失的模态数据。
                # --> loss_fn
                # 文本模态是已知的，因此 loss_score_l 为零。
                # 缺失 a 和 v ，对模型减少还原出 a 、v 和 l 的能力。
                loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
                loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
                loss_score_l = torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                # 使用扩散模型生成缺失的音频和视觉模态数据。
                # Euler_Maruyama_sampler 是用于从扩散模型中采样的函数，
                # 通过给定的 ScoreNet 和条件生成新的模态数据。
                proj_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn,
                                                  text.size(0),
                                                  device='cuda', condition=conditions)
                proj_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn,
                                                  text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                # 通过预定义的重建网络（self.rec_a 和 self.rec_v），
                # 对生成的音频和视觉模态数据进行进一步处理和优化。
                proj_x_a = self.rec_a(proj_x_a)
                proj_x_v = self.rec_v(proj_x_v)
                # 使用均方误差（MSE）损失函数计算生成的音频和视觉模态数据与真实数据之间的差异，
                # 用于指导模型优化生成过程。
                loss_rec = self.MSE(proj_x_a, gt_a) + self.MSE(proj_x_v, gt_v)
            elif ava_modal_idx[0] == 1:  # has video
                conditions = proj_x_v
                loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
                loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
                loss_score_v = torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn,
                                                  text.size(0),
                                                  device='cuda', condition=conditions)
                proj_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn,
                                                  text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_l = self.rec_l(proj_x_l)
                proj_x_a = self.rec_a(proj_x_a)
                loss_rec = self.MSE(proj_x_l, gt_l) + self.MSE(proj_x_a, gt_a)
            else:  # has audio
                conditions = proj_x_a
                loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
                loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
                loss_score_a = torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn,
                                                  text.size(0),
                                                  device='cuda', condition=conditions)
                proj_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn,
                                                  text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_l = self.rec_l(proj_x_l)
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.MSE(proj_x_l, gt_l) + self.MSE(proj_x_v, gt_v)
        if num_modal == 2:  # two modalities are available
            if set(modal_idx) - set(ava_modal_idx) == {0}:  # L is missing (V,A available)
                conditions = self.cat_va(
                    torch.cat([proj_x_v, proj_x_a], dim=1))  # cat two avail modalities as conditions
                loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
                loss_score_v, loss_score_a = torch.tensor(0), torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn,
                                                  text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_l = self.rec_l(proj_x_l)
                loss_rec = self.MSE(proj_x_l, gt_l)
            if set(modal_idx) - set(ava_modal_idx) == {1}:  # V is missing (L,A available)
                conditions = self.cat_la(
                    torch.cat([proj_x_l, proj_x_a], dim=1))  # cat two avail modalities as conditions
                loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
                loss_score_l, loss_score_a = torch.tensor(0), torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn,
                                                  text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.MSE(proj_x_v, gt_v)
            if set(modal_idx) - set(ava_modal_idx) == {2}:  # A is missing (L,V available)
                conditions = self.cat_lv(
                    torch.cat([proj_x_l, proj_x_v], dim=1))  # cat two avail modalities as conditions
                loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
                loss_score_l, loss_score_v = torch.tensor(0), torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn,
                                                  text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_a = self.rec_a(proj_x_a)
                loss_rec = self.MSE(proj_x_a, gt_a)
        if num_modal == 3:  # no missing
            loss_score_l, loss_score_v, loss_score_a = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            loss_rec = torch.tensor(0)

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        # self.trans_l_with_a：使用音频特征对文本特征进行跨模态注意力操作。
        # self.trans_l_with_v：使用视觉特征对文本特征进行跨模态注意力操作。
        # 将两个跨模态注意力的结果拼接在一起，并通过自注意力机制 self.trans_l_mem 进一步处理。
        # 提取处理后的最后一个时间步的特征 last_h_l，用于最终的预测。
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

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
        # 将三个模态的最后时间步特征拼接在一起。
        # 通过两层全连接层（self.proj1 和 self.proj2）和 ReLU 激活函数对拼接后的特征进行处理。
        # 使用 Dropout 进行正则化，防止过拟合。
        # 残差连接（last_hs_proj += last_hs）帮助保持特征的完整性和梯度的流动。
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