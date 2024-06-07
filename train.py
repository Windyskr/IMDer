from run import IMDER_run
import torch

import torch

dummy_state_dict = {'dummy_key': 'dummy_value'}
torch.save(dummy_state_dict, 'ckpt_path_l.pth')
torch.save(dummy_state_dict, 'ckpt_path_v.pth')
torch.save(dummy_state_dict, 'ckpt_path_a.pth')

from run import IMDER_run

default_unet_config = {
    'target': 'trains.singleTask.model.ldm.modules.diffusionmodules.openaimodel.UNetModel',  # 修改为实际的 U-Net 模型类路径
    'image_size': 256,
    'in_channels': 3,
    'model_channels': 64,
    'out_channels': 3,
    'num_res_blocks': 2,
    'attention_resolutions': [16, 8],
    'dropout': 0.1,
    'channel_mult': [1, 2, 4, 8],
    'conv_resample': True,
    'dims': 2,
    'num_classes': None,
    'use_checkpoint': False,
    'use_fp16': False,
    'num_heads': 8,
    'num_head_channels': -1,
    'num_heads_upsample': -1,
    'use_scale_shift_norm': True,
    'resblock_updown': True,
    'use_new_attention_order': False,
    'use_spatial_transformer': False,
    'transformer_depth': 1,
    'context_dim': None,
    'n_embed': None,
    'legacy': True
}

# 虚拟检查点路径
ckpt_path_l = 'ckpt_path_l.pth'
ckpt_path_v = 'ckpt_path_v.pth'
ckpt_path_a = 'ckpt_path_a.pth'

IMDER_run(model_name='imder',
          dataset_name='mosi',
          seeds=[1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119],
          mr=0.1,
          unet_config_l=default_unet_config,
          unet_config_v=default_unet_config,
          unet_config_a=default_unet_config,
          ckpt_path_l=ckpt_path_l,
          ckpt_path_v=ckpt_path_v,
          ckpt_path_a=ckpt_path_a)