import random
import os
import numpy as np
import torch
from functools import lru_cache
from torch.optim.lr_scheduler import LambdaLR


def seed_everything(seed=1029):
    """
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


@lru_cache
def sinusoidal_position(pos, output_dim):
    position_ids = torch.tensor([pos], dtype=torch.float).unsqueeze(-1)
    indices = torch.arange(0, output_dim // 2, dtype=torch.float)
    indices = torch.pow(10000, -2 * indices / output_dim)
    embeddings = position_ids * indices
    embeddings = torch.stack(
        [torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = embeddings.repeat((1, *([1] * len(embeddings.shape))))
    embeddings = torch.reshape(embeddings, (1, output_dim))
    return embeddings


def rope(pos, v):
    # 计算a和b的RoPE编码
    pos_emb = sinusoidal_position(pos, int(v.shape[-1]))

    # 应用RoPE编码
    cos_pos = pos_emb[..., None, 1::2].repeat_interleave(
        2, dim=-1).view(-1)
    sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1).view(-1)

    rope_v = v * cos_pos + \
        torch.stack([-v[..., 1::2], v[..., ::2]], -
                    1).reshape(v.shape) * sin_pos
    return rope_v


def get_linear_schedule_with_warmup_two_stage(optimizer, num_warmup_steps_stage_one, num_training_steps_stage_one, num_warmup_steps_stage_two, num_training_steps_stage_two, stage_one_lr_scale, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_training_steps_stage_one:
            if current_step < num_warmup_steps_stage_one:
                return float(current_step) / float(max(1, num_warmup_steps_stage_one))
            return max(
                0.0, float(num_training_steps_stage_one - current_step) / float(max(
                    1, num_training_steps_stage_one - num_warmup_steps_stage_one)) * stage_one_lr_scale
            )
        else:
            current_step = current_step - num_training_steps_stage_one
            if current_step < num_warmup_steps_stage_two:
                return float(current_step) / float(max(1, num_warmup_steps_stage_two))
            return max(
                0.0, float(num_training_steps_stage_two - current_step) /
                float(max(1, num_training_steps_stage_two -
                      num_warmup_steps_stage_two))
            )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
