import sys
import os
import gc
import time
import math
import random
import json
import logging
import argparse
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import yaml
import datasets
import deepspeed
from tqdm import tqdm
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
# NOTE: transformers 側の DeepSpeed 統合
from transformers.integrations.deepspeed import HfDeepSpeedConfig
 # ここが zero.Init を内部で仕込んでくれる:contentReference[oaicite:1]{index=1}

from accelerate.utils import set_seed

from train_functions import configure_optimizer, train_step
from profiler import timer, time_function


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================
#  RWKV / 環境セットアップ
# ============================================================

def setup_env():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    rwkv_insidea_path = os.path.join(parent_dir, 'rwkv_inside')
    sys.path.append(rwkv_insidea_path)
    sys.path.append(parent_dir)
    print(f'add path: {rwkv_insidea_path} to sys.path')

    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = os.environ.get('RWKV_T_MAX', '4096')
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_CTXLEN'] = os.environ.get('RWKV_CTXLEN', '4096')

    if 'WKV' not in os.environ:
        os.environ['WKV'] = ''
    if "RWKV_TRAIN_TYPE" not in os.environ:
        os.environ["RWKV_TRAIN_TYPE"] = ''

    RWKV_VERSION = os.environ.get('RWKV_VERSION', 'v7')
    if RWKV_VERSION == 'v7':
        os.environ["RWKV_MY_TESTING"] = 'x070'
    else:
        os.environ["RWKV_MY_TESTING"] = 'x060'
    print(f'RWKV_VERSION is {RWKV_VERSION}')


setup_env()

# ============================================================
#  メモリ測定ユーティリティ
# ============================================================

def measure_model_memory(model: torch.nn.Module, detailed: bool = True) -> Dict[str, float]:
    gc.collect()
    torch.cuda.empty_cache()

    total_size = 0
    param_size = 0
    buffer_size = 0

    for name, param in model.named_parameters():
        if param.is_cuda:
            size = param.numel() * param.element_size()
            param_size += size
            if detailed:
                size_mb = size / 1024 ** 2
                print(f"Parameter {name}: {param.shape}, {param.dtype}, {size_mb:.2f} MB")

    for name, buffer in model.named_buffers():
        if buffer.is_cuda:
            size = buffer.numel() * buffer.element_size()
            buffer_size += size
            if detailed:
                size_mb = size / 1024 ** 2
                print(f"Buffer {name}: {buffer.shape}, {buffer.dtype}, {size_mb:.2f} MB")

    total_size = param_size + buffer_size

    return {
        'total_gb': total_size / 1024 ** 3,
        'param_gb': param_size / 1024 ** 3,
        'buffer_gb': buffer_size / 1024 ** 3,
        'total_mb': total_size / 1024 ** 2,
        'param_mb': param_size / 1024 ** 2,
        'buffer_mb': buffer_size / 1024 ** 2,
    }


# ============================================================
#  Arg Parser
# ============================================================

def create_arg_parser():
    node_rank = int(os.environ.get('NODE_RANK', 0))
    num_gpus = int(os.environ.get('NUM_GPUS', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    print(f'node_rank: {node_rank}, num_gpus: {num_gpus}, world_size: {world_size}')

    parser = argparse.ArgumentParser(description='Hybrid RWKV/Transformer Trainer with DeepSpeed ZeRO-3')

    parser.add_argument('--config_file', type=str, default='configs/test_hybrid.yaml')
    parser.add_argument('--preprocessed_data', type=str, nargs='+')
    parser.add_argument('--raw_data', type=str, nargs='+')
    parser.add_argument('--need_to_pad', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default='/data/rwkv/tmp')
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--has_group_norm', action='store_true', default=False)
    parser.add_argument('--gate_free', action='store_true', default=False)
    parser.add_argument('--min_len', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=4096)
    parser.add_argument('--freeze_mlp', action='store_true', default=False)
    parser.add_argument('--teacher_model_id', type=str, default=None)

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--grad_cp', type=int, default=0)
    parser.add_argument('--save_per_batches', type=int, default=10000)
    parser.add_argument('--my_exit', type=int, default=300)

    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr_init', type=float, default=6e-4)
    parser.add_argument('--lr_final', type=float, default=1e-5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--layerwise_lr', type=float, nargs='+', default=1)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=50)

    parser.add_argument('--epoch_begin', type=int, default=0)
    parser.add_argument('--epoch_count', type=int, default=150)
    parser.add_argument('--epoch_save', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=150)

    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--val_check_interval', type=int, default=5000)
    parser.add_argument('--num_sanity_val_steps', type=int, default=0)
    parser.add_argument('--log_every_n_steps', type=int, default=5000)

    parser.add_argument('--enable_checkpointing', type=bool, default=False)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)

    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--micro_bsz', type=int, default=2)
    parser.add_argument('--real_bsz', type=int)

    parser.add_argument('--my_pile_stage', type=int, default=0)
    parser.add_argument('--my_pile_edecay', type=float, default=0.0)
    parser.add_argument('--weight_decay_final', type=float, default=-1)

    parser.add_argument('--proj_dir', type=str)
    parser.add_argument('--eval_every_steps', type=int, default=100)
    parser.add_argument('--wandb', type=str, default='hybrid_trainer')
    parser.add_argument('--run_name', type=str, default='hybrid_trainer_a800')
    parser.add_argument('--strategy', type=str, default='deepspeed_stage_3')

    parser.add_argument("--ds_bucket_mb", default=200, type=int)
    parser.add_argument('--my_qa_mask', type=int, default=0)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--train_type', type=str, default='')
    parser.add_argument('--skip_steps', type=int, default=0)
    parser.add_argument('--full_params', action='store_true', default=False)

    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--ckpt_id', type=str, default=None)

    # DeepSpeed
    parser.add_argument('--deepspeed', action='store_true', help='Enable DeepSpeed')
    parser.add_argument('--deepspeed_config', type=str, default=None)
    parser.add_argument('--deepspeed_stage', type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument('--deepspeed_offload', action='store_true', default=False)
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=None)
    parser.add_argument('--local_rank', type=int, default=None)

    # Distillation stage (1: HS align, 2: KL, 3: SFT)
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--max_trained_tokens', type=int, default=100_000_000)
    parser.add_argument('--terminate_at_loss', type=float, default=0.0)

    parser.add_argument('--freeze_attention', type=int, default=0)
    parser.add_argument('--hybrid_attention_layers', type=int, default=0)
    parser.add_argument('--freeze_hybrid_attention', type=int, default=0)
    parser.add_argument('--allow_quant_frozen_layers', type=int, default=1)

    # 量子化系は削除するが、Config読み込みのために arg だけ残す
    parser.add_argument('--quant_mode', type=str, default="none")
    parser.add_argument('--peftmode', type=str, default="full")
    parser.add_argument('--peft_r', type=int, default=64)
    parser.add_argument('--peft_scaling', type=float, default=0.5)
    parser.add_argument('--peft_dropout', type=float, default=0.01)

    parser.add_argument('--bnb_optimizer_mode', type=int, default=0)  # もう使わないが互換のため

    return parser


# ============================================================
#  LR / WD スケジューラ
# ============================================================

def lr_schedule(args, step):
    w_step = args.warmup_steps
    if args.lr_final == args.lr_init or args.epoch_count == 0:
        return args.lr_init

    decay_step = step - args.my_pile_edecay * args.epoch_steps
    decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
    progress = (decay_step - w_step + 1) / (decay_total - w_step)
    progress = min(1, max(0, progress))

    if args.lr_final == 0 or args.lr_init == 0:
        lr = args.lr_init + (args.lr_final - args.lr_init) * progress
    else:
        lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * progress)

    if step < w_step:
        lr = lr * (0.01 + 0.99 * step / w_step)

    return lr


def weight_decay_schedule(args, progress):
    if args.weight_decay_final > 0:
        return args.weight_decay * math.exp(
            math.log(args.weight_decay_final / args.weight_decay) * progress
        )
    return args.weight_decay


def on_train_batch_start(args, model_engine, global_step, epoch):
    real_step = global_step + args.epoch_begin * args.epoch_steps

    lr = lr_schedule(args, real_step)

    progress = (real_step - args.warmup_steps + 1) / (
        (args.epoch_count - args.my_pile_edecay) * args.epoch_steps - args.warmup_steps
    )
    progress = min(1, max(0, progress))
    wd_now = weight_decay_schedule(args, progress)

    for param_group in model_engine.optimizer.param_groups:
        if param_group["weight_decay"] > 0:
            param_group["weight_decay"] = wd_now
        if args.layerwise_lr > 0:
            param_group["lr"] = lr * param_group.get("my_lr_scale", 1.0)
        else:
            param_group["lr"] = lr

    if global_step == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "train_log.txt"), "a") as f:
            f.write(f"NEW RUN {time.strftime('%Y-%m-%d %H:%M:%S')}\n{vars(args)}\n")

    return lr, wd_now


# ============================================================
#  ログ / 進捗管理
# ============================================================

pbar = None
total_loss = 0.0
total_updates = 0
trained_tokens = 0
avg_loss = 0.0


def on_train_batch_end(
    args,
    batch_idx,
    model_engine,
    teacher_engine,
    loss,
    teacher_loss,
    kl_loss,
    student_cross_entropy_loss,
    global_step,
    epoch,
    last_log_time,
    token_per_step,
    is_accumulation_step,
    pbar,
    grad_norm=0.0,
):
    global total_loss, total_updates, trained_tokens, avg_loss

    current_time = time.time()
    elapsed_time = max(current_time - last_log_time, 1e-6)
    steps_per_second = 1.0 / elapsed_time
    kt_s = token_per_step * steps_per_second / 1000.0

    total_loss += loss
    total_updates += 1
    avg_loss = total_loss / total_updates

    trained_tokens += token_per_step

    if is_accumulation_step and model_engine.global_rank == 0:
        if pbar is None:
            pbar = tqdm(total=args.epoch_steps, desc=f"Epoch {epoch}")

        pbar.update(1)
        pbar.set_postfix(
            {
                'loss': f'{avg_loss:.4f}',
                'steps/s': f'{steps_per_second:.2f}',
                'kt/s': f'{kt_s:.2f}',
                'trained_tokens': f'{trained_tokens / 1e6:.2f} MT',
                'remained_tokens': f'{(args.max_trained_tokens - trained_tokens) / 1e6:.2f} MT',
            }
        )
        timer.print_stats(global_step)
        if args.wandb:
            wandb.log(
                {
                    "loss": loss,
                    "lr": model_engine.optimizer.param_groups[0]['lr'],
                    "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                    "weight_decay": model_engine.optimizer.param_groups[0]['weight_decay'],
                    "steps_per_second": steps_per_second,
                    "kt/s": kt_s,
                    "global_step": global_step,
                    "Gtokens": global_step * token_per_step * args.accumulate_grad_batches / 1e9,
                    "epoch": epoch,
                    "teacher_loss": teacher_loss,
                    "kl_loss": kl_loss,
                    "student_cross_entropy_loss": student_cross_entropy_loss,
                }
            )

    real_step = batch_idx
    if real_step % args.save_per_batches == 0 and real_step > 0:
        if os.path.exists(args.output_dir):
            if model_engine.local_rank == 0:
                checkpoints = os.listdir(args.output_dir)
                checkpoints = [
                    f for f in checkpoints
                    if os.path.isdir(os.path.join(args.output_dir, f))
                ]
                checkpoints.sort(
                    key=lambda x: os.path.getctime(os.path.join(args.output_dir, x))
                )
                if len(checkpoints) > 2:
                    print(f'deleting older checkpoints {checkpoints[0]}')
                    import shutil

                    shutil.rmtree(os.path.join(args.output_dir, checkpoints[0]))

        output_dir = f"{args.output_dir}/epoch_{epoch}_step_{real_step}"
        print(f'saving checkpoint to {output_dir}')

        if model_engine.global_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model_engine.module
                if hasattr(model_engine, "module")
                else model_engine
            )

            full_state_dict = model_to_save.state_dict()
            trainable_keys = {
                name for name, param in model_to_save.named_parameters()
                if param.requires_grad
            }
            filtered_state_dict = {
                k: v for k, v in full_state_dict.items() if k in trainable_keys
            }

            save_path = os.path.join(output_dir, "trainable_only_weights.pth")
            try:
                torch.save(filtered_state_dict, save_path)
                print(f"✅ Saved trainable-only weights to {save_path}")
            except Exception as e:
                print(f"Error saving weights: {e}")
                import traceback

                traceback.print_exc()

    return current_time, pbar


# ============================================================
#  TeacherAttn 管理
# ============================================================

import contextlib


class TeacherAttnManager:
    def __init__(self, model_engine, layers: List[int]):
        self.model_engine = model_engine
        self.layers = layers
        self.stored_teacher_attns = {}
        self.stored_vfirst_state = {}
        self.stored_kfirst_state = {}

    @contextlib.contextmanager
    def temporarily_remove_teacher_attn(self):
        try:
            for layer_idx in self.layers:
                attention_wrapper = (
                    self.model_engine.module.model.model.layers[layer_idx].self_attn
                )
                if hasattr(attention_wrapper, 'teacher_attn'):
                    self.stored_teacher_attns[layer_idx] = attention_wrapper.teacher_attn
                    if (
                        hasattr(attention_wrapper, '_modules')
                        and 'teacher_attn' in attention_wrapper._modules
                    ):
                        del attention_wrapper._modules['teacher_attn']
                    attention_wrapper.teacher_attn = None
                if hasattr(attention_wrapper, 'v_first_state'):
                    self.stored_vfirst_state[layer_idx] = attention_wrapper.v_first_state
                    attention_wrapper.v_first_state = None
                if hasattr(attention_wrapper, 'k_first_state'):
                    self.stored_kfirst_state[layer_idx] = attention_wrapper.k_first_state
                    attention_wrapper.k_first_state = None

            yield
        finally:
            for layer_idx, stored_attn in self.stored_teacher_attns.items():
                attention_wrapper = (
                    self.model_engine.module.model.model.layers[layer_idx].self_attn
                )
                attention_wrapper.teacher_attn = stored_attn
                if hasattr(attention_wrapper, 'add_module'):
                    attention_wrapper.add_module("teacher_attn", stored_attn)
                v_first_state = self.stored_vfirst_state.get(layer_idx, None)
                k_first_state = self.stored_kfirst_state.get(layer_idx, None)
                if v_first_state is not None:
                    attention_wrapper.v_first_state = v_first_state
                if k_first_state is not None:
                    attention_wrapper.k_first_state = k_first_state

            self.stored_teacher_attns.clear()
            self.stored_vfirst_state.clear()
            self.stored_kfirst_state.clear()


# ============================================================
#  メイン
# ============================================================

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' in os.environ and args.local_rank is None:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    deepspeed.init_distributed()  # NCCL/RCCl 初期化

    if args.world_size is None:
        args.world_size = dist.get_world_size()

    print(args)

    # --- コンフィグ読み込み ---
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    DeviceID = f'cuda:{args.local_rank}'
    args.DeviceID = DeviceID

    dtype = torch.bfloat16

    # ===============================
    # DeepSpeed ZeRO-3 Config 構築
    # ===============================
    args.deepspeed_offload = False
    if args.deepspeed:
        if args.deepspeed_config:
            with open(args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
        else:
            reduce_bucket_size = int(args.ds_bucket_mb * 1024 * 1024)

            ds_config = {
                "train_batch_size": args.train_batch_size,
                "train_micro_batch_size_per_gpu": args.micro_bsz,
                "gradient_accumulation_steps": args.accumulate_grad_batches,
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": args.deepspeed_stage,  # デフォルト 3
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": reduce_bucket_size,
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "stage3_prefetch_bucket_size": 1e9,
                    "stage3_param_persistence_threshold": 1e6,
                    "gather_16bit_weights_on_model_save": True,
                },
                "gradient_clipping": args.gradient_clip_val,
                "gradient_checkpointing": args.grad_cp == 1,
                "wall_clock_breakdown": False,
                "zero_allow_untested_optimizer": True,
                "distributed_training": {
                    "deepspeed_multinode_launcher": "standard",
                },
            }

            if args.deepspeed_offload:
                ds_config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
                ds_config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }

        # ★ Transformers 側に ZeRO-3 + zero.Init を伝える
        dschf = HfDeepSpeedConfig(ds_config)

    # ===============================
    # モデル / トークナイザ ロード
    # ===============================

    # この from_pretrained 呼び出しは、上の HfDeepSpeedConfig により、
    # 内部で deepspeed.zero.Init() コンテキストが有効化されます。
    # → 各 Parameter は構築時点で ZeRO-3 shard として分割され、
    #    「1GPU にフルモデルをロードしてから分割」する OOM パターンを回避:contentReference[oaicite:2]{index=2}
    transformer_model = AutoModelForCausalLM.from_pretrained(
        config['Llama']['model_id'],
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    args.freeze_attention = config['freeze_attention']
    args.hybrid_attention_layers = config['hybrid_attention_layers']
    args.freeze_hybrid_attention = config['freeze_hybrid_attention']
    args.allow_quant_frozen_layers = config['allow_quant_frozen_layers']
    args.quant_mode = "none"
    args.peftmode = config['peftmode']
    args.peft_r = config['peft_r']
    args.peft_scaling = config['peft_scaling']
    args.peft_dropout = config['peft_dropout']
    args.mlp_quant_mode = "none"
    args.bnb_optimizer_mode = 0
    args.transformer_layers = config['RWKV']['transformer_layers']
    args.disable_qk_norm = config['disable_qk_norm']

    args.architecture = config.get('architecture', 'hxa079')
    os.environ["architecture"] = args.architecture

    tokenizer = AutoTokenizer.from_pretrained(config['Llama']['model_id'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    args.my_pos_emb = 0
    args.ctx_len = 4096
    args.n_layer = transformer_model.config.text_config.num_hidden_layers
    args.n_embd = transformer_model.config.text_config.hidden_size

    args.dim_ffn = transformer_model.config.text_config.intermediate_size
    args.config = transformer_model.config
    args.num_attention_heads = transformer_model.config.text_config.num_attention_heads
    args.num_key_value_heads = transformer_model.config.text_config.num_key_value_heads
    args.rms_norm_eps = transformer_model.config.text_config.rms_norm_eps
    args.head_size_a = getattr(
        transformer_model.config,
        'head_dim',
        transformer_model.config.text_config.hidden_size // transformer_model.config.text_config.num_attention_heads,
    )
    args.dim_att = transformer_model.config.text_config.num_attention_heads * args.head_size_a
    args.is_attention_bias = getattr(transformer_model.config, 'attention_bias', True)
    args.is_attention_output_bias = getattr(
        transformer_model.config, 'attention_output_bias', False
    )
    args.pre_ffn = 0
    args.head_qk = 0
    args.tiny_att_dim = 0
    args.tiny_att_layer = -999
    args.vocab_size = transformer_model.config.text_config.vocab_size
    args.layers = config['RWKV']['layers']
    args.pad_id = tokenizer.eos_token_id
    args.betas = (args.beta1, args.beta2)
    args.kl_weight = config['kl_weight']
    args.ce_weight = config['ce_weight']
    args.enable_AKL = config.get('enable_AKL', False)
    args.model_file = config['model_file']
    args.real_bsz = args.train_batch_size
    args.is_sft = config.get('is_sft', False)
    args.is_all_labels_kl = config.get('is_all_labels_kl', False)

    print(f'num_hidden_layers: {transformer_model.config.text_config.num_hidden_layers}')

    # Teacher Self-Attn モジュールの参照を保持（実体は transformer_model 内部）
    teacher_attn_module_list = torch.nn.ModuleList()
    for layer_idx in range(transformer_model.config.text_config.num_hidden_layers):
        llama_layer = transformer_model.model.language_model.layers[layer_idx]
        teacher_attn_module_list.append(llama_layer.self_attn)
    for n, p in teacher_attn_module_list.named_parameters():
        p.requires_grad = False

    os.environ["RWKV_HEAD"] = str(int(args.n_embd // args.head_size_a))
    os.environ["RWKV_HEAD_SIZE_A"] = str(int(args.head_size_a))
    os.environ["RWKV_MIRCO_BSZ"] = str(int(args.micro_bsz))

    os.environ['RWKV_ATTN_PEFTMODE'] = str(args.peftmode)
    os.environ['RWKV_ATTN_QUANT'] = "none"
    os.environ['RWKV_ATTN_PEFT_R'] = str(args.peft_r)
    os.environ['RWKV_ATTN_PEFT_SCALING'] = str(args.peft_scaling)
    os.environ['RWKV_ATTN_PEFT_DROPOUT'] = str(args.peft_dropout)

    from hybrid_model_vl import HybridModel, VFirstHolder, KFirstHolder

    # HybridModel も zero.Init で構築された transformer_model を中に抱える
    model = HybridModel(transformer_model, args, tokenizer)

    # ====== 教師重みコピー部分（元コードそのまま保持） ======
    def SearchTensor(mod, keyname):
        for name, param in mod.named_parameters():
            if keyname in name:
                return param
        return None

    weight_mul_r = 1.0
    weight_mul_k = 1.0
    weight_mul_v = 1.0
    weight_mul_o = 1.0

    with torch.no_grad():
        for i in range(args.n_layer):
            if i in args.transformer_layers:
                weight_mul_r = weight_mul_k = weight_mul_v = weight_mul_o = 1.0
            else:
                weight_mul_r = 0.5
                weight_mul_k = 0.5
                weight_mul_v = 0.3
                weight_mul_o = 0.5

            print(f'layer = {i} transfer to student')

            for name, param in model.named_parameters():
                if f'model.language_model.layers.{i}.self_attn.student_attn' not in name:
                    continue

                # 以下、q/k/v/o + norm のコピーは元コードと同等に残す
                if 'receptance.weight' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.q_proj.weight')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_r)
                        print(f'{name}: copied from teacher q_proj.weight')
                elif 'receptance.bias' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.q_proj.bias')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_r)
                        print(f'{name}: copied from teacher q_proj.bias')

                if 'key.weight' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.k_proj.weight')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_k)
                        print(f'{name}: copied from teacher k_proj.weight')
                elif 'key.bias' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.k_proj.bias')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_k)
                        print(f'{name}: copied from teacher k_proj.bias')

                if 'value.weight' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.v_proj.weight')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_v)
                        print(f'{name}: copied from teacher v_proj.weight')
                elif 'value.bias' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.v_proj.bias')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_v)
                        print(f'{name}: copied from teacher v_proj.bias')

                if 'output.weight' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.o_proj.weight')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_o)
                        print(f'{name}: copied from teacher o_proj.weight')
                elif 'output.bias' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.o_proj.bias')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_o)
                        print(f'{name}: copied from teacher o_proj.bias')

                if 'q_proj.weight' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.q_proj.weight')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_r)
                        print(f'{name}: copied from teacher q_proj.weight')
                elif 'q_proj.bias' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.q_proj.bias')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_r)
                        print(f'{name}: copied from teacher q_proj.bias')

                if 'k_proj.weight' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.k_proj.weight')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_k)
                        print(f'{name}: copied from teacher k_proj.weight')
                elif 'k_proj.bias' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.k_proj.bias')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_k)
                        print(f'{name}: copied from teacher k_proj.bias')

                if 'v_proj.weight' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.v_proj.weight')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_v)
                        print(f'{name}: copied from teacher v_proj.weight')
                elif 'v_proj.bias' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.v_proj.bias')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_v)
                        print(f'{name}: copied from teacher v_proj.bias')

                if 'o_proj.weight' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.o_proj.weight')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_o)
                        print(f'{name}: copied from teacher o_proj.weight')
                elif 'o_proj.bias' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.o_proj.bias')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s * weight_mul_o)
                        print(f'{name}: copied from teacher o_proj.bias')

                if 'r_norm.weight' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.q_norm.weight')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s)
                        print(f'{name}: copied from teacher q_norm.weight')
                if 'q_norm.weight' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.q_norm.weight')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s)
                        print(f'{name}: copied from teacher q_norm.weight')
                if 'k_norm.weight' in name:
                    s = SearchTensor(teacher_attn_module_list, f'{i}.k_norm.weight')
                    if s is not None and s.shape == param.shape:
                        param.copy_(s)
                        print(f'{name}: copied from teacher k_norm.weight')

    if args.ckpt_file is not None:
        model.load_check_point(args.ckpt_file)

    print(f'Stage1 Only self_attn params are trainable')

    # ====== requires_grad の設定 ======
    for name, param in model.named_parameters():
        Attention = 0
        for i in range(args.n_layer):
            t = f'layers.{i}.'
            if t in name and i in args.transformer_layers:
                Attention = 1
                break
            elif t in name:
                Attention = 0
                break

        if (
            Attention == 0
            and args.freeze_attention
            and (
                'self_attn.student_attn' in name
                and any(k in name for k in ['receptance', 'key', 'value'])
            )
        ):
            param.requires_grad = False
        elif (
            Attention == 1
            and args.freeze_hybrid_attention
            and (
                'self_attn.student_attn' in name
                and any(k in name for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm'])
            )
        ):
            param.requires_grad = False
        elif 'self_attn.student_attn' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # LoRA/Bone 用の requires_grad 調整（元コードに近いが量子化はしない）
    lora_base_modules = set()
    if args.peftmode != 'full':
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                base_module_name = name.rsplit('.lora_', 1)[0]
                lora_base_modules.add(base_module_name)
                param.requires_grad = True
            elif 'bone' in name:
                base_module_name = name.rsplit('.bone', 1)[0]
                lora_base_modules.add(base_module_name)
                param.requires_grad = True

        for name, param in model.named_parameters():
            for base_module in lora_base_modules:
                if name == f"{base_module}.weight":
                    param.requires_grad = False
                elif name == f"{base_module}.bias":
                    param.requires_grad = False

    print("\n=== Final trainable parameters ===")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  {name}: {param.shape}")

    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"\nTrainable params: {trainable_params:,} / Total params: {total_params:,} "
        f"({trainable_params / total_params * 100:.2f}%)"
    )
    print('GPU memory BEFORE DeepSpeed init:')
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print(measure_model_memory(model))

    # ===============================
    # データローダ準備
    # ===============================

    if args.preprocessed_data is not None:
        print(f'load preprocessed data from {args.preprocessed_data}')
        from data.multi_source_datasets import data_collator_with_pad
        from functools import partial

        pad_token_id = tokenizer.pad_token_id
        collator = partial(
            data_collator_with_pad,
            max_seq_length=args.max_seq_length,
            pad_token_id=pad_token_id,
        )

        train_datasets = []
        for data_path in args.preprocessed_data:
            ds = datasets.load_from_disk(data_path)
            train_datasets.append(ds)

        train_ds = datasets.concatenate_datasets(train_datasets)

        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.micro_bsz,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=collator,
        )
        val_dataloader = None
        if args.local_rank == 0:
            print('load preprocessed data done')
    elif args.raw_data is not None:
        args.raw_data = args.raw_data[0].split(",")
        print(f'load raw data from {args.raw_data}')
        from data.raw_dataset import (
            load_datasets_from_directories,
            TypedDataset,
            TypedStreamingCLMDataCollator,
        )

        all_ds, feature_types = load_datasets_from_directories(args.raw_data, tokenizer)
        typed_dataset = TypedDataset(all_ds, feature_types)
        data_collator = TypedStreamingCLMDataCollator(
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            min_length=args.max_seq_length,
            typed_dataset=typed_dataset,
            need_to_pad=args.need_to_pad,
        )

        train_sampler = DistributedSampler(
            typed_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True,
        )
        train_dataloader = torch.utils.data.DataLoader(
            typed_dataset,
            batch_size=args.micro_bsz,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=data_collator,
        )
        val_dataloader = None
    else:
        print("No dataset specified. EXIT.")
        sys.exit(1)

    # ===============================
    # DeepSpeed 初期化
    # ===============================

    if not args.deepspeed:
        print('not using deepspeed, EXIT')
        sys.exit(1)

    print(f'Configuring optimizer with args {args}')
    optimizer = configure_optimizer(model, args)

    if args.local_rank == 0:
        print(f'optimizer is {optimizer}')
        num_total_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(
            f'num_total_params: {num_total_params}, '
            f'num_trainable_params: {num_trainable_params}, '
            f'percent: {num_trainable_params / num_total_params * 100:.2f}%'
        )
        print('GPU memory BEFORE DeepSpeed init (again):')
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    trainable_params_iter = (p for p in model.parameters() if p.requires_grad)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=trainable_params_iter,
        optimizer=optimizer,
        config=ds_config,
    )

    # DeepSpeed Engine 内部に全て移管したので生モデルを削除
    del model
    gc.collect()
    torch.cuda.empty_cache()

    for name, m in model_engine.module.model.named_parameters():
        print(f'{name} requires_grad = {m.requires_grad}')

    print('wait 5sec after DeepSpeed init')
    time.sleep(5)

    # v_first / k_first は DeepSpeed 管理外の補助バッファなので、
    # DeepSpeed 初期化の「後」でローカル GPU に確保する
    vfirst_holder = VFirstHolder(
        args.micro_bsz,
        args.max_seq_length,
        args.num_key_value_heads,
        args.head_size_a,
        device=torch.device(DeviceID),
    )
    vfirst_holder.requires_grad_(False)

    kfirst_holder = KFirstHolder(
        args.micro_bsz,
        args.max_seq_length,
        args.num_key_value_heads,
        args.head_size_a,
        device=torch.device(DeviceID),
    )
    kfirst_holder.requires_grad_(False)

    print('ZeRO-3 will shard the model; set v_first/k_first holder to each target layer')
    for layer_idx in args.layers:
        attn_wrapper = model_engine.module.model.model.layers[layer_idx].self_attn
        attn_wrapper.v_first_state = vfirst_holder
        attn_wrapper.k_first_state = kfirst_holder

    timer.initialize_with_engine(model_engine)

    if args.local_rank == 0:
        print('GPU memory AFTER DeepSpeed init:')
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Stage1: teacher_attn を wrapper に刺す（teacher_engine は単なる参照）
    if args.stage == 1:
        teacher_engine = teacher_attn_module_list
        for layer_idx in args.layers:
            if args.local_rank == 0:
                print(f'set teacher attn for layer {layer_idx}')
            attention_wrapper = model_engine.module.model.model.layers[layer_idx].self_attn
            teacher_attn = teacher_engine[layer_idx]
            attention_wrapper.teacher_attn = teacher_attn
            attention_wrapper.add_module("teacher_attn", teacher_attn)
        torch.cuda.empty_cache()
        if args.local_rank == 0:
            print('GPU memory AFTER attaching teacher_attn:')
            print(torch.cuda.memory_summary(device=None, abbreviated=False))
    else:
        teacher_engine = None

    # ===============================
    # wandb
    # ===============================
    if args.wandb and model_engine.global_rank == 0:
        print(f'init wandb, project={args.wandb}, name={args.run_name}')
        wandb.init(project=args.wandb, name=args.run_name, config=vars(args))
        print(f'begin training with {args.max_epochs} epochs')

    # ===============================
    # トレーニングループ
    # ===============================

    args.epoch_steps = len(train_dataloader) // max(args.accumulate_grad_batches, 1)
    global_step = 0
    last_log_time = time.time()
    token_per_step = args.max_seq_length * args.micro_bsz * args.world_size

    terminate = False
    teacher_attn_manager = TeacherAttnManager(model_engine, args.layers)

    gc.collect()
    torch.cuda.empty_cache()

    for epoch in range(args.max_epochs):
        model_engine.train()
        if model_engine.global_rank == 0:
            pbar = tqdm(total=args.epoch_steps, desc=f"Epoch {epoch}")
        gc.collect()
        torch.cuda.empty_cache()

        for batch_idx, batch in enumerate(train_dataloader):
            if terminate:
                break

            lr, wd_now = on_train_batch_start(args, model_engine, global_step, epoch)

            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            loss, teacher_loss, kl_loss, student_cross_entropy_loss = train_step(
                model_engine,
                batch,
                args,
                teacher_engine,
                tokenizer,
                global_step=global_step,
                log_path=os.path.join(args.output_dir, 'stage1log.csv'),
            )

            model_engine.backward(loss)

            is_accumulation_step = (batch_idx + 1) % args.accumulate_grad_batches == 0
            grad_norm = None
            if is_accumulation_step:
                global_step += 1
                try:
                    grad_norm = model_engine.get_global_grad_norm()
                except AttributeError:
                    grad_norm = None

            model_engine.step()

            last_log_time, pbar = on_train_batch_end(
                args,
                batch_idx,
                model_engine,
                teacher_engine,
                loss.item(),
                teacher_loss,
                kl_loss,
                student_cross_entropy_loss,
                global_step,
                epoch,
                last_log_time,
                token_per_step,
                is_accumulation_step,
                pbar,
                grad_norm=grad_norm,
            )

            if trained_tokens >= args.max_trained_tokens:
                terminate = True
                break

        if args.output_dir:
            if args.deepspeed:
                with teacher_attn_manager.temporarily_remove_teacher_attn():
                    try:
                        print(
                            f"Saving checkpoint to {args.output_dir} at epoch {epoch} "
                            f"rank {model_engine.global_rank}"
                        )
                        model_engine.save_checkpoint(
                            args.output_dir, f"checkpoint-epoch{epoch}"
                        )
                    except Exception as e:
                        print(f"Error saving checkpoint: {e}")
                        import traceback

                        traceback.print_exc()

        if terminate:
            break

    if model_engine.global_rank == 0:
        print("Training finished.")
