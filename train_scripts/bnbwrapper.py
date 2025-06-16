
# import torch
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import bitsandbytes as bnb
# from dataclasses import dataclass
# from typing import Optional
# import gc

# import bitsandbytes as bnb
# from bitsandbytes.functional import QuantState

# @dataclass
# class QuantizedState:
#     """量子化された重みと状態を保持"""
#     qweight: torch.Tensor  # 量子化された重み (int8)
#     scales: tuple   # スケール
#     zeros: Optional[torch.Tensor] = None  # ゼロポイント
#     bias: Optional[torch.Tensor] = None   # バイアス
#     shape: tuple = None    # 元の形状

# class QuantizedLinearWrapper(nn.Module):
#     """
#     DeepSpeedから量子化状態を保護するカスタムLinear層
#     """
#     def __init__(self, in_features: int, out_features: int, bias: bool = True):
#         super().__init__()
#         print('Creating Quanztized Linear Wrapper')
#         self.in_features = in_features
#         self.out_features = out_features
        
#         # ダミーパラメータ（DeepSpeedに認識させるため）
#         self.dummy_weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        
#         # 量子化状態はバッファとして保存
#         # self.register_buffer('qweight', None)
#         # self.register_buffer('scales', None)
#         # self.register_buffer('zeros', None)
        
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
#         else:
#             self.register_parameter('bias', None)

#         print('Creating Quanztized Linear Wrapper End')
            
#     def set_quantized_state(self, state: QuantizedState):
#         """量子化された状態を設定"""
#         print('qweight move')
#         self.qweight = state.qweight
#         print('scales move')
#         self.scales = state.scales
#         if state.zeros is not None:
#             self.zeros = state.zeros
#         if state.bias is not None and self.bias is not None:
#             self.bias.data = state.bias
            
#     def forward(self, x):
#         """推論時に量子化された重みを使用"""
#         # if self.training:
#         #     # トレーニング中はダミー出力（このレイヤーは凍結されているはず）
#         #     return torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
#         # else:
#         # 推論時: 量子化された重みを使って計算
#         # Int8をFP16に展開
#         #weight_fp16 = self.qweight.to(x.dtype) * self.scales

#         deweight= bnb.functional.dequantize(self.qweight,state=self.scales).to(torch.bfloat16)
        
#         # 通常のlinear演算
#         #output = F.linear(x, weight_fp16.reshape(self.out_features, self.in_features), self.bias)
#         output = F.linear(x, deweight, self.bias)
#         return output

# def quantize_linear_to_int8(linear_module, device):
#     """
#     Linear層を手動でInt8量子化
#     """
#     with torch.no_grad():
#         # FP16の重みを取得
#         w = linear_module.weight.data.to(torch.bfloat16).to(device)
        
#         # # 量子化のスケールを計算
#         # w_abs_max = w.abs().max()
#         # scales = w_abs_max / 127.0
        
#         # # Int8に量子化
#         # qweight = torch.round(w / scales).clamp(-128, 127).to(torch.int8)
#         qweight, scales= bnb.functional.quantize((w))

#         #print(f'qweight = {qweight.shape} scales = {scales.shape}')

#         print('Quanted')
        
#         # 状態を作成
#         state = QuantizedState(
#             qweight=qweight,#.flatten(),  # フラット化して保存
#             scales=scales,
#             bias=linear_module.bias.data.clone() if linear_module.bias is not None else None,
#             shape=w.shape
#         )

#         print('state savaed')

#         #del linear_module.weight
        
#     return state

# def quantize_and_replace_with_wrapper(model, patterns=['mlp'], threshold=0):
#     """
#     指定パターンのLinear層を量子化してカスタムラッパーに置き換え
#     """
#     device = next(model.parameters()).device
#     modules_to_replace = []
    
#     # 置き換え対象を特定
#     for name, module in model.named_modules():
#         if any(pattern in name for pattern in patterns):
#             if isinstance(module, nn.Linear) and module.weight.numel() > threshold:
#                 modules_to_replace.append((name, module))
#                 print(f"Will quantize and wrap: {name} (shape: {module.weight.shape})")
    
#     # 実際の置き換え
#     for name, module in modules_to_replace:
#         try:
#             # 1. 手動で量子化
#             quantized_state = quantize_linear_to_int8(module, device)

#             print('create custom wrapper')
            
#             # 2. カスタムラッパーを作成
#             wrapper = QuantizedLinearWrapper(
#                 module.in_features,
#                 module.out_features,
#                 bias=module.bias is not None
#             )#.to(device)
#             #print(f'wrapper device = {wrapper.device}')
#             print('set quantized state')
#             # 3. 量子化状態を設定
#             wrapper.set_quantized_state(quantized_state)
            
#             # 4. モジュールを置き換え
#             parent_name = '.'.join(name.split('.')[:-1])
#             child_name = name.split('.')[-1]
            
#             if parent_name:
#                 parent = model
#                 for part in parent_name.split('.'):
#                     parent = getattr(parent, part)
#             else:
#                 parent = model
                
#             setattr(parent, child_name, wrapper)
#             print(f"Successfully quantized: {name}")
            
#         except Exception as e:
#             print(f"Error quantizing {name}: {e}")
#             continue
            
#         # メモリクリーンアップ
#         del module
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     print('finished')
    
#     return model



import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from dataclasses import dataclass
from typing import Optional
import gc

@dataclass
class QuantizedState:
    """量子化された重みと状態を保持"""
    qweight: torch.Tensor  # 量子化された重み (int8)
    scales: torch.Tensor   # スケール
    zeros: Optional[torch.Tensor] = None  # ゼロポイント
    bias: Optional[torch.Tensor] = None   # バイアス
    shape: tuple = None    # 元の形状

class QuantizedLinearWrapper(nn.Module):
    """
    DeepSpeedから量子化状態を保護するカスタムLinear層
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # ダミーパラメータ（DeepSpeedに認識させるため）
        self.dummy_weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        
        # 量子化状態はバッファとして保存
        self.register_buffer('qweight', None)
        self.register_buffer('scales', None)
        self.register_buffer('zeros', None)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter('bias', None)
            
    def set_quantized_state(self, state: QuantizedState):
        """量子化された状態を設定"""
        self.qweight = state.qweight
        self.scales = state.scales
        if state.zeros is not None:
            self.zeros = state.zeros
        if state.bias is not None and self.bias is not None:
            self.bias.data = state.bias
            
    def forward(self, x):
        """推論時に量子化された重みを使用"""
        # if self.training:
        #     # トレーニング中はダミー出力（このレイヤーは凍結されているはず）
        #     return torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
        # else:
        # 推論時: 量子化された重みを使って計算
        # Int8をFP16に展開
        weight_fp16 = self.qweight.to(x.dtype) * self.scales
        
        # 通常のlinear演算
        output = F.linear(x, weight_fp16.reshape(self.out_features, self.in_features), self.bias)
        return output

def quantize_linear_to_int8(linear_module, device):
    """
    Linear層を手動でInt8量子化
    """
    with torch.no_grad():
        # FP16の重みを取得
        w = linear_module.weight.data.to(torch.bfloat16).to(device)
        
        # 量子化のスケールを計算
        w_abs_max = w.abs().max()
        scales = w_abs_max / 127.0
        
        # Int8に量子化
        qweight = torch.round(w / scales).clamp(-128, 127).to(torch.int8)
        
        # 状態を作成
        state = QuantizedState(
            qweight=qweight.flatten(),  # フラット化して保存
            scales=scales,
            bias=linear_module.bias.data.clone() if linear_module.bias is not None else None,
            shape=w.shape
        )

        del w
        
    return state

def quantize_and_replace_with_wrapper(model, patterns=['mlp'], threshold=0):
    """
    指定パターンのLinear層を量子化してカスタムラッパーに置き換え
    """
    device = next(model.parameters()).device
    modules_to_replace = []
    
    # 置き換え対象を特定
    for name, module in model.named_modules():
        if any(pattern in name for pattern in patterns):
            if isinstance(module, nn.Linear) and module.weight.numel() > threshold:
                modules_to_replace.append((name, module))
                print(f"Will quantize and wrap: {name} (shape: {module.weight.shape})")
    
    # 実際の置き換え
    for name, module in modules_to_replace:
        try:
            # 1. 手動で量子化
            quantized_state = quantize_linear_to_int8(module, device)
            
            # 2. カスタムラッパーを作成
            wrapper = QuantizedLinearWrapper(
                module.in_features,
                module.out_features,
                bias=module.bias is not None
            ).to(device)
            
            # 3. 量子化状態を設定
            wrapper.set_quantized_state(quantized_state)
            
            # 4. モジュールを置き換え
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            else:
                parent = model
                
            setattr(parent, child_name, wrapper)
            print(f"Successfully quantized: {name}")
            
        except Exception as e:
            print(f"Error quantizing {name}: {e}")
            continue
            
        # メモリクリーンアップ
        del module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return model


import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
from typing import List, Optional, Dict
import gc

def replace_linear_with_8bit(module, threshold=6.0, has_fp16_weights=False,device=None):
    """
    Linear層を8bit Linear層に正しく置き換える
    """
    # 新しい8bit Linearモジュールを作成
    new_module = bnb.nn.Linear8bitLt(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        has_fp16_weights=has_fp16_weights,
        threshold=threshold
    )
    
    # 重みを8bitモジュールに移動（この時点で量子化される）
    new_module.weight.data = module.weight.data#.to(dtype=torch.float32,device=device)
    if module.bias is not None:
        new_module.bias.data = module.bias.data
    
    # 新しいモジュールをCUDAに移動
    #new_module = new_module.cuda()
    
    return new_module

def quantize_mlp_layers_properly(model, threshold=6.0,device=None):
    """
    LLaMA/Mistral系モデルのMLP層を正しく8ビット量子化
    """
    # 量子化する層のパスを収集
    layers_to_quantize = []
    
    for name, module in model.named_modules():
        # MLP層の判定（LLaMA/Mistral系のMLP構造）
        is_mlp = any(pattern in name for pattern in [
            'mlp.gate_proj',  # ゲート投影層
            'mlp.down_proj',  # ダウン投影層
            'mlp.up_proj',    # アップ投影層
            # 他のモデル構造も対応
            'mlp.c_fc', 'mlp.c_proj',  # GPT-2
            'mlp.fc_in', 'mlp.fc_out',  # GPT-Neo
            'mlp.dense_h_to_4h', 'mlp.dense_4h_to_h'  # GPT-J
        ])
        
        if is_mlp and isinstance(module, nn.Linear):
            layers_to_quantize.append(name)
    
    print(f"量子化対象のMLP層: {len(layers_to_quantize)}個")
    
    # 各層を順番に量子化
    for layer_name in layers_to_quantize:
        # モジュールのパスを分解
        parts = layer_name.split('.')
        parent = model
        
        # 親モジュールを取得
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # 現在のモジュールを取得
        old_module = getattr(parent, parts[-1])
        
        # 8bitモジュールに置き換え
        print(f"量子化中: {layer_name}")
        new_module = replace_linear_with_8bit(old_module, threshold=threshold,device=device)
        
        # モジュールを置き換え
        setattr(parent, parts[-1], new_module)
        
        # 元のモジュールを削除してメモリを解放
        del old_module
        torch.cuda.empty_cache()
    
    return model