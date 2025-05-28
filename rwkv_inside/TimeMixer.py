import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load
#HEAD_SIZE = 64
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
print(f'parent_dir: {parent_dir}')
is_wind_cuda = False

#from tritonbighead import RUN_CUDA_RWKV7g
from backstepping_longhead import RUN_CUDA_RWKV7g



    
class RWKV_Tmix_x070_Mose(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

          

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            #D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)


            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)


            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()
            

    
    #@torch.compile
    def forward(self, x, v_first,attention_mask):
        B, T, C = x.size()
        #removed tokenshift
        H = self.n_head
        r = self.receptance(x)
        w = -F.softplus(-(self.w0 + torch.tanh(x @ self.w1) @ self.w2)) - 0.6 # soft-clamp to (-inf, -0.5) modified -0.5->-0.6
        k = self.key(x)
        v = self.value(x)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)

        x = x.view(B, T, C)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x) 
        return x, v_first







class RWKV_Tmix_x070_Mose_v2(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

          

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            #D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)


            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            self.out_scale = nn.Parameter(torch.tensor(-11.5129))


            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()
            

    
    #@torch.compile
    def forward(self, x, v_first,attention_mask):
        B, T, C = x.size()
        #removed tokenshift
        H = self.n_head
        r = self.receptance(x)
        w = -F.softplus(-(self.w0 + torch.tanh(x @ self.w1) @ self.w2)) - 0.6 # soft-clamp to (-inf, -0.5) modified -0.5->-0.6
        k = self.key(x)
        v = self.value(x)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C) # Groupnorm is back!

        #x = x.view(B, T, C)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x) * (torch.sigmoid(self.out_scale)*1e5) #hopefully absorb MLP scale
        return x, v_first
    










def repeat_kv_original(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads along the head dimension (GQA).
    Input:  (B, T, H_kv, D)
    Output: (B, T, H_kv * n_rep, D)
    """
    B, T, H_kv, D = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # Expand head dim
    hidden_states = hidden_states[:, :, :, None, :]  # (B, T, H_kv, 1, D)
    hidden_states = hidden_states.expand(B, T, H_kv, n_rep, D)  # (B, T, H_kv, n_rep, D)
    return hidden_states.reshape(B, T, H_kv * n_rep, D).contiguous()


def is_nan(t, name):
    if torch.isnan(t).any():
        return True 
    else:
        return False
    # if torch.isnan(t).any():
    #     print(f"⚠️ NaN detected in {name}")
    # if torch.isinf(t).any():
    #     print(f"⚠️ Inf detected in {name}")
    # if t.abs().max() > 1e4:
    #     print(f"⚠️ Large value in {name}: max={t.abs().max().item()}")

def check_abs_max(t, name, threshold=1e4):
    max_abs = t.abs().max().item()
    print(f"[{name}] abs max: {max_abs:.4e}")
    if max_abs > threshold:
        print(f"⚠️ WARNING: {name} has large value (>{threshold}) → may cause NaN soon!")

class SafeGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-4):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = x.clamp(min=-1e4, max=1e4)
        out = self.norm(x)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out
class RWKV_Tmix_x070_Mose_cxa073(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size


        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads



        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.layer_architecture = nn.Parameter(torch.tensor(73.0))

        
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))


            #Change to GQA Style
            self.receptance = nn.Linear(C, self.num_attention_heads * self.head_size, bias=self.args.is_attention_bias)
            self.key = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.value = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.output = nn.Linear(self.num_attention_heads * self.head_size, C, bias=self.args.is_attention_output_bias)
            #SafeGroupNorm
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            #self.ln_x = SafeGroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            #self.out_scale = nn.Parameter(torch.tensor(-11.5129))

            #later copy from teacher weights
            #maybe can copy from any GQA Models
            self.receptance.weight.data.zero_()
            self.key.weight.data.zero_()
            self.value.weight.data.zero_()
            self.output.weight.data.zero_()
           

    
    #@torch.compile
    def forward(self, x, v_first,attention_mask):
        B, T, C = x.size()
        #removed tokenshift
        H = self.n_head
        r = self.receptance(x)
        w = -F.softplus(-(self.w0 + torch.tanh(x @ self.w1) @ self.w2)) -0.6#- 0.6 # soft-clamp to (-inf, -0.5) modified -0.5->-0.6
        k = self.key(x)
        v = self.value(x)


        k = k.view(B, T, self.num_key_value_heads, self.head_size)
        v = v.view(B, T, self.num_key_value_heads, self.head_size)

        # repeat k/v heads if n_kv_heads < n_heads
        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, self.num_key_value_groups)#reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, self.num_key_value_groups)#reshape(B,T,-1) #(B,T,C)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        #so now all B,T,C tensors

        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual



        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        # check_abs_max(r, "r")
        # check_abs_max(w, "w")
        # check_abs_max(k, "k")
        # check_abs_max(v, "v")
        # check_abs_max(-kk, "-kk")
        # check_abs_max(kk * a, "kk*a")




        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)# * 2



        #print(f'layerid = {self.layer_id} Afterwkv x = {is_nan(x,"x")}')
        x = self.ln_x(x.view(B * T, C)).view(B, T, C) # Groupnorm is back!
        #x = x.view(B,T,-1)

        #print(f'layerid = {self.layer_id} Afterln x = {is_nan(x,"x")}')

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)

        #print(f'layerid = {self.layer_id} Afterwkv sum = {is_nan(x,"x")}')
        x = self.output(x)# * (torch.sigmoid(self.out_scale)*1e5) #hopefully absorb MLP scale initial=1.0

        #print(f'layerid = {self.layer_id} r = {is_nan(r,"r")} w = {is_nan(w,"w")} k = {is_nan(k,"k")} v = {is_nan(v,"v")} kk = {is_nan(kk,"kk")} x = {is_nan(x,"x")}')
        return x, v_first
    






class RWKV_Tmix_x070_Mose_cxa074(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size


        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads



        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.layer_architecture = nn.Parameter(torch.tensor(73.0))

        
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))


            #Change to GQA Style
            self.receptance = nn.Linear(C, self.num_attention_heads * self.head_size, bias=self.args.is_attention_bias)
            self.key = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.value = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.output = nn.Linear(self.num_attention_heads * self.head_size, C, bias=self.args.is_attention_output_bias)
            #SafeGroupNorm
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            #self.ln_x = SafeGroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            #self.out_scale = nn.Parameter(torch.tensor(-11.5129))

            #later copy from teacher weights
            #maybe can copy from any GQA Models
            self.receptance.weight.data.zero_()
            self.key.weight.data.zero_()
            self.value.weight.data.zero_()
            self.output.weight.data.zero_()
           

    
    #@torch.compile
    def forward(self, x, v_first,attention_mask):
        B, T, C = x.size()

        
        #removed tokenshift
        H = self.n_head
        r = self.receptance(x)
        w = -F.softplus(-(self.w0 + torch.tanh((x) @ self.w1) @ self.w2)) - 0.6 # soft-clamp to (-inf, -0.5) modified -0.5->-0.6
        k = self.key(x)
        v = self.value(x)


        k = k.view(B, T, self.num_key_value_heads, self.head_size)
        v = v.view(B, T, self.num_key_value_heads, self.head_size)

        # repeat k/v heads if n_kv_heads < n_heads
        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, self.num_key_value_groups)#reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, self.num_key_value_groups)#reshape(B,T,-1) #(B,T,C)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        #so now all B,T,C tensors

        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)


        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)
        #print(f'layerid = {self.layer_id} Afterwkv x = {is_nan(x,"x")}')

        x = self.ln_x(x.view(B * T, C)).view(B, T, C) # Groupnorm is back!
        #x = x.view(B,T,-1)



        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)

        #x = x * 2.0

        x = self.output(x)

        return x, v_first


class RWKV_Tmix_x070_Mose_cxa075(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size


        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads



        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.layer_architecture = nn.Parameter(torch.tensor(73.0))

        
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            #D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(torch.zeros(D_GATE_LORA, C))


            #Change to GQA Style
            self.receptance = nn.Linear(C, self.num_attention_heads * self.head_size, bias=self.args.is_attention_bias)
            self.key = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.value = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.output = nn.Linear(self.num_attention_heads * self.head_size, C, bias=self.args.is_attention_output_bias)
            #SafeGroupNorm
            #self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            #self.ln_x = SafeGroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            #self.out_scale = nn.Parameter(torch.tensor(-11.5129))

            #later copy from teacher weights
            #maybe can copy from any GQA Models
            self.receptance.weight.data.zero_()
            self.key.weight.data.zero_()
            self.value.weight.data.zero_()
            self.output.weight.data.zero_()
           

    
    #@torch.compile
    def forward(self, x, v_first,attention_mask):
        B, T, C = x.size()
        #removed tokenshift
        H = self.n_head
        r = self.receptance(x)
        w = -F.softplus(-(self.w0 + torch.tanh(x @ self.w1) @ self.w2)) -0.5
        k = self.key(x)
        v = self.value(x)


        k = k.view(B, T, self.num_key_value_heads, self.head_size)
        v = v.view(B, T, self.num_key_value_heads, self.head_size)

        # repeat k/v heads if n_kv_heads < n_heads
        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, self.num_key_value_groups)#reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, self.num_key_value_groups)#reshape(B,T,-1) #(B,T,C)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        #so now all B,T,C tensors

        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual

        g_delta = torch.sigmoid(x @ self.g1) @ self.g2
        g = 1.0 + g_delta

        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        # check_abs_max(r, "r")
        # check_abs_max(w, "w")
        # check_abs_max(k, "k")
        # check_abs_max(v, "v")
        # check_abs_max(-kk, "-kk")
        # check_abs_max(kk * a, "kk*a")
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)# * 2

        #x = self.ln_x(x.view(B * T, C)).view(B, T, C) # Groupnorm is back!
        x = x.view(B,T,-1)

        #print(f'layerid = {self.layer_id} Afterln x = {is_nan(x,"x")}')

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)

        #print(f'layerid = {self.layer_id} Afterwkv sum = {is_nan(x,"x")}')
        x = self.output(x*g)# * (torch.sigmoid(self.out_scale)*1e5) #hopefully absorb MLP scale initial=1.0

        #print(f'layerid = {self.layer_id} r = {is_nan(r,"r")} w = {is_nan(w,"w")} k = {is_nan(k,"k")} v = {is_nan(v,"v")} kk = {is_nan(kk,"kk")} x = {is_nan(x,"x")}')
        return x, v_first
    


class RWKV_Tmix_x070_Mose_cxa076(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size

        self.max_position_embeddings = args.config.max_position_embeddings
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.rms_norm_eps = args.rms_norm_eps

        self.RKNormMode = True

        print(f'layer = {layer_id} head_size {self.head_size} n_head {self.n_head}')



        assert args.dim_att % self.n_head == 0
        H = self.num_attention_heads#self.n_head
        N = self.head_size

        C = H*N#args.n_embd
        Hidden_dim = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.layer_architecture = nn.Parameter(torch.tensor(76.0))

        
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(Hidden_dim, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, self.num_attention_heads * self.head_size), 0.1))
            decay_speed = torch.ones(self.num_attention_heads * self.head_size)
            for n in range(self.num_attention_heads * self.head_size):
                decay_speed[n] = -7 + 5 * (n / ((self.num_attention_heads * self.head_size) - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,self.num_attention_heads * self.head_size) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(Hidden_dim, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, self.num_attention_heads * self.head_size), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,self.num_attention_heads * self.head_size))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(Hidden_dim, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, self.num_attention_heads * self.head_size), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,self.num_attention_heads * self.head_size)+1.0)

            #self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            #self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            #D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(Hidden_dim, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))


            #Change to GQA Style
            self.receptance = nn.Linear(Hidden_dim, self.num_attention_heads * self.head_size, bias=self.args.is_attention_bias)
            self.key = nn.Linear(Hidden_dim, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.value = nn.Linear(Hidden_dim, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.output = nn.Linear(self.num_attention_heads * self.head_size, Hidden_dim, bias=self.args.is_attention_output_bias)

            if self.RKNormMode == True:
                self.r_norm = Qwen3RMSNorm(self.head_size, eps=self.rms_norm_eps) 
                self.k_norm = Qwen3RMSNorm(self.head_size, eps=self.rms_norm_eps) 


            #later copy from teacher weights
            #maybe can copy from any GQA Models
            self.receptance.weight.data.zero_()
            self.key.weight.data.zero_()
            self.value.weight.data.zero_()
            self.output.weight.data.zero_()


     
           

    
    @torch.compile
    def forward(self, x, v_first,attention_mask,position_embeddings,position_ids):
        B, T, C = x.size()
        #removed tokenshift
        H = self.num_attention_heads#self.n_head


        if self.RKNormMode == True:
            r = self.r_norm(self.receptance(x).view(B,T,self.num_attention_heads,-1))
            k = self.k_norm(self.key(x).view(B,T,self.num_key_value_heads,-1))
        else:
            r = self.receptance(x)
            k = self.key(x)

        
        w = -F.softplus(-(self.w0 + torch.tanh(x @ self.w1) @ self.w2)) -0.5
        
        v = self.value(x)


        k = k.view(B, T, self.num_key_value_heads, self.head_size)
        v = v.view(B, T, self.num_key_value_heads, self.head_size)

        # repeat k/v heads if n_kv_heads < n_heads
        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        #so now all B,T,C tensors

        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual

        g = torch.sigmoid(x @ self.g1) @ self.g2

        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        #kk = k * self.k_k removed k_k
        kk = F.normalize(k.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,-1)
        #k = k * (1 + (a-1) * self.k_a)
        k = k * (1.0 - w + a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)
        x = x.view(B,T,-1)
        #Attention Scaling
        x = x * (self.head_size ** -0.5) 

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,-1)

        x = self.output(x*g)

        #print(f'layerid = {self.layer_id} r = {is_nan(r,"r")} w = {is_nan(w,"w")} k = {is_nan(k,"k")} v = {is_nan(v,"v")} kk = {is_nan(kk,"kk")} x = {is_nan(x,"x")}')
        return x, v_first
    
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
def generate_rotary_embedding(max_seqlen:int, dim:int, theta:float = 10000.0, scale:float = 1):
    #inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float).to(device) / dim))

    angular_velocity = theta ** -(torch.arange(0, dim, 2, dtype=torch.float) / dim) / scale # frequencies from 1.0 ... 1/theta
    angles = torch.outer(torch.arange(max_seqlen), angular_velocity)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((angles, angles), dim=-1)
    return torch.stack([emb.cos(), emb.sin()], dim=0)
    #return torch.polar(torch.ones_like(angles), angles)
# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def compute_qwen3_rope_cache(seq_len, rotary_dim, device, dtype, rope_theta):
            half_dim = rotary_dim // 2
            freq_seq = torch.arange(half_dim, dtype=dtype, device=device)
            inv_freq = 1.0 / (rope_theta ** (freq_seq / half_dim))
            positions = torch.arange(seq_len, dtype=dtype, device=device)
            freqs = torch.einsum("i,j->ij", positions, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos.unsqueeze(0), sin.unsqueeze(0), inv_freq

class RWKV_Tmix_x070_Mose_cxa078(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.Attention = 0
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size

        self.max_position_embeddings = args.config.max_position_embeddings
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.rms_norm_eps = args.rms_norm_eps
        self.rope_theta = args.config.rope_theta

        self.RKNormMode = True

        print(f'layer = {layer_id} head_size {self.head_size} n_head {self.n_head}')



        assert args.dim_att % self.n_head == 0
        H = self.num_attention_heads#self.n_head
        N = self.head_size

        C = H*N#args.n_embd
        Hidden_dim = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.layer_architecture = nn.Parameter(torch.tensor(77.0))

        
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32))# suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(Hidden_dim, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, self.num_attention_heads * self.head_size), 0.1))
            decay_speed = torch.ones(self.num_attention_heads * self.head_size)
            for n in range(self.num_attention_heads * self.head_size):
                decay_speed[n] = -7 + 5 * (n / ((self.num_attention_heads * self.head_size) - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,self.num_attention_heads * self.head_size) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(Hidden_dim, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, self.num_attention_heads * self.head_size), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,self.num_attention_heads * self.head_size))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(Hidden_dim, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, self.num_attention_heads * self.head_size), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,self.num_attention_heads * self.head_size)+1.0)

            #self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            #self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            #D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(Hidden_dim, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))


            #Change to GQA Style
            self.receptance = nn.Linear(Hidden_dim, self.num_attention_heads * self.head_size, bias=self.args.is_attention_bias)
            self.key = nn.Linear(Hidden_dim, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.value = nn.Linear(Hidden_dim, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.output = nn.Linear(self.num_attention_heads * self.head_size, Hidden_dim, bias=self.args.is_attention_output_bias)

            if self.RKNormMode == True:
                self.r_norm = Qwen3RMSNorm(self.head_size, eps=self.rms_norm_eps) 
                self.k_norm = Qwen3RMSNorm(self.head_size, eps=self.rms_norm_eps) 


            #later copy from teacher weights
            #maybe can copy from any GQA Models
            self.receptance.weight.data.zero_()
            self.key.weight.data.zero_()
            self.value.weight.data.zero_()
            self.output.weight.data.zero_()


     
           

    
    @torch.compile
    def forward(self, x, v_first,attention_mask,position_embeddings,position_ids):
        B, T, C = x.size()
        #removed tokenshift
        H = self.num_attention_heads#self.n_head


        if self.RKNormMode == True:
            r = self.r_norm(self.receptance(x).view(B,T,self.num_attention_heads,-1))
            k = self.k_norm(self.key(x).view(B,T,self.num_key_value_heads,-1))
        else:
            r = self.receptance(x)
            k = self.key(x)

        
        w = -F.softplus(-(self.w0 + torch.tanh(x @ self.w1) @ self.w2)) -0.5
        
        v = self.value(x)


        k = k.view(B, T, self.num_key_value_heads, self.head_size)
        v = v.view(B, T, self.num_key_value_heads, self.head_size)

        #cos, sin = position_embeddings
        #disable hf's pos calc

        cos, sin, inv_freq_own = compute_qwen3_rope_cache(T, self.head_size, k.device, torch.float32, self.rope_theta)

        cos=cos.to(dtype=torch.bfloat16)
        sin=sin.to(dtype=torch.bfloat16)

        r, k = apply_rotary_pos_emb(r, k, cos, sin, unsqueeze_dim=2)

        # repeat k/v heads if n_kv_heads < n_heads
        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        #so now all B,T,C tensors

        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual

        g = torch.sigmoid(x @ self.g1) @ self.g2
        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        kk = F.normalize(k.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,-1)
        k = k * (1.0 - w + a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)
        x = x.view(B,T,-1)
        x = x * (self.head_size ** -0.5) 

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,-1)

        x = self.output(x*g)

        #print(f'layerid = {self.layer_id} r = {is_nan(r,"r")} w = {is_nan(w,"w")} k = {is_nan(k,"k")} v = {is_nan(v,"v")} kk = {is_nan(kk,"kk")} x = {is_nan(x,"x")}')
        return x, v_first
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
#     # x: [B, H, T, D]
#     x1, x2 = x[..., ::2], x[..., 1::2]  # 偶数・奇数次元に分解
#     x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
#     return x_rot.flatten(-2)

# def build_rope_cache(seq_len: int, dim: int, device, dtype, theta: float = 10000.0):
#     half_dim = dim // 2
#     freq_seq = torch.arange(half_dim, device=device, dtype=dtype)
#     inv_freq = 1.0 / (theta ** (freq_seq / half_dim))
#     t = torch.arange(seq_len, device=device, dtype=dtype)
#     freqs = torch.einsum("i,j->ij", t, inv_freq)
#     emb = torch.cat([freqs, freqs], dim=-1)
#     return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]  # [1, 1, T, D]
from typing import Callable, Optional, Tuple, Union
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv_original(key, module.num_key_value_groups)
    value_states = repeat_kv_original(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
    #     logger.warning_once(
    #         "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
    #         " Please set your attention to `eager` if you want any of these features."
    #     )

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv_original(key, module.num_key_value_groups)
        value = repeat_kv_original(value, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
class GQAWithRopeAttention(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        # self.embed_dim = embed_dim
        # self.num_heads = num_heads
        # self.kv_heads = kv_heads
        # self.head_dim = embed_dim // num_heads
        self.training = True
        
        # self.rope_theta = rope_theta
        self.args = args
        self.layer_id = layer_id
        self.Attention = 1

        self.head_size = args.head_size_a
        self.scaling = self.head_size ** -0.5
        self.n_head = args.dim_att // self.head_size

        self.max_position_embeddings = args.config.max_position_embeddings
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.rms_norm_eps = args.rms_norm_eps
        self.rope_theta = args.config.rope_theta

        self.QKNormMode = True

        print(f'layer = {layer_id} head_size {self.head_size} n_head {self.n_head}')



        assert args.dim_att % self.n_head == 0
        H = self.num_attention_heads#self.n_head
        N = self.head_size

        C = H*N#args.n_embd
        Hidden_dim = args.n_embd

        # assert embed_dim % num_heads == 0
        # assert num_heads % kv_heads == 0

        # self.q_proj = nn.Linear(embed_dim, embed_dim)
        # self.k_proj = nn.Linear(embed_dim, kv_heads * self.head_dim)
        # self.v_proj = nn.Linear(embed_dim, kv_heads * self.head_dim)
        # self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.q_proj = nn.Linear(Hidden_dim, self.num_attention_heads * self.head_size, bias=self.args.is_attention_bias)
        self.k_proj = nn.Linear(Hidden_dim, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
        self.v_proj = nn.Linear(Hidden_dim, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_size, Hidden_dim, bias=self.args.is_attention_output_bias)

        if self.QKNormMode == True:
            self.q_norm = Qwen3RMSNorm(self.head_size, eps=self.rms_norm_eps) 
            self.k_norm = Qwen3RMSNorm(self.head_size, eps=self.rms_norm_eps) 

    # def forward_(self, x, position_embeddings,past_kv: dict = None, use_cache=False):
    #     # x: [B, T, C]
    #     B, T, C = x.size()
    #     H, H_kv, D = self.num_attention_heads, self.num_key_value_heads, self.head_size
    #     G = H // H_kv  # group数

    #     if self.QKNormMode:
    #         q = self.q_norm(self.q_proj(x).view(B, T, H, D)).transpose(1, 2)  # [B, H, T, D]
    #         k = self.k_norm(self.k_proj(x).view(B, T, H_kv, D)).transpose(1, 2)  # [B, H_kv, T, D]
    #     else:
    #         q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
    #         k = self.k_proj(x).view(B, T, H_kv, D).transpose(1, 2)  # [B, H_kv, T, D]
    #     v = self.v_proj(x).view(B, T, H_kv, D).transpose(1, 2)  # [B, H_kv, T, D]

    #     # RoPEの適用
    #     #cos, sin = build_rope_cache(T, D, x.device, x.dtype, self.rope_theta)
        
    #     #cos, sin, inv_freq_own = compute_qwen3_rope_cache(T, self.head_size, k.device, torch.float32, self.rope_theta)
    #     cos ,sin = position_embeddings

    #     #cos=cos.to(dtype=torch.bfloat16)
    #     #sin=sin.to(dtype=torch.bfloat16)

    #     # q = apply_rope(q, cos, sin)
    #     # k = apply_rope(k, cos, sin)
    #     q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    #     # 過去のKVを追加
    #     if past_kv is not None:
    #         k = torch.cat([past_kv["k"], k], dim=2)  # 時間軸に結合
    #         v = torch.cat([past_kv["v"], v], dim=2)

    #     # KVキャッシュ保存
    #     if use_cache:
    #         new_kv = {"k": k.detach(), "v": v.detach()}
    #     else:
    #         new_kv = None

    #     # GQA: [B, H, T, D] vs [B, H_kv, S, D] → repeat KV
    #     k = k.unsqueeze(1).repeat(1, G, 1, 1, 1).view(B, H, -1, D)
    #     v = v.unsqueeze(1).repeat(1, G, 1, 1, 1).view(B, H, -1, D)

    #     # SDPA (B, H, T, D), (B, H, S, D)
    #     out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # causal処理
    #     out = out.transpose(1, 2).contiguous().view(B, T, C)
    #     return self.o_proj(out)#, new_kv
    
    @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
       # past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_size)
        B, T, C = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        #cos, sin = position_embeddings
        cos, sin, inv_freq_own = compute_qwen3_rope_cache(T, self.head_size, key_states.device, torch.float32, self.rope_theta)
        cos=cos.to(dtype=torch.bfloat16)
        sin=sin.to(dtype=torch.bfloat16)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = sdpa_attention_forward#eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
        #         logger.warning_once(
        #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
        #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        #         )
        #     else:
        #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,# if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=False,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output#, attn_weights

    
