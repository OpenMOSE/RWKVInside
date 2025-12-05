import torch
import torch.nn.functional as F

def check_flashattention_support():
    """PyTorch SDPAのFlashAttentionサポートを確認する"""
    
    print("="*60)
    print("PyTorch SDPA FlashAttention サポート確認")
    print("="*60)
    
    # PyTorchバージョン確認
    print(f"\nPyTorchバージョン: {torch.__version__}")
    
    # CUDAの利用可能性確認
    cuda_available = torch.cuda.is_available()
    print(f"CUDA利用可能: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA バージョン: {torch.version.cuda}")
        print(f"cuDNN バージョン: {torch.backends.cudnn.version()}")
        
        # GPUデバイス情報
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            capability = torch.cuda.get_device_capability(i)
            print(f"  Compute Capability: {capability[0]}.{capability[1]}")
    
    print("\n" + "-"*60)
    print("SDPA バックエンド確認")
    print("-"*60)
    
    # 利用可能なSDPAバックエンドを確認
    backends = {
        "flash": "FlashAttention",
        "mem_efficient": "Memory Efficient Attention",
        "math": "Math (PyTorch native)",
        "cudnn": "cuDNN"
    }
    
    # PyTorch 2.0以降のバックエンド確認方法
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        print("\n利用可能なSDPAバックエンド:")
        
        # 各バックエンドの状態を確認
        sdp_kernel = torch.backends.cuda.sdp_kernel
        
        # コンテキストマネージャを使って各バックエンドをテスト
        for backend_name, display_name in backends.items():
            try:
                # バックエンド属性の存在確認
                if hasattr(sdp_kernel, f"enable_{backend_name}"):
                    print(f"  ✓ {display_name}: サポートあり")
                else:
                    print(f"  ✗ {display_name}: 属性なし")
            except Exception as e:
                print(f"  ? {display_name}: 確認エラー - {e}")
    else:
        print("SDPAバックエンド情報が利用できません（PyTorch 2.0以降が必要）")
    
    print("\n" + "-"*60)
    print("実際のテスト")
    print("-"*60)
    
    if cuda_available and hasattr(torch.backends.cuda, "sdp_kernel"):
        try:
            # テスト用のテンソルを作成
            batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
            device = torch.device("cuda")
            dtype = torch.float16  # FlashAttentionはfloat16/bfloat16で動作
            
            # クエリ、キー、バリューを作成
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                          device=device, dtype=dtype)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                          device=device, dtype=dtype)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                          device=device, dtype=dtype)
            
            # FlashAttentionを明示的に有効化してテスト
            print("\nFlashAttentionバックエンドでのテスト:")
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False
                ):
                    output = F.scaled_dot_product_attention(q, k, v)
                    print(f"  ✓ FlashAttentionで実行成功")
                    print(f"    出力shape: {output.shape}")
            except Exception as e:
                print(f"  ✗ FlashAttention実行失敗: {e}")
            
            # Memory Efficientバックエンドでのテスト
            print("\nMemory Efficientバックエンドでのテスト:")
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=False,
                    enable_mem_efficient=True
                ):
                    output = F.scaled_dot_product_attention(q, k, v)
                    print(f"  ✓ Memory Efficientで実行成功")
                    print(f"    出力shape: {output.shape}")
            except Exception as e:
                print(f"  ✗ Memory Efficient実行失敗: {e}")
            
            # デフォルトバックエンドの確認
            print("\nデフォルトバックエンド（自動選択）でのテスト:")
            output = F.scaled_dot_product_attention(q, k, v)
            print(f"  ✓ 実行成功")
            print(f"    出力shape: {output.shape}")
            
        except Exception as e:
            print(f"テスト実行エラー: {e}")
    else:
        print("CUDAが利用できないか、PyTorchバージョンが古いためテストをスキップ")
    
    print("\n" + "="*60)
    print("FlashAttentionの要件:")
    print("="*60)
    print("""
    1. PyTorch 2.0以降
    2. CUDA 11.6以降
    3. GPU Compute Capability 7.5以降（Turing以降）
       - RTX 20xx, RTX 30xx, RTX 40xx
       - A100, H100, etc.
    4. データ型: float16またはbfloat16
    5. アテンションマスク: causalまたはなし
    6. ドロップアウト: サポート（確率値のみ）
    """)

if __name__ == "__main__":
    check_flashattention_support()