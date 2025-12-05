import os
import math
import torch
import safetensors
from safetensors.torch import save_file
import sys
import gc
import argparse

def get_tensor_size_in_bytes(tensor):
    """テンソルのメモリサイズをバイト単位で計算"""
    return tensor.element_size() * tensor.nelement()

def split_and_save_checkpoint(input_path, output_prefix, max_size_gb=10):
    """
    Pytorchチェックポイントを読み込み、指定されたサイズごとに分割してSafeTensorsとして保存
    
    Args:
        input_path: 入力のPytorchチェックポイントファイルのパス (.pth or .pt)
        output_prefix: 出力ファイルの接頭辞 
        max_size_gb: 各分割ファイルの最大サイズ (GB単位)
    """
    print(f"チェックポイント {input_path} を読み込み中...")
    
    # チェックポイントの読み込み
    try:
        checkpoint = torch.load(input_path, map_location="cpu")
    except Exception as e:
        print(f"チェックポイントの読み込みに失敗しました: {e}")
        return
    
    # state_dictの取得（モデル自体またはstate_dictを想定）
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        print("サポートされていないチェックポイント形式です。state_dictを含むディクショナリが必要です。")
        return
    
    # 最大ファイルサイズをバイト単位に変換
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024
    
    # テンソルをグループ化して分割ファイルに割り当てる
    part_index = 0
    current_part = {}
    current_size = 0
    saved_parts = []
    
    print(f"チェックポイントを{max_size_gb}GBごとに分割しています...")
    
    # キーを並べ替えて、より効率的に詰め込むことができる可能性を高める
    keys = sorted(state_dict.keys(), key=lambda k: get_tensor_size_in_bytes(state_dict[k]), reverse=True)
    
    for key in keys:
        tensor = state_dict[key]
        tensor_size = get_tensor_size_in_bytes(tensor)
        
        # 単一のテンソルが最大サイズより大きい場合の警告
        if tensor_size > max_size_bytes:
            print(f"警告: テンソル '{key}' は {tensor_size / (1024**3):.2f}GB で、最大サイズの{max_size_gb}GBを超えています。")
            # それでも追加する（このテンソルは単独でファイルになる）
            save_file({key: tensor}, f"{output_prefix}_part{part_index:03d}.safetensors")
            saved_parts.append(f"{output_prefix}_part{part_index:03d}.safetensors")
            part_index += 1
            continue
            
        # 現在のパートに追加するとサイズオーバーする場合、現在のパートを保存
        if current_size + tensor_size > max_size_bytes and current_part:
            save_file(current_part, f"{output_prefix}_part{part_index:03d}.safetensors")
            saved_parts.append(f"{output_prefix}_part{part_index:03d}.safetensors")
            part_index += 1
            current_part = {}
            current_size = 0
            
            # メモリ解放
            gc.collect()
        
        # テンソルを現在のパートに追加
        current_part[key] = tensor
        current_size += tensor_size
    
    # 最後のパートが残っていれば保存
    if current_part:
        save_file(current_part, f"{output_prefix}_part{part_index:03d}.safetensors")
        saved_parts.append(f"{output_prefix}_part{part_index:03d}.safetensors")
    
    # メタデータの保存（どのキーがどのファイルにあるかの情報）
    metadata = {}
    for file_idx, file_path in enumerate(saved_parts):
        # チェックポイント内の各パートをロード
        part_data = safetensors.torch.load_file(file_path)
        for key in part_data.keys():
            metadata[key] = file_idx
    
    # メタデータを保存
    torch.save(metadata, f"{output_prefix}_metadata.pt")
    
    print(f"分割完了！ {len(saved_parts)}個のファイルに保存されました。")
    for i, part in enumerate(saved_parts):
        part_size = os.path.getsize(part) / (1024**3)
        print(f"  - {part}: {part_size:.2f}GB")
    
    print(f"メタデータは {output_prefix}_metadata.pt に保存されました。")

def load_split_checkpoint(metadata_path, target_model=None):
    """
    分割されたチェックポイントを読み込み、辞書として返すか、モデルに直接ロード
    
    Args:
        metadata_path: メタデータファイルのパス
        target_model: (オプション) 読み込み先のモデル
    
    Returns:
        結合されたstate_dictまたはNone（モデルに直接ロードした場合）
    """
    print(f"メタデータ {metadata_path} を読み込み中...")
    
    # メタデータの読み込み
    metadata = torch.load(metadata_path)
    
    # パスの取得
    base_prefix = os.path.splitext(metadata_path)[0].replace('_metadata', '')
    
    # 全てのキーを保持する辞書
    complete_state_dict = {}
    
    # 最大ファイルインデックスを見つける
    max_idx = max(metadata.values())
    
    # 各パートを順番に読み込み
    for idx in range(max_idx + 1):
        file_path = f"{base_prefix}_part{idx:03d}.safetensors"
        print(f"ファイル {file_path} を読み込み中...")
        
        part_state_dict = safetensors.torch.load_file(file_path)
        complete_state_dict.update(part_state_dict)
        
        # メモリを節約するためにパートを解放
        del part_state_dict
        gc.collect()
    
    # モデルが提供された場合は直接ロード
    if target_model is not None:
        target_model.load_state_dict(complete_state_dict)
        return None
    
    return complete_state_dict

# 使用例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorchのチェックポイントを分割または読み込む')
    subparsers = parser.add_subparsers(dest='command', help='コマンド')
    
    # 分割コマンドの引数
    split_parser = subparsers.add_parser('split', help='チェックポイントを分割')
    split_parser.add_argument('input_path', type=str, help='入力ファイルのパス (.pth or .pt)')
    split_parser.add_argument('output_prefix', type=str, help='出力ファイルの接頭辞')
    split_parser.add_argument('--max_size_gb', type=float, default=10.0, help='各分割ファイルの最大サイズ (GB単位)')
    
    # 読み込みコマンドの引数
    load_parser = subparsers.add_parser('load', help='分割されたチェックポイントを読み込む')
    load_parser.add_argument('metadata_path', type=str, help='メタデータファイルのパス')
    
    args = parser.parse_args()
    
    if args.command == 'split':
        split_and_save_checkpoint(args.input_path, args.output_prefix, args.max_size_gb)
    
    elif args.command == 'load':
        state_dict = load_split_checkpoint(args.metadata_path)
        print(f"チェックポイントを読み込みました。{len(state_dict)}個のキーが含まれています。")
    
    else:
        parser.print_help()



#python convertsafetensor.py split /workspace/output/mistral3small/PRWKV-7-Mistral-Small-Instruct-Preview-v0.1.pth /workspace/output/mistral3small/safetensor/model --max_size_gb 10