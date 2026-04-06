import argparse
import json
import glob
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# 确保您已经安装了必要的库: pip install transformers torch sentencepiece
try:
    from transformers import AutoTokenizer
except ImportError:
    print("错误：'transformers'库未安装。请运行 'pip install transformers torch sentencepiece'")
    exit()


def _analyze_single_directory(directory_path: Path, tokenizer) -> Optional[Dict[str, Any]]:
    """
    (辅助函数) 分析单个目录，计算其中'think'字段的token统计信息。
    """
    print(f"\n--- 正在处理子目录: {directory_path.name} ---")
    
    search_pattern = str(directory_path / "output_*.json")
    file_paths = glob.glob(search_pattern)

    if not file_paths:
        print(f"在目录 '{directory_path.name}' 中没有找到 'output_*.json' 文件，已跳过。")
        return None

    all_think_token_counts: List[int] = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    continue
                for item in data:
                    if isinstance(item, dict) and 'think' in item:
                        think_text = item.get('think', '')
                        if think_text and isinstance(think_text, str):
                            token_ids = tokenizer.encode(think_text, add_special_tokens=False)
                            all_think_token_counts.append(len(token_ids))
        except Exception as e:
            print(f"处理文件 '{file_path}' 时发生错误: {e}")

    if not all_think_token_counts:
        print(f"在目录 '{directory_path.name}' 的文件中没有找到有效的'think'内容。")
        return None

    print(f"在 '{directory_path.name}' 中成功处理了 {len(all_think_token_counts)} 个'think'块。")
    
    counts_array = np.array(all_think_token_counts)
    stats: Dict[str, Any] = {
        "total_think_blocks": len(all_think_token_counts),
        "mean_token_count": float(np.mean(counts_array)),
        "variance": float(np.var(counts_array)),
        "std_deviation": float(np.std(counts_array)),
        "median_token_count": float(np.median(counts_array)),
        "min_token_count": int(np.min(counts_array)),
        "max_token_count": int(np.max(counts_array)),
    }
    return stats


def print_res(output_dir):
    with open(os.path.join(output_dir, "eval.json"), "r") as f:
        iou_res = json.load(f)
    with open(os.path.join(output_dir, "token_stats.json"), "r") as f:
        all_results = json.load(f)
    pad_num = 20
    print(" "*pad_num, "Token", "\tgIoU", "\tcIoU")
    for k, v in all_results.items():
        print(f"{k:<{pad_num}} {v['mean_token_count']:.2f} \t{iou_res[k]['gIoU']*100:.2f} \t{iou_res[k]['cIoU']*100:.2f}")

def analyze_all_subdirectories(top_level_dir: str, force: bool, output_filename: str = "token_stats.json"):
    """
    遍历主目录下的所有子目录，对每个子目录执行token分析，并将结果聚合。
    """
    top_level_path = Path(top_level_dir).resolve()
    output_dir = top_level_path.parent
    output_path = output_dir / output_filename
    if os.path.exists(output_path) and not force:
        print_res(output_dir)
        return
    
    # --- 1. 加载分词器 (只加载一次) ---
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"分词器 '{model_path}' 加载成功。")
    except Exception as e:
        print(f"错误：加载分词器失败: {e}")
        return

    # --- 2. 遍历所有子目录并进行分析 ---
    subdirectories = [d for d in top_level_path.iterdir() if d.is_dir()]

    if not subdirectories:
        print(f"错误：在主目录 '{top_level_dir}' 中没有找到任何子目录。")
        return
        
    all_results: Dict[str, Any] = {}
    for subdir in subdirectories:
        # 调用辅助函数处理每个子目录
        stats = _analyze_single_directory(subdir, tokenizer)
        if stats:
            all_results[subdir.name] = stats
            
    if not all_results:
        print("处理完成，但没有在任何子目录中收集到有效数据。")
        return

    # --- 3. 保存聚合后的结果 ---
    try:
        existing_data = {}
        if output_path.exists():
            print(f"\n发现已有的结果文件: {output_path}，将进行更新。")
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"警告：已有的结果文件 '{output_path}' 格式损坏，将被覆盖。")
        
        existing_data.update(all_results)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)
            
        print(f"\n所有统计结果已成功聚合保存到: {output_path}")
        print("--- 本次运行聚合的结果预览 ---")
        print(json.dumps(all_results, indent=4))

    except Exception as e:
        print(f"错误：无法保存最终结果文件。 {e}")

    print_res(output_dir)

def find_infer_json_dirs(base_directory: str) -> list[str]:
    """
    遍历指定目录下所有子目录，查找并返回所有以 'infer_json' 结尾的目录的完整路径。

    Args:
        base_directory: 要开始搜索的根目录路径。

    Returns:
        一个包含所有符合条件的 'infer_json' 目录的完整路径的列表。
        如果目录不存在，或者没有找到，则返回空列表。
    """
    if not os.path.isdir(base_directory):
        print(f"错误：'{base_directory}' 不是一个有效的目录。")
        return []

    found_infer_json_paths = []

    # os.walk() 会生成一个三元组 (dirpath, dirnames, filenames)
    # dirpath 是当前正在遍历的目录的路径
    # dirnames 是当前目录下的所有子目录名列表
    # filenames 是当前目录下的所有文件名列表
    for root, dirs, files in os.walk(base_directory):
        # 检查当前目录的名称是否是 'infer_json'
        # os.path.basename(root) 返回路径的最后一个组成部分
        if os.path.basename(root) == 'infer_json':
            found_infer_json_paths.append(root)
            # 注意：这里有一个重要的优化。
            # 如果当前目录是 'infer_json'，那么它的任何子目录都不会再是
            # 符合我们条件的“以 infer_json 为最后一层”的目录。
            # 因此，我们可以通过清空 dirs 列表来告诉 os.walk 不要再深入遍历这个 infer_json 目录的子目录。
            # 但如果需求是 infer_json/sub_infer_json 这种结构也要找，那就不能清空 dirs。
            # 根据题目“以infer_json为最后一层目录”，这意味着 infer_json 下的子目录不符合条件，所以这里清空 dirs 是合适的。
            dirs[:] = [] # 清空 dirs 列表，阻止 os.walk 进一步遍历当前目录的子目录。

    return found_infer_json_paths


def main():
    parser = argparse.ArgumentParser(
        description="遍历主目录下的所有子目录，对每个子目录中的output_*.json文件进行'think'字段的token统计，并聚合结果。"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="包含多个实验子目录的主目录路径。"
    )
    parser.add_argument(
        "-f",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"错误：提供的路径 '{args.directory}' 不是一个有效的目录。")
        return
    
    if "infer_json" in args.directory:
        analyze_all_subdirectories(args.directory, args.f)
    else:
        dirs = find_infer_json_dirs(args.directory)
        for d in dirs:
            print("\n", d)
            analyze_all_subdirectories(d, args.f)

if __name__ == "__main__":
    main()