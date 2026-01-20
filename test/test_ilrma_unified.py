"""
ILRMA 算法统一测试脚本

通过修改 test_config.py 中的 ACTIVE_ILRMA 来切换不同的ILRMA实现:
- ILRMA: 标准实现
- ILRMA_V2: 改进版本
- ILRMA_SR: 带steering vector的版本

调整参数也在 test_config.py 中进行
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
from src.audio import STFT, load_audio_sf, save_audio_sf
from src.bss import ILRMA, ILRMA_V2, ILRMA_SR, ILRMA_REALTIME
from test_config import *


class ILRMATestRunner(torch.nn.Module):
    """ILRMA算法测试运行器"""

    def __init__(self, algorithm: str, config: dict):
        super().__init__()
        self.algorithm = algorithm
        self.config = config
        self.frame_shift = config["frame_shift"]
        self.stft = STFT(win_len=self.frame_shift * 2, shift_len=self.frame_shift)
        # 根据算法类型初始化分离器
        if algorithm == "ILRMA":
            self.separator = ILRMA(
                n_components=config["n_components"],
                k_NMF_bases=config["k_NMF_bases"],
                n_iter=config["n_iter"]
            )
        elif algorithm == "ILRMA_V2":
            self.separator = ILRMA_V2(
                n_components=config["n_components"],
                k_NMF_bases=config["k_NMF_bases"],
                n_iter=config["n_iter"]
            )
        elif algorithm == "ILRMA_SR":
            self.separator = ILRMA_SR(
                n_components=config["n_components"],
                k_NMF_bases=config["k_NMF_bases"],
                n_iter=config["n_iter"]
            )
        elif algorithm == "ILRMA_REALTIME":
            self.separator = ILRMA_REALTIME(
                n_components=config["n_components"],
                k_NMF_bases=config["k_NMF_bases"],
                n_iter=config["n_iter"],
                observation_window_sec=config.get("observation_window_sec", 5.0),
                update_interval_frames=config.get("update_interval_frames", 16),
                hop_length=config["frame_shift"],
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        print(f"初始化 {algorithm}")
        for key, value in config.items():
            print(f"  {key}: {value}")

    def forward(self, mix, steering_vector=None):
        mix_spec = self.stft.transform(mix)
        print(f"频谱图形状: {mix_spec.shape}")
        
        if self.algorithm == "ILRMA_SR" and steering_vector is not None:
            sep_spec = self.separator(mix_spec, steering_vector)
        else:
            sep_spec = self.separator(mix_spec)
        
        output = self.stft.inverse(sep_spec)
        return output


def main():
    # 打印当前配置
    print_config("ILRMA")
    
    # 加载音频
    mix_fname = AUDIO_FILE
    mix, sr = load_audio_sf(mix_fname, n_channels=N_CHANNELS)
    print(f"\n加载音频: {mix_fname}")
    print(f"时长: {mix.shape[1] / sr:.2f}秒, 采样率: {sr}Hz")
    
    # 获取配置并创建模型
    algorithm, config = get_ilrma_config()
    model = ILRMATestRunner(algorithm, config)
    
    # 对于ILRMA_SR，可以在这里生成steering vector
    steering_vector = None
    if algorithm == "ILRMA_SR":
        # TODO: 如果需要测试ILRMA_SR，在这里生成steering vector
        # from test_ilrma_sr import generate_mixture, calculate_steering_vector
        # steering_vector = calculate_steering_vector(...)
        print("\n注意: ILRMA_SR需要steering vector，当前使用None（将使用默认行为）")
    
    # 运行分离
    print(f"\n开始分离...")
    with torch.no_grad():
        output = model(mix, steering_vector)
    
    # 保存输出
    if SAVE_OUTPUT:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        fname = Path(mix_fname).stem
        output_path = output_dir / f"{fname}_{algorithm}_{OUTPUT_POSTFIX}.wav"
        save_audio_sf(str(output_path), output, sr)
        print(f"\n✓ 输出已保存到: {output_path}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
