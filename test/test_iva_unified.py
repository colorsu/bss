"""
IVA 算法统一测试脚本

通过修改 test_config.py 中的 ACTIVE_IVA 来切换不同的IVA实现:
- IVA_NG: Natural Gradient IVA
- AUX_IVA: Auxiliary Function IVA

调整参数也在 test_config.py 中进行
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path
from src.audio import STFT, load_audio_sf, save_audio_sf
from src.bss import IVA_NG, AUX_IVA_ISS
from test_config import *


class IVATestRunner(torch.nn.Module):
    """IVA算法测试运行器"""

    def __init__(self, algorithm: str, config: dict):
        super().__init__()
        self.algorithm = algorithm
        self.config = config
        self.frame_shift = config["frame_shift"]
        self.stft = STFT(win_len=self.frame_shift * 2, shift_len=self.frame_shift)
        
        # 根据算法类型初始化分离器
        if algorithm == "IVA_NG":
            self.separator = IVA_NG(
                n_components=config["n_components"],
                learning_rate=config["learning_rate"],
                n_iter=config["n_iter"],
                ref_mic=config.get("ref_mic", 1)
            )
        elif algorithm == "AUX_IVA":
            self.separator = AUX_IVA_ISS(
                n_iter=config["n_iter"],
                contrast_func=config["contrast_func"],
                ref_mic=config.get("ref_mic", 1)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        print(f"初始化 {algorithm}")
        for key, value in config.items():
            print(f"  {key}: {value}")

    def forward(self, mix):
        mix_spec = self.stft.transform(mix)
        print(f"频谱图形状: {mix_spec.shape}")
        
        sep_spec = self.separator(mix_spec)
        output = self.stft.inverse(sep_spec)
        
        return output


def main():
    # 打印当前配置
    print_config("IVA")
    
    # 加载音频
    mix_fname = AUDIO_FILE
    mix, sr = load_audio_sf(mix_fname, n_channels=N_CHANNELS)
    print(f"\n加载音频: {mix_fname}")
    print(f"时长: {mix.shape[1] / sr:.2f}秒, 采样率: {sr}Hz")
    
    # 获取配置并创建模型
    algorithm, config = get_iva_config()
    model = IVATestRunner(algorithm, config)
    
    # 运行分离
    print(f"\n开始分离...")
    with torch.no_grad():
        output = model(mix)
    
    # 保存输出
    if SAVE_OUTPUT:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        fname = Path(mix_fname).stem
        output_path = output_dir / f"{fname}_{algorithm}.wav"
        save_audio_sf(str(output_path), output, sr)
        print(f"\n✓ 输出已保存到: {output_path}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
