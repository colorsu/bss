"""
测试配置文件 - 统一管理所有BSS算法的测试参数

使用方法:
1. 修改 ACTIVE_CONFIG 来选择要测试的算法
2. 修改 AUDIO_FILE 来选择输入音频
3. 在各算法的配置字典中调整参数
4. 运行 test_iva.py 或 test_ilrma.py
"""

# ==================== 音频文件配置 ====================
# AUDIO_FILE = "/Users/kolor/myWork/data/JL_FB_ANF.wav"
# AUDIO_FILE = "/Users/kolor/myWork/data/星巴克-0626.wav"
AUDIO_FILE = "/Users/kolor/myWork/data/train_3p1_1209.wav"
# AUDIO_FILE = "/Users/kolor/myWork/data/train_3p1_short.wav"
# AUDIO_FILE = "/Users/kolor/myWork/bss/generated_mix.wav"
# AUDIO_FILE = "../noise_test_2ch.wav"
OUTPUT_POSTFIX = '1A'

N_CHANNELS = 3  # 输入音频的声道数


# ==================== IVA 算法配置 ====================
# 选择要测试的IVA算法: "IVA_NG", "AUX_IVA", "AUX_IVA_ONLINE", "AUX_OVER_IVA", "AUX_OVER_IVA_ONLINE"
ACTIVE_IVA = "AUX_IVA"

IVA_NG_CONFIG = {
    "n_iter": 200,
    "learning_rate": 0.1,
    "frame_shift": 256,
    "n_components": 2,
    "ref_mic": 1,
}

AUX_IVA_CONFIG = {
    "n_iter": 100,
    "frame_shift": 256,
    "contrast_func": "laplace",  # "gaussian", "laplace"
    "ref_mic": 1,
}

AUX_IVA_ONLINE_CONFIG = {
    # Online algorithm updates per time frame
    "n_iter": 1,  # kept for compatibility/printing; not used directly
    "n_iter_per_frame": 1,
    "alpha": 0.98,
    "frame_shift": 256,
    "contrast_func": "gaussian",  # Options: "laplace", "gaussian", "logcosh", "exp", "pow1.5", "pow0.5", "power"
    "gamma": 0.002,  # Exponent for "power" contrast function: g(r) = r^gamma
    "proj_back_type": "mdp",  # Options: "mdp" (Minimal Distortion), "scale_constraint", "none"
    "ref_mic": 1,
}

AUX_OVER_IVA_CONFIG = {
    "num_targets": 1,  # Number of target sources K (must be < num_mics M)
    "n_iter": 50,  # Total IP iterations
    "frame_shift": 512,
    "contrast_func": "power",
    "gamma": 0.05,
    "proj_back_type": "scale_constraint",
    "ref_mic": 1,
    "tol": 1e-8,
}

AUX_OVER_IVA_ONLINE_CONFIG = {
    "num_targets": 1,  # Number of target sources K (must be < num_mics M)
    "n_iter": 1,  # ISS iterations per frame
    "n_iter_per_frame": 1,
    "alpha": 0.99,
    "frame_shift": 512,
    "contrast_func": "power",
    "gamma": 0.02,
    "proj_back_type": "mdp",
    "ref_mic": 1,
}


# ==================== ILRMA 算法配置 ====================
# 选择要测试的ILRMA算法: "ILRMA", "ILRMA_V2", "ILRMA_SR", 或 "ILRMA_REALTIME"
ACTIVE_ILRMA = "ILRMA_REALTIME"

ILRMA_CONFIG = {
    "n_iter": 100,
    "frame_shift": 512,
    "n_components": N_CHANNELS,
    "k_NMF_bases": 8,
}

ILRMA_V2_CONFIG = {
    "n_iter": 100,
    "frame_shift": 256*5,
    "n_components": N_CHANNELS,
    "k_NMF_bases": 16,
}

ILRMA_SR_CONFIG = {
    "n_iter": 100,
    "frame_shift": 256,
    "n_components": 2,
    "k_NMF_bases": 8,
    # 对于ILRMA_SR，需要在代码中生成steering vector
}

ILRMA_REALTIME_CONFIG = {
    "n_iter": 1,  # 后台更新时的迭代次数
    "frame_shift": 512,  # 实时帧移
    "n_components": N_CHANNELS,
    "k_NMF_bases": 1,
    "observation_window_sec": 5.0,  # 观测窗长（秒）
    "update_interval_frames": 8,  # 更新间隔（帧数）
}


# ==================== RCSCME 算法配置 ====================
RCSCME_CONFIG = {
    "n_iter": 50,
    "frame_shift": 512,
    "n_components": 2,
    "k_NMF_bases": 8,
}


# ==================== 输出配置 ====================
OUTPUT_DIR = "test_outputs"  # 输出目录
SAVE_OUTPUT = True  # 是否保存输出音频


# ==================== 辅助函数 ====================
def get_iva_config():
    """获取当前激活的IVA配置"""
    if ACTIVE_IVA == "IVA_NG":
        return ACTIVE_IVA, IVA_NG_CONFIG
    elif ACTIVE_IVA == "AUX_IVA":
        return ACTIVE_IVA, AUX_IVA_CONFIG
    elif ACTIVE_IVA == "AUX_IVA_ONLINE":
        return ACTIVE_IVA, AUX_IVA_ONLINE_CONFIG
    elif ACTIVE_IVA == "AUX_OVER_IVA":
        return ACTIVE_IVA, AUX_OVER_IVA_CONFIG
    elif ACTIVE_IVA == "AUX_OVER_IVA_ONLINE":
        return ACTIVE_IVA, AUX_OVER_IVA_ONLINE_CONFIG
    else:
        raise ValueError(f"Unknown IVA algorithm: {ACTIVE_IVA}")


def get_ilrma_config():
    """获取当前激活的ILRMA配置"""
    if ACTIVE_ILRMA == "ILRMA":
        return ACTIVE_ILRMA, ILRMA_CONFIG
    elif ACTIVE_ILRMA == "ILRMA_V2":
        return ACTIVE_ILRMA, ILRMA_V2_CONFIG
    elif ACTIVE_ILRMA == "ILRMA_SR":
        return ACTIVE_ILRMA, ILRMA_SR_CONFIG
    elif ACTIVE_ILRMA == "ILRMA_REALTIME":
        return ACTIVE_ILRMA, ILRMA_REALTIME_CONFIG
    else:
        raise ValueError(f"Unknown ILRMA algorithm: {ACTIVE_ILRMA}")


def print_config(algo_type="IVA"):
    """打印当前配置"""
    print("=" * 60)
    print(f"当前测试配置 - {algo_type}")
    print("=" * 60)
    print(f"音频文件: {AUDIO_FILE}")
    print(f"声道数: {N_CHANNELS}")
    
    if algo_type == "IVA":
        algo_name, config = get_iva_config()
        print(f"算法: {algo_name}")
        for key, value in config.items():
            print(f"  {key}: {value}")
    elif algo_type == "ILRMA":
        algo_name, config = get_ilrma_config()
        print(f"算法: {algo_name}")
        for key, value in config.items():
            print(f"  {key}: {value}")
    elif algo_type == "RCSCME":
        print(f"算法: RCSCME")
        for key, value in RCSCME_CONFIG.items():
            print(f"  {key}: {value}")
    
    print("=" * 60)
