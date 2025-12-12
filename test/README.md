# BSS 算法测试框架

简化的测试框架，方便快速切换不同算法和参数进行测试。

## 文件结构

```txt
test/
├── test_config.py          # 配置文件 - 在这里修改所有参数
├── test_iva_unified.py     # IVA算法统一测试脚本（新）
├── test_ilrma_unified.py   # ILRMA算法统一测试脚本（新）
├── test_iva_ng.py          # IVA_NG原始测试（旧）
├── test_iva.py             # AUX_IVA原始测试（旧）
├── test_ilrma.py           # ILRMA原始测试（旧）
├── test_rcscme.py          # RCSCME算法测试脚本
├── test_ilrma_sr.py        # ILRMA_SR专用测试（生成mixture和steering vector）
└── README.md               # 本文档
```

## 使用方法

### 1. 测试 IVA 算法

**步骤：**

1. 编辑 `test_config.py`
2. 设置 `AUDIO_FILE` 为你的测试音频路径
3. 设置 `ACTIVE_IVA` 为 `"IVA_NG"` 或 `"AUX_IVA"`
4. 调整对应的配置参数
5. 运行测试：

```bash
python test/test_iva_unified.py
```

**配置示例：**

```python
# test_config.py

AUDIO_FILE = "/Users/kolor/myWork/data/地铁-0626.wav"
ACTIVE_IVA = "IVA_NG"  # 或 "AUX_IVA"

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
    "contrast_func": "laplace",  # 或 "gaussian"
    "ref_mic": 1,
}
```

### 2. 测试 ILRMA 算法

**步骤：**

1. 编辑 `test_config.py`
2. 设置 `AUDIO_FILE` 为你的测试音频路径
3. 设置 `ACTIVE_ILRMA` 为 `"ILRMA"`, `"ILRMA_V2"`, 或 `"ILRMA_SR"`
4. 调整对应的配置参数
5. 运行测试：

```bash
python test/test_ilrma_unified.py
```

**配置示例：**

```python
# test_config.py

AUDIO_FILE = "/Users/kolor/myWork/data/地铁-0626.wav"
ACTIVE_ILRMA = "ILRMA_V2"  # 或 "ILRMA", "ILRMA_SR"

ILRMA_CONFIG = {
    "n_iter": 100,
    "frame_shift": 512,
    "n_components": 2,
    "k_NMF_bases": 8,
}

ILRMA_V2_CONFIG = {
    "n_iter": 100,
    "frame_shift": 512,
    "n_components": 2,
    "k_NMF_bases": 8,
}
```

### 3. 对比不同实现

要对比不同实现（例如 IVA_NG vs AUX_IVA），只需：

1. 修改 `test_config.py` 中的 `ACTIVE_IVA`，运行一次
2. 再次修改 `ACTIVE_IVA` 为另一个算法，再运行一次
3. 比较 `test_outputs/` 目录下的输出文件

输出文件会自动以算法名称命名，例如：

- `test_outputs/地铁-0626_IVA_NG.wav`
- `test_outputs/地铁-0626_AUX_IVA.wav`

### 4. 对比不同参数

要对比不同参数（例如不同的 learning_rate），手动修改配置：

```python
# 第一次运行
IVA_NG_CONFIG = {
    "learning_rate": 0.1,
    # ... 其他参数
}
# 运行 test_iva_unified.py 后，手动重命名输出为 地铁-0626_IVA_NG_lr0.1.wav

# 第二次运行
IVA_NG_CONFIG = {
    "learning_rate": 0.05,
    # ... 其他参数
}
# 运行 test_iva_unified.py 后，手动重命名输出为 地铁-0626_IVA_NG_lr0.05.wav
```

## 参数说明

### IVA 参数

| 参数 | 说明 | IVA_NG | AUX_IVA |
|------|------|--------|---------|
| `n_iter` | 迭代次数 | ✓ | ✓ |
| `learning_rate` | 学习率 | ✓ | ✗ |
| `frame_shift` | 帧移（STFT） | ✓ | ✓ |
| `n_components` | 源数量 | ✓ | ✗ |
| `contrast_func` | 对比函数 | ✗ | ✓ |
| `ref_mic` | 参考麦克风 | ✓ | ✓ |

### ILRMA 参数

| 参数 | 说明 | ILRMA | ILRMA_V2 | ILRMA_SR |
|------|------|-------|----------|----------|
| `n_iter` | 迭代次数 | ✓ | ✓ | ✓ |
| `frame_shift` | 帧移（STFT） | ✓ | ✓ | ✓ |
| `n_components` | 源数量 | ✓ | ✓ | ✓ |
| `k_NMF_bases` | NMF基数量 | ✓ | ✓ | ✓ |

## 输出管理

- 所有输出默认保存在 `test_outputs/` 目录
- 可在 `test_config.py` 中修改 `OUTPUT_DIR`
- 可设置 `SAVE_OUTPUT = False` 来禁用保存

## 常见使用场景

### 场景1: 测试不同的W迭代方法

```python
# 测试 IVA_NG（Natural Gradient更新）
ACTIVE_IVA = "IVA_NG"
# 运行 test_iva.py

# 测试 AUX_IVA（Auxiliary Function更新）
ACTIVE_IVA = "AUX_IVA"
# 运行 test_iva.py
```

### 场景2: 测试不同的对比函数

```python
ACTIVE_IVA = "AUX_IVA"

# 测试 Gaussian
AUX_IVA_CONFIG["contrast_func"] = "gaussian"
# 运行 test_iva_unified.py

# 测试 Laplace
AUX_IVA_CONFIG["contrast_func"] = "laplace"
# 运行 test_iva_unified.py
```

### 场景3: 测试不同的NMF基数

```python
ACTIVE_ILRMA = "ILRMA_V2"

# 测试 K=8
ILRMA_V2_CONFIG["k_NMF_bases"] = 8
# 运行 test_ilrma_unified.py

# 测试 K=16
ILRMA_V2_CONFIG["k_NMF_bases"] = 16
# 运行 test_ilrma_unified.py
```

## 快速开始示例

```bash
# 1. 修改 test_config.py，设置你的音频文件路径
# AUDIO_FILE = "/path/to/your/audio.wav"

# 2. 测试 IVA_NG
python test/test_iva_unified.py

# 3. 修改 test_config.py，切换到 AUX_IVA
# ACTIVE_IVA = "AUX_IVA"

# 4. 再次测试
python test/test_iva_unified.py

# 5. 比较 test_outputs/ 目录下的两个输出文件
```

## 旧文件

旧的测试文件（`test_iva_ng.py`, `test_iva.py`, `test_ilrma.py` 等）已被保留，如果需要可以继续使用。新的统一测试脚本（`test_iva_unified.py` 和 `test_ilrma_unified.py`）提供了更方便的配置管理方式。
