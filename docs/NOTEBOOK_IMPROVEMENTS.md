# Notebook 改进建议

本文档列出了 `main_pipeline_updated.ipynb` 的改进建议和具体修复方案。

## 目录
1. [严重问题（需要立即修复）](#严重问题)
2. [中等问题（影响用户体验）](#中等问题)
3. [轻微问题（可优化）](#轻微问题)
4. [推荐的最佳实践](#推荐的最佳实践)

---

## 严重问题

### 1. Cell 2 和 Cell 3 重复安装依赖

**问题位置**: Cell 2-3

**当前代码**:
```python
# Cell 2
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/Mo119m/primates-sound-detection.git
%cd primates-sound-detection
!pip install -q -r requirements.txt

# Cell 3
!pip install -q librosa soundfile tensorflow scikit-learn pandas matplotlib
from google.colab import drive
drive.mount('/content/drive')
```

**建议修复**: 合并为一个setup cell

```python
# Cell: 0. Setup & Installation
from google.colab import drive
import sys
import os

# Mount Google Drive
drive.mount('/content/drive')

# Clone repository if not exists
if not os.path.exists('/content/primates-sound-detection'):
    !git clone https://github.com/Mo119m/primates-sound-detection.git
    %cd primates-sound-detection
else:
    %cd primates-sound-detection
    print("Repository already exists, skipping clone.")

# Install dependencies
!pip install -q -r requirements.txt

# Add src to path
sys.path.append('src')

print("✓ Setup complete!")
```

---

### 2. Cell 26 引用不存在的脚本

**问题位置**: Cell 26

**当前代码**:
```python
exec(open('run_hard_negative_mining.py').read())
```

**问题**: `run_hard_negative_mining.py` 文件不存在

**建议修复1**: 如果有hard_negative_mining模块
```python
# Import and run hard negative mining
from hard_negative_mining import extract_candidates

# Extract uncertain predictions as hard negative candidates
candidates_dir = extract_candidates(
    trained_model,
    long_audio_files,
    output_dir=os.path.join(config.AUDIO_ROOT, 'hard_negative_candidates'),
    min_confidence=0.5,
    max_confidence=0.85
)

print(f"✓ Hard negative candidates extracted to: {candidates_dir}")
print("\nNext steps:")
print("1. Go to Google Drive and listen to files in 'hard_negative_candidates'")
print("2. Delete actual primate calls (model is correct)")
print("3. Keep false positives (background sounds)")
print("4. Move verified files to 'verified_hard_negatives' folder")
```

**建议修复2**: 如果没有模块，使用detection模块
```python
# Extract hard negative candidates using detection module
import os

# Create output directory
candidates_dir = os.path.join(config.AUDIO_ROOT, 'hard_negative_candidates')
os.makedirs(candidates_dir, exist_ok=True)

# Process long audio files and extract uncertain predictions
uncertain_detections = []

for audio_file in long_audio_files[:5]:  # Start with first 5 files
    print(f"Processing {os.path.basename(audio_file)}...")

    # Detect with lower threshold to capture uncertain predictions
    detections = detection.detect_in_long_audio(
        trained_model,
        audio_file,
        confidence_threshold=0.5  # Lower threshold
    )

    # Filter for uncertain predictions (0.5 - 0.85 confidence)
    uncertain = detections[
        (detections['confidence'] >= 0.5) &
        (detections['confidence'] <= 0.85)
    ]

    if len(uncertain) > 0:
        # Extract clips
        utils.extract_detected_audio_clips(
            audio_file,
            uncertain,
            candidates_dir,
            padding=0.5
        )

    print(f"  Found {len(uncertain)} uncertain predictions")

print(f"\n✓ Extracted {len(os.listdir(candidates_dir))} candidate clips")
print(f"✓ Saved to: {candidates_dir}")
```

---

### 3. 缺少必要的imports

**问题位置**: Cell 17, 22, 等多处

**建议修复**: 在Cell 5添加所有必要的imports

```python
# Cell 5: Import all necessary modules
import config
import data_loader
import preprocessing
import augmentation
import model as model_module
import train
import detection
import utils

# Standard library imports
import os
import sys
from pathlib import Path

# Data science imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Colab specific
from IPython.display import display, HTML, Audio
import warnings
warnings.filterwarnings('ignore')

# Check TensorFlow/GPU
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"Keras backend: {tf.keras.backend.backend()}")

# Print configuration summary
config.print_config_summary()
```

---

## 中等问题

### 4. 缺少错误处理

**建议**: 在关键操作处添加try-except块

**示例1: 音频文件加载**
```python
# Cell 13: Get long audio files with error handling
import os

try:
    long_audio_files = data_loader.get_long_audio_files()

    if len(long_audio_files) == 0:
        raise ValueError("No long audio files found! Please check LONG_AUDIO_ROOT path.")

    print(f"✓ Found {len(long_audio_files)} long audio files:")
    for i, file in enumerate(long_audio_files[:10], 1):
        print(f"  {i}. {os.path.basename(file)}")
    if len(long_audio_files) > 10:
        print(f"  ... and {len(long_audio_files) - 10} more")

except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print(f"\nPlease check that the following directory exists:")
    print(f"  {config.LONG_AUDIO_ROOT}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
```

**示例2: 模型加载**
```python
# Cell: Load trained model with error handling
import os

try:
    model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.h5')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    trained_model = model_module.load_trained_model(model_path)
    print(f"✓ Model loaded successfully from: {model_path}")

except FileNotFoundError as e:
    print(f"✗ {e}")
    print("\nPlease train a model first by running the training cell.")
except Exception as e:
    print(f"✗ Error loading model: {e}")
```

---

### 5. 内存管理优化

**问题**: 处理大量长音频文件时可能内存溢出

**建议**: 添加内存清理和批处理

```python
# Cell 19: Process all long audio files with memory management
import gc

# Process in batches to avoid memory issues
BATCH_SIZE = 10  # Process 10 files at a time
all_detections = {}

for i in range(0, len(long_audio_files), BATCH_SIZE):
    batch = long_audio_files[i:i+BATCH_SIZE]

    print(f"\n Processing batch {i//BATCH_SIZE + 1}/{(len(long_audio_files)-1)//BATCH_SIZE + 1}")
    print(f"Files {i+1} to {min(i+BATCH_SIZE, len(long_audio_files))}")

    for audio_file in batch:
        filename = os.path.basename(audio_file)
        print(f"  Processing: {filename}...")

        try:
            detections_df = detection.detect_in_long_audio(
                trained_model,
                audio_file,
                confidence_threshold=config.DETECTION_CONFIDENCE_THRESHOLD
            )

            all_detections[filename] = detections_df

            # Save immediately
            csv_path = detection.save_detections(detections_df, filename)
            print(f"    ✓ Found {len(detections_df)} detections, saved to {csv_path}")

        except Exception as e:
            print(f"    ✗ Error processing {filename}: {e}")
            all_detections[filename] = pd.DataFrame()  # Empty dataframe for failed files

    # Clear memory after each batch
    gc.collect()

print(f"\n Completed processing {len(all_detections)} files")
```

---

### 6. 训练输出优化

**问题**: Cell 10输出过大

**建议**: 使用回调函数控制输出或使用tqdm进度条

```python
# Add to train.py or use in cell 10
from tensorflow.keras.callbacks import Callback
from tqdm.keras import TqdmCallback

class SimpleLogs(Callback):
    """Simplified logging for cleaner notebook output"""
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, "
              f"acc={logs['accuracy']:.4f}, "
              f"val_loss={logs['val_loss']:.4f}, "
              f"val_acc={logs['val_accuracy']:.4f}")

# In training cell:
# Add SimpleLogs() or TqdmCallback(verbose=1) to callbacks list
```

---

## 轻微问题

### 7. 变量命名规范

**建议**: 统一使用描述性命名

```python
# 当前命名
detections_df           # 初始检测结果
improved_detections     # 改进后的检测结果
all_detections         # 所有文件的检测结果

# 建议命名
detections_v1          # 版本1（初始模型）
detections_v2          # 版本2（hard negative mining后）
detections_all_files   # 所有文件的检测结果字典
```

---

### 8. GPU检测和配置

**建议**: 在setup阶段添加GPU检测

```python
# Cell: GPU Check and Configuration
import tensorflow as tf

print("=== GPU Configuration ===")
print(f"TensorFlow version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU detected: {len(gpus)} device(s)")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")

    # Enable memory growth to prevent TensorFlow from allocating all GPU memory
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"Warning: {e}")
else:
    print("⚠ No GPU detected - training will use CPU (much slower)")
    print("Recommendation: Use GPU runtime (Runtime > Change runtime type > GPU)")
```

---

### 9. 数据验证

**建议**: 在训练前验证数据质量

```python
# Cell: Data Quality Check (add after Cell 8)
import librosa

print("=== Data Quality Check ===\n")

# Check species data
print("Species Data:")
for species, clips in species_data.items():
    durations = [librosa.get_duration(y=clip, sr=config.SAMPLE_RATE) for clip in clips[:10]]
    avg_duration = np.mean(durations)
    print(f"  {species}: {len(clips)} clips, avg duration: {avg_duration:.2f}s")

    if abs(avg_duration - config.CLIP_DURATION) > 0.5:
        print(f"    ⚠ Warning: Duration mismatch! Expected {config.CLIP_DURATION}s")

# Check background data
bg_durations = [librosa.get_duration(y=clip, sr=config.SAMPLE_RATE)
                for clip in background_data[:10]]
avg_bg_duration = np.mean(bg_durations)
print(f"\nBackground: {len(background_data)} clips, avg duration: {avg_bg_duration:.2f}s")

# Check for audio quality issues
print("\n Checking for silent or corrupted clips...")
silent_count = 0
for clips in species_data.values():
    for clip in clips[:20]:  # Sample first 20
        if np.max(np.abs(clip)) < 0.01:  # Very quiet
            silent_count += 1

if silent_count > 0:
    print(f"⚠ Warning: Found {silent_count} potentially silent clips")
else:
    print("✓ All sampled clips have audio signal")
```

---

## 推荐的最佳实践

### 1. Cell结构建议

推荐的cell顺序和组织：

```
1. Setup (环境配置、安装、imports)
2. GPU Check (GPU检测和配置)
3. Configuration (配置文件和参数打印)
4. Data Loading (数据加载)
5. Data Quality Check (数据验证)
6. Data Exploration (可选：数据探索和可视化)
7. Training (模型训练)
8. Model Evaluation (模型评估)
9. Detection - Single File (单个文件测试)
10. Detection - All Files (所有文件处理)
11. Analysis & Reporting (分析和报告)
12. Hard Negative Mining (可选)
13. Model Comparison (可选)
14. Export & Save (导出和保存)
```

### 2. 添加进度指示器

```python
# 使用tqdm显示进度
from tqdm.auto import tqdm

for file in tqdm(long_audio_files, desc="Processing files"):
    # process file
    pass
```

### 3. 创建检查点

```python
# 在关键步骤保存中间结果
import pickle

# Save data after loading
checkpoint_path = '/content/drive/MyDrive/chimp-audio/checkpoints'
os.makedirs(checkpoint_path, exist_ok=True)

with open(f'{checkpoint_path}/species_data.pkl', 'wb') as f:
    pickle.dump(species_data, f)

print("✓ Data checkpoint saved")
```

### 4. 添加配置验证

```python
# Cell: Verify Configuration
import os

print("=== Configuration Verification ===\n")

# Check if all required directories exist
dirs_to_check = [
    ('Audio Root', config.AUDIO_ROOT),
    ('Long Audio Root', config.LONG_AUDIO_ROOT),
    ('Output Root', config.OUTPUT_ROOT),
]

all_exist = True
for name, path in dirs_to_check:
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {name}: {path}")
    if not exists:
        all_exist = False

# Check if species folders exist
print("\nSpecies Folders:")
for species, folder in config.SPECIES_FOLDERS.items():
    path = os.path.join(config.AUDIO_ROOT, folder)
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"

    if exists:
        file_count = len([f for f in os.listdir(path) if f.endswith(('.wav', '.mp3'))])
        print(f"{status} {species}: {folder} ({file_count} files)")
    else:
        print(f"{status} {species}: {folder} (NOT FOUND)")
        all_exist = False

if all_exist:
    print("\n All paths verified! Ready to proceed.")
else:
    print("\n⚠ Some paths not found. Please check your Google Drive structure.")
```

---

## 总结

### 优先级修复清单

**必须修复**:
1. ✅ 合并Cell 2和3的重复代码
2. ✅ 修复Cell 26的脚本引用问题
3. ✅ 添加必要的imports

**强烈建议**:
4. ⚡ 添加错误处理机制
5. ⚡ 实现内存管理和批处理
6. ⚡ 添加GPU检测和配置

**可选优化**:
7. 💡 改善训练输出显示
8. 💡 添加数据质量检查
9. 💡 统一变量命名规范

---

## 实施建议

### 方式1: 创建改进版notebook
创建一个新的 `main_pipeline_v2.ipynb`，集成所有改进。

### 方式2: 增量更新
逐步修改现有notebook，每次修改后测试一个完整流程。

### 方式3: 模块化重构
将部分notebook代码移到Python模块中，notebook只保留关键步骤。

---

**文档版本**: 1.0
**最后更新**: 2026-01-11
**维护者**: Claude Code Assistant
