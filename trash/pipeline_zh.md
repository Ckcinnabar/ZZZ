# 語音情緒分類流程
## Librosa + CNN + LSTM 架構

**專案名稱**：語音情緒分類系統
**框架**：TensorFlow/Keras with GPU
**目標**：5種情緒（快樂、悲傷、憤怒、驚訝、中性）
**成功指標**：加權 F1-score ≥ 0.60

---

## 流程概覽

```
音訊檔案 (.wav, 3秒)
    ↓
[1] 資料收集與組織
    ↓
[2] 訓練/測試集分割（分層、說話者獨立）
    ↓
[3] 探索性資料分析
    ↓
[4] 特徵提取（Librosa → Mel-Spectrograms）
    ↓
[5] 資料增強（擴展 5-8 倍）
    ├─ 時間拉伸
    ├─ 音高轉換
    ├─ 背景噪音
    └─ SpecAugment
    ↓
[6] 基準模型（隨機森林）
    ↓
[7] CNN+LSTM 模型訓練
    ↓
[8] 最終評估（盲測試集）
```

---

## Notebook 執行順序

### 階段 1：資料基礎

#### **01_data_collection_organization.ipynb**
**目的**：載入並組織音訊資料集

**輸入**：
- 原始 .wav 檔案（3秒音訊片段）
- 情緒標籤

**輸出**：
- `data/metadata.csv` - （檔名、情緒、說話者ID、時長、採樣率）
- 資料完整性報告

**關鍵步驟**：
1. 從資料目錄載入所有 .wav 檔案
2. 驗證音訊屬性：
   - 時長 = 3秒（±0.1秒容差）
   - 採樣率一致性（建議 22050 Hz 或 16000 Hz）
   - 無損壞檔案
3. 建立包含說話者歸屬的 metadata CSV
4. 視覺化類別分布
5. 設定隨機種子（Python、NumPy: seed=42）

**憲章合規性**：
- ✅ 可重現（設定隨機種子）
- ✅ 已記錄（markdown 單元格解釋每個步驟）
- ✅ 資料完整性（驗證檢查）

---

#### **02_train_test_split.ipynb**
**目的**：建立分層訓練/測試集分割（關鍵 - 憲章原則 III）

**輸入**：
- `data/metadata.csv`

**輸出**：
- `data/train_split.csv` - 訓練集檔名和標籤
- `data/test_split.csv` - 測試集檔名和標籤
- 分割統計報告

**關鍵步驟**：
1. **關鍵**：確保說話者獨立分割
   - 沒有說話者同時出現在訓練和測試集
   - 防止資料洩漏
2. 分層分割（80/20 或 70/30）保持類別平衡
3. 設定隨機種子（seed=42）以確保可重現性
4. 永久保存分割索引
5. 記錄分割統計：
   - 訓練集：N 個樣本（每類 X 個）
   - 測試集：M 個樣本（每類 Y 個）

**憲章合規性**：
- ✅ 不可協商：零測試資料洩漏
- ✅ 可重現（保存索引、隨機種子）
- ✅ 可稽核（分割已記錄和版本化）

**警告**：測試集現已凍結。永遠不要檢查或用於模型開發。

---

#### **03_exploratory_data_analysis.ipynb**
**目的**：視覺檢查和資料品質分析

**輸入**：
- 僅訓練集音訊檔案
- `data/train_split.csv`

**輸出**：
- 波形視覺化
- 每種情緒的頻譜圖範例
- 資料品質報告

**關鍵步驟**：
1. 載入每種情緒類別的樣本音訊檔案
2. 視覺化波形：
   - 時域信號
   - 每種情緒的振幅特徵
3. 生成頻譜圖進行視覺檢查
4. 音訊品質檢查：
   - 靜音檢測（片段太安靜？）
   - 削波檢測（振幅 > 1.0）
   - 背景噪音水平
5. 統計分析：
   - 時長分布
   - 每類的振幅統計
   - 類別平衡驗證

**憲章合規性**：
- ✅ 僅使用訓練資料
- ✅ 在 markdown 中記錄發現
- ✅ 視覺化解釋資料特徵

---

### 階段 2：特徵工程

#### **04_feature_extraction_librosa.ipynb**
**目的**：使用 Librosa 提取 mel-spectrograms

**輸入**：
- 訓練音訊檔案：`data/train_split.csv`
- 測試音訊檔案：`data/test_split.csv`

**輸出**：
- `features/train_melspec.npy` - 訓練 mel-spectrograms（形狀：N_train, time, mels）
- `features/test_melspec.npy` - 測試 mel-spectrograms（形狀：N_test, time, mels）
- `features/train_labels.npy` - 訓練標籤（編碼為整數 0-4）
- `features/test_labels.npy` - 測試標籤
- 特徵提取配置日誌

**關鍵步驟**：
1. **配置**（保存以確保可重現性）：
   ```python
   SR = 22050  # 採樣率
   N_FFT = 1024  # FFT 窗口大小
   HOP_LENGTH = 512  # 跳躍長度
   N_MELS = 128  # Mel 頻帶數量
   FMAX = 8000  # 最大頻率（Hz）
   ```

2. 對每個音訊檔案：
   ```python
   # 載入音訊
   y, sr = librosa.load(wav_file, sr=SR)

   # 提取 mel-spectrogram
   melspec = librosa.feature.melspectrogram(
       y=y, sr=sr,
       n_fft=N_FFT,
       hop_length=HOP_LENGTH,
       n_mels=N_MELS,
       fmax=FMAX
   )

   # 轉換為對數刻度（dB）
   melspec_db = librosa.power_to_db(melspec, ref=np.max)
   ```

3. 預期輸出形狀：
   - 3秒音訊，22050 Hz = 66,150 個樣本
   - 跳躍長度 512 → ~129 時間幀
   - 每個樣本的最終形狀：**（128 mels, ~129 time frames）**

4. 保存為 numpy 陣列（.npy）以便快速載入

5. 標籤編碼：
   ```python
   emotion_map = {
       'happy': 0,
       'sad': 1,
       'angry': 2,
       'surprised': 3,
       'neutral': 4
   }
   ```

**憲章合規性**：
- ✅ 可重現（記錄參數）
- ✅ 訓練/測試分別處理（無洩漏）
- ✅ 版本化輸出（features_v1/）

---

#### **05_data_augmentation.ipynb**
**目的**：使用音訊增強擴展訓練集 5-8 倍

**輸入**：
- 僅訓練音訊檔案（來自 `train_split.csv`）

**輸出**：
- `features/train_melspec_augmented.npy` - 擴展的訓練集（5-8倍大）
- `features/train_labels_augmented.npy` - 對應標籤
- 增強策略報告

**增強技術**：

1. **時間拉伸**（±10-15% 速度變化）
   ```python
   y_stretched = librosa.effects.time_stretch(y, rate=rate)
   # rate ∈ [0.85, 0.90, 1.10, 1.15]
   ```

2. **音高轉換**（±2 半音）
   ```python
   y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n)
   # n ∈ [-2, -1, +1, +2]
   ```

3. **背景噪音添加**
   ```python
   noise = np.random.normal(0, 0.005, len(y))
   y_noisy = y + noise
   ```

4. **SpecAugment**（應用於 mel-spectrograms）
   ```python
   # 時間遮罩：遮蔽隨機時間幀
   # 頻率遮罩：遮蔽隨機 mel 頻帶
   ```

**策略**（< 50 樣本/類的範例）：
- 原始：每類 ~40-50 個樣本
- 對每個樣本應用 2-3 種增強
- 目標：增強後每類 200-400 個樣本

**關鍵步驟**：
1. 為每個訓練樣本生成增強版本
2. 從增強音訊中提取 mel-spectrograms
3. 對 mel-spectrograms 應用 SpecAugment
4. 合併原始 + 增強資料
5. 打亂增強資料集
6. 報告最終資料集大小

**憲章合規性**：
- ✅ 僅增強訓練資料（測試集未觸碰）
- ✅ 可重現（設定隨機種子）
- ✅ 已記錄（記錄增強參數）

**關鍵**：絕不增強測試資料。這違反評估完整性。

---

### 階段 3：建模

#### **06_baseline_model.ipynb**
**目的**：使用簡單 ML 模型建立基準性能

**輸入**：
- `features/train_melspec.npy`（原始，未增強）
- `features/test_melspec.npy`
- 標籤

**輸出**：
- 基準性能指標
- CNN+LSTM 的比較基準

**方法**：
1. **特徵聚合**：將 mel-spectrogram 縮減為固定特徵
   ```python
   # 對每個 mel-spectrogram，提取：
   - 跨時間軸的平均值、標準差、最小值、最大值
   - 結果：128 mels × 4 統計量 = 512 特徵
   ```

2. **模型**：隨機森林或邏輯迴歸
   ```python
   from sklearn.ensemble import RandomForestClassifier
   rf = RandomForestClassifier(n_estimators=100, random_state=42)
   rf.fit(X_train_agg, y_train)
   ```

3. **評估**：
   - 準確率
   - 加權 F1-score
   - 每類的精確度、召回率、F1
   - 混淆矩陣

**預期性能**：加權 F1 約 0.40-0.50（簡單特徵）

**憲章合規性**：
- ✅ 基準已記錄
- ✅ 評估指標符合標準（加權 F1）
- ✅ 混淆矩陣已視覺化

---

#### **07_cnn_lstm_model.ipynb**
**目的**：訓練 CNN+LSTM 深度學習模型

**輸入**：
- `features/train_melspec_augmented.npy` - 增強訓練資料
- `features/test_melspec.npy` - 測試資料（僅用於最終評估）
- 標籤

**輸出**：
- `models/cnn_lstm_best.h5` - 訓練好的模型
- 訓練歷史（損失、準確率曲線）
- 驗證性能
- 模型架構摘要

**架構**（小數據集的輕量級設計）：

```python
from tensorflow.keras import layers, models

def build_cnn_lstm(input_shape=(128, 129, 1), num_classes=5):
    model = models.Sequential([
        # 輸入：(128 mels, 129 時間幀, 1 通道)

        # CNN 區塊 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # → (64, 64, 32)
        layers.Dropout(0.3),

        # CNN 區塊 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # → (32, 32, 64)
        layers.Dropout(0.3),

        # CNN 區塊 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        # 僅沿頻率軸池化，保留時間維度
        layers.MaxPooling2D((2, 1)),  # → (16, 32, 128)
        layers.Dropout(0.4),

        # 重塑為 LSTM：(time_steps, features)
        layers.Reshape((32, 16*128)),  # → (32 時間步, 2048 特徵)

        # LSTM 層
        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
        layers.Dropout(0.5),

        # 全連接層
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
```

**訓練配置**：

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 編譯
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 類別權重（用於不平衡資料）
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# 回調函數
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# 訓練
history = model.fit(
    X_train, y_train,
    validation_split=0.2,  # 20% 訓練資料用於驗證
    epochs=100,
    batch_size=16,  # 小批次適合小數據集
    class_weight=dict(enumerate(class_weights)),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
```

**正則化策略**（對小數據集至關重要）：
- 高 dropout（0.3 → 0.5 穿過網路）
- BatchNormalization
- L2 正則化（可選）
- 早停（patience=15）
- 資料增強（已應用）

**訓練監控**：
1. 繪製訓練/驗證損失曲線
2. 繪製訓練/驗證準確率曲線
3. 監控過擬合（訓練 >> 驗證性能）

**憲章合規性**：
- ✅ 可重現（隨機種子、保存模型）
- ✅ 超參數已記錄
- ✅ 訓練過程已記錄

---

### 階段 4：評估

#### **08_final_evaluation.ipynb**
**目的**：盲測試集評估（憲章原則 IV）

**輸入**：
- `models/cnn_lstm_best.h5` - 訓練好的模型
- `features/test_melspec.npy` - 凍結測試集（訓練期間未觸碰）
- `features/test_labels.npy`

**輸出**：
- **加權 F1-score**（主要指標）
- 每類精確度、召回率、F1-score
- 混淆矩陣（視覺化）
- 分類報告
- 錯誤分析

**評估步驟**：

```python
# 載入模型
from tensorflow.keras.models import load_model
model = load_model('models/cnn_lstm_best.h5')

# 對測試集預測
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 計算指標
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

# 加權 F1-score（主要指標）
weighted_f1 = f1_score(y_test, y_pred_classes, average='weighted')

# 每類指標
report = classification_report(
    y_test, y_pred_classes,
    target_names=['happy', 'sad', 'angry', 'surprised', 'neutral']
)

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred_classes)
```

**成功標準檢查**：
- ✅ 加權 F1-score ≥ 0.60（目標）
- ✅ 每類 F1 ≥ 0.40 適用於所有情緒
- ✅ 訓練/測試集零重疊（在 notebook 02 中驗證）

**視覺化**：
1. **混淆矩陣熱圖**
   - 顯示哪些情緒被混淆
   - 對角線 = 正確預測

2. **每類性能柱狀圖**
   - 每種情緒的精確度、召回率、F1

3. **錯誤分析**：
   - 識別最常被誤分類的樣本
   - 聆聽音訊，檢查 mel-spectrograms
   - 假設模型為何失敗

**憲章合規性**：
- ✅ 僅在盲測試集上評估
- ✅ 報告加權 F1-score（要求指標）
- ✅ 混淆矩陣已視覺化
- ✅ 提供每類指標

---

## 目錄結構

```
ml/
├── pipeline.md                          # 英文版流程文件
├── pipeline_zh.md                       # 中文版流程文件（本檔案）
├── requirements.txt                     # Python 依賴項
│
├── data/                                # 音訊資料（如果大則不提交到 git）
│   ├── raw/                             # 原始 .wav 檔案
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── angry/
│   │   ├── surprised/
│   │   └── neutral/
│   ├── metadata.csv                     # 檔案清單
│   ├── train_split.csv                  # 訓練集（凍結）
│   └── test_split.csv                   # 測試集（凍結）
│
├── features/                            # 提取的特徵
│   ├── train_melspec.npy                # 原始訓練 mel-spectrograms
│   ├── train_melspec_augmented.npy      # 增強訓練資料
│   ├── train_labels.npy
│   ├── train_labels_augmented.npy
│   ├── test_melspec.npy                 # 測試 mel-spectrograms
│   └── test_labels.npy
│
├── models/                              # 訓練好的模型
│   ├── baseline_rf.pkl                  # 基準隨機森林
│   └── cnn_lstm_best.h5                 # 最佳 CNN+LSTM 模型
│
├── notebooks/                           # Jupyter notebooks
│   ├── 01_data_collection_organization.ipynb
│   ├── 02_train_test_split.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_feature_extraction_librosa.ipynb
│   ├── 05_data_augmentation.ipynb
│   ├── 06_baseline_model.ipynb
│   ├── 07_cnn_lstm_model.ipynb
│   └── 08_final_evaluation.ipynb
│
└── results/                             # 評估輸出
    ├── confusion_matrix.png
    ├── training_curves.png
    └── classification_report.txt
```

---

## 關鍵技術細節

### Mel-Spectrogram 配置
```python
SAMPLE_RATE = 22050      # 22.05 kHz（語音標準）
N_FFT = 1024             # FFT 窗口大小
HOP_LENGTH = 512         # 50% 重疊
N_MELS = 128             # Mel 頻帶數
F_MAX = 8000             # 最大頻率（Hz）- 專注於語音範圍
```

**結果形狀**：3秒音訊為 (128 mels, ~129 時間幀)

### CNN+LSTM 架構摘要
- **參數**：~500K-1M（輕量級）
- **輸入**：(batch, 128, 129, 1)
- **CNN**：3 個區塊（32→64→128 濾波器）
- **LSTM**：雙向，128 單元
- **輸出**：5 類 softmax
- **正則化**：Dropout (0.3→0.5), BatchNorm, 早停

### 小數據集訓練策略
1. **大量增強**：5-8 倍擴展（全部 4 種技術）
2. **強正則化**：Dropout 0.3-0.5
3. **早停**：Patience=15 epochs
4. **小批次**：16-32 樣本
5. **類別權重**：處理不平衡類別
6. **驗證監控**：追蹤過擬合

---

## 預期性能

### 基準（隨機森林）
- 加權 F1：0.40-0.50
- 簡單聚合特徵
- 訓練快速（<1 分鐘）

### CNN+LSTM（目標）
- 加權 F1：**≥ 0.60**（成功標準）
- 每類 F1：**≥ 0.40** 適用於所有情緒
- 訓練時間：10-30 分鐘（GPU），2-5 小時（CPU）

### 小數據集的挑戰
- **過擬合風險**：高 - 通過增強 + 正則化緩解
- **泛化能力**：可能難以應對新說話者
- **類別不平衡**：可能「中性」過度代表 - 使用類別權重

---

## 環境設置

### 1. 安裝 Python 3.8+

### 2. 建立虛擬環境（建議）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. 安裝依賴項
```bash
pip install -r requirements.txt
```

### 4. 驗證 GPU 設置（如果使用 NVIDIA GPU）
```python
import tensorflow as tf
print("可用 GPU：", tf.config.list_physical_devices('GPU'))
```

### 5. 啟動 Jupyter
```bash
jupyter notebook
```

---

## 憲章合規性檢查清單

每個 notebook 必須滿足：

- [ ] **原則 I**：在 Jupyter notebook (.ipynb) 中實現
- [ ] **原則 II**：可從頭到尾重現執行
  - [ ] 設定隨機種子（Python、NumPy、TensorFlow）
  - [ ] 資料路徑明確
  - [ ] 無需手動干預
- [ ] **原則 III**：資料完整性（不可協商）
  - [ ] 訓練/測試分割一次建立並凍結
  - [ ] 無測試資料洩漏
  - [ ] 轉換已記錄和版本化
- [ ] **原則 IV**：評估標準
  - [ ] 報告加權 F1-score
  - [ ] 提供每類指標
  - [ ] 混淆矩陣已視覺化
- [ ] **原則 V**：文件記錄
  - [ ] 標題單元格包含目的
  - [ ] Markdown 段落解釋步驟
  - [ ] 輸入/輸出規範
  - [ ] 包含發現的摘要單元格

---

## 下一步

1. **收集/組織音訊資料**（< 50 樣本/類目標）
2. **設置環境**（`pip install -r requirements.txt`）
3. **從 notebook 01 開始**：資料收集和組織
4. **按順序執行流程**：每個 notebook 依賴於前面的輸出
5. **監控憲章合規性**：在繼續之前檢查原則

---

## 參考資料

- **Librosa 文件**：https://librosa.org/doc/latest/index.html
- **TensorFlow/Keras**：https://www.tensorflow.org/api_docs/python/tf/keras
- **SpecAugment 論文**：Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
- **憲章**：`.specify/memory/constitution.md`

---

**版本**：1.0.0
**建立時間**：2025-11-19
**最後更新**：2025-11-19
