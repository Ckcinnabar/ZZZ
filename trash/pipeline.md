# Speech Emotion Classification Pipeline
## Librosa + CNN + LSTM Architecture

**Project**: Speech Emotion Classification System
**Framework**: TensorFlow/Keras with GPU
**Target**: 5 emotions (happy, sad, angry, surprised, neutral)
**Success Metric**: Weighted F1-score ≥ 0.60

---

## Pipeline Overview

```
Audio Files (.wav, 3-sec)
    ↓
[1] Data Collection & Organization
    ↓
[2] Train/Test Split (stratified, speaker-independent)
    ↓
[3] Exploratory Data Analysis
    ↓
[4] Feature Extraction (Librosa → Mel-Spectrograms)
    ↓
[5] Data Augmentation (expand 5-8x)
    ├─ Time Stretching
    ├─ Pitch Shifting
    ├─ Background Noise
    └─ SpecAugment
    ↓
[6] Baseline Model (Random Forest)
    ↓
[7] CNN+LSTM Model Training
    ↓
[8] Final Evaluation (Blind Test Set)
```

---

## Notebook Sequence

### Phase 1: Data Foundation

#### **01_data_collection_organization.ipynb**
**Purpose**: Load and organize audio dataset

**Inputs**:
- Raw .wav files (3-second clips)
- Emotion labels

**Outputs**:
- `data/metadata.csv` - (filename, emotion, speaker_id, duration, sample_rate)
- Data integrity report

**Key Steps**:
1. Load all .wav files from data directory
2. Verify audio properties:
   - Duration = 3 seconds (±0.1s tolerance)
   - Sample rate consistency (recommend 22050 Hz or 16000 Hz)
   - No corrupted files
3. Create metadata CSV with speaker attribution
4. Visualize class distribution
5. Set random seeds (Python, NumPy: seed=42)

**Constitution Compliance**:
- ✅ Reproducible (random seed set)
- ✅ Documented (markdown cells explain each step)
- ✅ Data integrity (verification checks)

---

#### **02_train_test_split.ipynb**
**Purpose**: Create stratified train/test split (CRITICAL - Constitution Principle III)

**Inputs**:
- `data/metadata.csv`

**Outputs**:
- `data/train_split.csv` - Training set filenames and labels
- `data/test_split.csv` - Test set filenames and labels
- Split statistics report

**Key Steps**:
1. **CRITICAL**: Ensure speaker-independent split
   - No speaker appears in both train and test
   - Prevents data leakage
2. Stratified split (80/20 or 70/30) maintaining class balance
3. Set random seed (seed=42) for reproducibility
4. Save split indices permanently
5. Document split statistics:
   - Train: N samples (X per class)
   - Test: M samples (Y per class)

**Constitution Compliance**:
- ✅ NON-NEGOTIABLE: Zero test data leakage
- ✅ Reproducible (saved indices, random seed)
- ✅ Auditable (split logged and versioned)

**Warning**: Test set is now FROZEN. Never examine or use for model development.

---

#### **03_exploratory_data_analysis.ipynb**
**Purpose**: Visual inspection and data quality analysis

**Inputs**:
- Training set audio files only
- `data/train_split.csv`

**Outputs**:
- Waveform visualizations
- Spectrogram examples per emotion
- Data quality report

**Key Steps**:
1. Load sample audio files per emotion class
2. Visualize waveforms:
   - Time-domain signals
   - Amplitude characteristics per emotion
3. Generate spectrograms for visual inspection
4. Audio quality checks:
   - Silence detection (clip too quiet?)
   - Clipping detection (amplitude > 1.0)
   - Background noise levels
5. Statistical analysis:
   - Duration distribution
   - Amplitude statistics per class
   - Class balance verification

**Constitution Compliance**:
- ✅ Only uses training data
- ✅ Documented findings in markdown
- ✅ Visualizations explain data characteristics

---

### Phase 2: Feature Engineering

#### **04_feature_extraction_librosa.ipynb**
**Purpose**: Extract mel-spectrograms using Librosa

**Inputs**:
- Training audio files: `data/train_split.csv`
- Test audio files: `data/test_split.csv`

**Outputs**:
- `features/train_melspec.npy` - Training mel-spectrograms (shape: N_train, time, mels)
- `features/test_melspec.npy` - Test mel-spectrograms (shape: N_test, time, mels)
- `features/train_labels.npy` - Training labels (encoded as integers 0-4)
- `features/test_labels.npy` - Test labels
- Feature extraction configuration log

**Key Steps**:
1. **Configuration** (saved for reproducibility):
   ```python
   SR = 22050  # Sample rate
   N_FFT = 1024  # FFT window size
   HOP_LENGTH = 512  # Hop length
   N_MELS = 128  # Number of mel bands
   FMAX = 8000  # Maximum frequency (Hz)
   ```

2. For each audio file:
   ```python
   # Load audio
   y, sr = librosa.load(wav_file, sr=SR)

   # Extract mel-spectrogram
   melspec = librosa.feature.melspectrogram(
       y=y, sr=sr,
       n_fft=N_FFT,
       hop_length=HOP_LENGTH,
       n_mels=N_MELS,
       fmax=FMAX
   )

   # Convert to log scale (dB)
   melspec_db = librosa.power_to_db(melspec, ref=np.max)
   ```

3. Expected output shape:
   - 3-second audio at 22050 Hz = 66,150 samples
   - Hop length 512 → ~129 time frames
   - Final shape per sample: **(128 mels, ~129 time frames)**

4. Save as numpy arrays (.npy) for fast loading

5. Label encoding:
   ```python
   emotion_map = {
       'happy': 0,
       'sad': 1,
       'angry': 2,
       'surprised': 3,
       'neutral': 4
   }
   ```

**Constitution Compliance**:
- ✅ Reproducible (parameters logged)
- ✅ Train/test processed separately (no leakage)
- ✅ Versioned output (features_v1/)

---

#### **05_data_augmentation.ipynb**
**Purpose**: Expand training set 5-8x using audio augmentation

**Inputs**:
- Training audio files only (from `train_split.csv`)

**Outputs**:
- `features/train_melspec_augmented.npy` - Expanded training set (5-8x larger)
- `features/train_labels_augmented.npy` - Corresponding labels
- Augmentation strategy report

**Augmentation Techniques**:

1. **Time Stretching** (±10-15% speed change)
   ```python
   y_stretched = librosa.effects.time_stretch(y, rate=rate)
   # rate ∈ [0.85, 0.90, 1.10, 1.15]
   ```

2. **Pitch Shifting** (±2 semitones)
   ```python
   y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n)
   # n ∈ [-2, -1, +1, +2]
   ```

3. **Background Noise Addition**
   ```python
   noise = np.random.normal(0, 0.005, len(y))
   y_noisy = y + noise
   ```

4. **SpecAugment** (applied to mel-spectrograms)
   ```python
   # Time masking: mask random time frames
   # Frequency masking: mask random mel bands
   ```

**Strategy** (example for < 50 samples/class):
- Original: ~40-50 samples per class
- Apply 2-3 augmentations per sample
- Target: 200-400 samples per class after augmentation

**Key Steps**:
1. For each training sample, generate augmented versions
2. Extract mel-spectrograms from augmented audio
3. Apply SpecAugment to mel-spectrograms
4. Combine original + augmented data
5. Shuffle augmented dataset
6. Report final dataset size

**Constitution Compliance**:
- ✅ Only augments training data (test set untouched)
- ✅ Reproducible (random seeds set)
- ✅ Documented (augmentation parameters logged)

**Critical**: NEVER augment test data. This violates evaluation integrity.

---

### Phase 3: Modeling

#### **06_baseline_model.ipynb**
**Purpose**: Establish baseline performance with simple ML model

**Inputs**:
- `features/train_melspec.npy` (original, not augmented)
- `features/test_melspec.npy`
- Labels

**Outputs**:
- Baseline performance metrics
- Comparison benchmark for CNN+LSTM

**Approach**:
1. **Feature aggregation**: Reduce mel-spectrogram to fixed features
   ```python
   # Per mel-spectrogram, extract:
   - Mean, std, min, max across time axis
   - Result: 128 mels × 4 statistics = 512 features
   ```

2. **Model**: Random Forest or Logistic Regression
   ```python
   from sklearn.ensemble import RandomForestClassifier
   rf = RandomForestClassifier(n_estimators=100, random_state=42)
   rf.fit(X_train_agg, y_train)
   ```

3. **Evaluation**:
   - Accuracy
   - Weighted F1-score
   - Per-class precision, recall, F1
   - Confusion matrix

**Expected Performance**: 0.40-0.50 weighted F1 (simple features)

**Constitution Compliance**:
- ✅ Baseline documented
- ✅ Evaluation metrics match standards (weighted F1)
- ✅ Confusion matrix visualized

---

#### **07_cnn_lstm_model.ipynb**
**Purpose**: Train CNN+LSTM deep learning model

**Inputs**:
- `features/train_melspec_augmented.npy` - Augmented training data
- `features/test_melspec.npy` - Test data (for final evaluation only)
- Labels

**Outputs**:
- `models/cnn_lstm_best.h5` - Trained model
- Training history (loss, accuracy curves)
- Validation performance
- Model architecture summary

**Architecture** (lightweight for small dataset):

```python
from tensorflow.keras import layers, models

def build_cnn_lstm(input_shape=(128, 129, 1), num_classes=5):
    model = models.Sequential([
        # Input: (128 mels, 129 time frames, 1 channel)

        # CNN Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # → (64, 64, 32)
        layers.Dropout(0.3),

        # CNN Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # → (32, 32, 64)
        layers.Dropout(0.3),

        # CNN Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        # Pool only along frequency axis, preserve time
        layers.MaxPooling2D((2, 1)),  # → (16, 32, 128)
        layers.Dropout(0.4),

        # Reshape for LSTM: (time_steps, features)
        layers.Reshape((32, 16*128)),  # → (32 time steps, 2048 features)

        # LSTM Layer
        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
        layers.Dropout(0.5),

        # Dense Layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
```

**Training Configuration**:

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Class weights (for imbalanced data)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Callbacks
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

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,  # 20% of training for validation
    epochs=100,
    batch_size=16,  # Small batch for small dataset
    class_weight=dict(enumerate(class_weights)),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
```

**Regularization Strategy** (critical for small dataset):
- Heavy dropout (0.3 → 0.5 through network)
- BatchNormalization
- L2 regularization (optional)
- Early stopping (patience=15)
- Data augmentation (already applied)

**Training Monitoring**:
1. Plot training/validation loss curves
2. Plot training/validation accuracy curves
3. Monitor for overfitting (train >> val performance)

**Constitution Compliance**:
- ✅ Reproducible (random seeds, saved model)
- ✅ Hyperparameters documented
- ✅ Training process logged

---

### Phase 4: Evaluation

#### **08_final_evaluation.ipynb**
**Purpose**: Blind test set evaluation (Constitution Principle IV)

**Inputs**:
- `models/cnn_lstm_best.h5` - Trained model
- `features/test_melspec.npy` - FROZEN test set (untouched during training)
- `features/test_labels.npy`

**Outputs**:
- **Weighted F1-score** (primary metric)
- Per-class precision, recall, F1-score
- Confusion matrix (visual)
- Classification report
- Error analysis

**Evaluation Steps**:

```python
# Load model
from tensorflow.keras.models import load_model
model = load_model('models/cnn_lstm_best.h5')

# Predict on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

# Weighted F1-score (PRIMARY METRIC)
weighted_f1 = f1_score(y_test, y_pred_classes, average='weighted')

# Per-class metrics
report = classification_report(
    y_test, y_pred_classes,
    target_names=['happy', 'sad', 'angry', 'surprised', 'neutral']
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
```

**Success Criteria Check**:
- ✅ Weighted F1-score ≥ 0.60 (target)
- ✅ Per-class F1 ≥ 0.40 for all emotions
- ✅ Zero overlap between train/test sets (verified in notebook 02)

**Visualizations**:
1. **Confusion Matrix Heatmap**
   - Shows which emotions are confused
   - Diagonal = correct predictions

2. **Per-Class Performance Bar Chart**
   - Precision, Recall, F1 for each emotion

3. **Error Analysis**:
   - Identify most misclassified samples
   - Listen to audio, examine mel-spectrograms
   - Hypothesize why model failed

**Constitution Compliance**:
- ✅ Evaluation on blind test set only
- ✅ Weighted F1-score reported (required metric)
- ✅ Confusion matrix visualized
- ✅ Per-category metrics provided

---

## Directory Structure

```
ml/
├── pipeline.md                          # This file
├── requirements.txt                     # Python dependencies
│
├── data/                                # Audio data (not committed to git if large)
│   ├── raw/                             # Original .wav files
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── angry/
│   │   ├── surprised/
│   │   └── neutral/
│   ├── metadata.csv                     # File inventory
│   ├── train_split.csv                  # Training set (FROZEN)
│   └── test_split.csv                   # Test set (FROZEN)
│
├── features/                            # Extracted features
│   ├── train_melspec.npy                # Original training mel-spectrograms
│   ├── train_melspec_augmented.npy      # Augmented training data
│   ├── train_labels.npy
│   ├── train_labels_augmented.npy
│   ├── test_melspec.npy                 # Test mel-spectrograms
│   └── test_labels.npy
│
├── models/                              # Trained models
│   ├── baseline_rf.pkl                  # Baseline Random Forest
│   └── cnn_lstm_best.h5                 # Best CNN+LSTM model
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
└── results/                             # Evaluation outputs
    ├── confusion_matrix.png
    ├── training_curves.png
    └── classification_report.txt
```

---

## Key Technical Details

### Mel-Spectrogram Configuration
```python
SAMPLE_RATE = 22050      # 22.05 kHz (standard for speech)
N_FFT = 1024             # FFT window size
HOP_LENGTH = 512         # 50% overlap
N_MELS = 128             # Mel frequency bands
F_MAX = 8000             # Max frequency (Hz) - focus on speech range
```

**Resulting Shape**: (128 mels, ~129 time frames) for 3-second audio

### CNN+LSTM Architecture Summary
- **Parameters**: ~500K-1M (lightweight)
- **Input**: (batch, 128, 129, 1)
- **CNN**: 3 blocks (32→64→128 filters)
- **LSTM**: Bidirectional, 128 units
- **Output**: 5-class softmax
- **Regularization**: Dropout (0.3→0.5), BatchNorm, early stopping

### Training Strategy for Small Dataset
1. **Heavy augmentation**: 5-8x expansion (all 4 techniques)
2. **Strong regularization**: Dropout 0.3-0.5
3. **Early stopping**: Patience=15 epochs
4. **Small batches**: 16-32 samples
5. **Class weighting**: Handle imbalanced classes
6. **Validation monitoring**: Track overfitting

---

## Expected Performance

### Baseline (Random Forest)
- Weighted F1: 0.40-0.50
- Simple aggregated features
- Fast to train (<1 min)

### CNN+LSTM (Target)
- Weighted F1: **≥ 0.60** (success criterion)
- Per-class F1: **≥ 0.40** for all emotions
- Training time: 10-30 min (GPU), 2-5 hours (CPU)

### Challenges with Small Dataset
- **Overfitting risk**: High - mitigated by augmentation + regularization
- **Generalization**: May struggle with new speakers
- **Class imbalance**: Likely 'neutral' overrepresented - use class weights

---

## Environment Setup

### 1. Install Python 3.8+

### 2. Create Virtual Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify GPU Setup (if using NVIDIA GPU)
```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
```

### 5. Launch Jupyter
```bash
jupyter notebook
```

---

## Constitution Compliance Checklist

Each notebook MUST satisfy:

- [ ] **Principle I**: Implemented in Jupyter notebook (.ipynb)
- [ ] **Principle II**: Reproducible top-to-bottom execution
  - [ ] Random seeds set (Python, NumPy, TensorFlow)
  - [ ] Data paths explicit
  - [ ] No manual interventions required
- [ ] **Principle III**: Data integrity (NON-NEGOTIABLE)
  - [ ] Train/test split created ONCE and frozen
  - [ ] No test data leakage
  - [ ] Transformations logged and versioned
- [ ] **Principle IV**: Evaluation standards
  - [ ] Weighted F1-score reported
  - [ ] Per-class metrics provided
  - [ ] Confusion matrix visualized
- [ ] **Principle V**: Documentation
  - [ ] Title cell with purpose
  - [ ] Markdown sections explaining steps
  - [ ] Input/output specification
  - [ ] Summary cell with findings

---

## Next Steps

1. **Collect/organize audio data** (< 50 samples per class target)
2. **Set up environment** (`pip install -r requirements.txt`)
3. **Start with notebook 01**: Data collection and organization
4. **Follow pipeline sequentially**: Each notebook depends on previous outputs
5. **Monitor constitution compliance**: Check principles before moving forward

---

## References

- **Librosa Documentation**: https://librosa.org/doc/latest/index.html
- **TensorFlow/Keras**: https://www.tensorflow.org/api_docs/python/tf/keras
- **SpecAugment Paper**: Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
- **Constitution**: `.specify/memory/constitution.md`

---

**Version**: 1.0.0
**Created**: 2025-11-19
**Last Updated**: 2025-11-19
