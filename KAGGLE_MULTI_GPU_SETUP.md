# Kaggle Multi-GPU Training Guide (2x T4 GPUs)

## Quick Setup for 2x T4 GPUs on Kaggle

Kaggle provides 2x NVIDIA T4 GPUs (15GB VRAM each = 30GB total) for accelerated training. Here's how to utilize both GPUs effectively.

### 1. Enable GPU in Kaggle

1. Open your Kaggle notebook
2. Go to **Settings** (right sidebar)
3. Under **Accelerator**, select: **GPU T4 x2**
4. Click **Save**

### 2. Verify GPU Availability

Add this cell at the beginning of your notebook:

```python
import tensorflow as tf

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs available: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"GPU {i}: {gpu}")
    
# Verify GPU memory
if gpus:
    for gpu in gpus:
        print(f"\n{gpu.name}:")
        print(f"  Memory limit: {tf.config.experimental.get_memory_info(gpu.device_name)}")
```

### 3. Initialize Multi-GPU Strategy

Use TensorFlow's `MirroredStrategy` for data parallelism across 2 GPUs:

```python
import tensorflow as tf

# Create multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()

print(f"Number of devices: {strategy.num_replicas_in_sync}")
# Should output: Number of devices: 2

# Get device information
print("\nDevices:")
for i, device in enumerate(strategy.extended.worker_devices):
    print(f"  Device {i}: {device}")
```

### 4. Build Model Inside Strategy Scope

**CRITICAL**: All model creation must happen inside `strategy.scope()`:

```python
with strategy.scope():
    # Build model
    model = get_news_model(params_final)
    
    # Set learning rate
    model.optimizer.learning_rate.assign(0.0005)
    
    print("✓ Model created on both GPUs")
```

### 5. Scale Batch Size

With 2 GPUs, scale your batch size by 2x:

```python
# Original batch size for single GPU
base_batch_size = 256

# Scale for multi-GPU (2x T4)
global_batch_size = base_batch_size * strategy.num_replicas_in_sync
print(f"Global batch size: {global_batch_size}")  # Should be 512

# Each GPU will process: global_batch_size / num_replicas_in_sync
# = 512 / 2 = 256 per GPU
```

### 6. Train with Multi-GPU

Training code remains mostly the same, just use scaled batch size:

```python
history = model.fit(
    [train_input_ids, train_input_masks, train_segment_ids, train_imagesX],
    trainY_processed,
    batch_size=global_batch_size,  # 512 (256 per GPU)
    epochs=20,
    verbose=1,
    shuffle=True,
    validation_data=(
        [test_input_ids, test_input_masks, test_segment_ids, test_imagesX],
        testY_processed
    ),
    callbacks=[early_stop]
)
```

## Complete Implementation for Kaggle

### Cell 1: Import & GPU Setup

```python
import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt

import cv2
from os import listdir
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
from tensorflow.keras import backend as K

import gc

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Suppress warnings
tf.get_logger().setLevel('ERROR')

print("TensorFlow version:", tf.__version__)

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
print(f"\n{'='*60}")
print(f"GPU CONFIGURATION")
print(f"{'='*60}")
print(f"Number of GPUs available: {len(gpus)}")

if gpus:
    # Enable memory growth to prevent OOM errors
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Memory growth enabled for {gpu.name}")
        except RuntimeError as e:
            print(f"Memory growth setting error: {e}")
    
    # Create multi-GPU strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"\n✓ MirroredStrategy initialized")
    print(f"  Number of devices: {strategy.num_replicas_in_sync}")
    print(f"  Devices: {strategy.extended.worker_devices}")
else:
    print("⚠ No GPUs found, using CPU")
    strategy = tf.distribute.get_strategy()

print(f"{'='*60}\n")
```

### Cell 2: Configuration

```python
# Model configuration
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = 23
img_length = 224
img_width = 224
img_channels = 3

# Multi-GPU batch size configuration
BASE_BATCH_SIZE = 256  # Per GPU
GLOBAL_BATCH_SIZE = BASE_BATCH_SIZE * strategy.num_replicas_in_sync

print(f"Batch Size Configuration:")
print(f"  Base batch size (per GPU): {BASE_BATCH_SIZE}")
print(f"  Global batch size (total): {GLOBAL_BATCH_SIZE}")
print(f"  Effective batch size per GPU: {GLOBAL_BATCH_SIZE // strategy.num_replicas_in_sync}")
```

### Cell 3: Model Definition (Updated)

```python
def get_news_model(params):
    """Build the multimodal fake news detection model."""
    # Note: clear_session should be called OUTSIDE strategy.scope()
    # tf.keras.backend.clear_session()
    
    # BERT encoder function
    def bert_encode(input_ids, input_mask, segment_ids):
        bert_layer = hub.KerasLayer(
            bert_path,
            trainable=False,
            signature="tokens",
            signature_outputs_as_dict=True,
        )
        bert_inputs = {
            "input_ids": input_ids, 
            "input_mask": input_mask, 
            "segment_ids": segment_ids
        }
        bert_outputs = bert_layer(bert_inputs)
        return bert_outputs["pooled_output"]

    # Text input branch
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids", dtype=tf.int32)
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks", dtype=tf.int32)
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids", dtype=tf.int32)
    
    bert_output = tf.keras.layers.Lambda(
        lambda inputs: bert_encode(inputs[0], inputs[1], inputs[2]),
        output_shape=(768,),
        name="bert_encoding"
    )([in_id, in_mask, in_segment])

    if params['text_no_hidden_layer'] > 0:
        for i in range(params['text_no_hidden_layer']):
            bert_output = tf.keras.layers.Dense(params['text_hidden_neurons'], activation='relu')(bert_output)
            bert_output = tf.keras.layers.Dropout(params['dropout'])(bert_output)

    text_repr = tf.keras.layers.Dense(params['repr_size'], activation='relu')(bert_output)

    # Image input branch (VGG19)
    conv_base = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    conv_base.trainable = False

    input_image = tf.keras.layers.Input(shape=(3, 224, 224))
    transposed_image = tf.keras.layers.Lambda(lambda x: tf.transpose(x, [0, 2, 3, 1]))(input_image)
    base_output = conv_base(transposed_image)
    flat = tf.keras.layers.Flatten()(base_output)

    if params['vis_no_hidden_layer'] > 0:
        for i in range(params['vis_no_hidden_layer']):
            flat = tf.keras.layers.Dense(params['vis_hidden_neurons'], activation='relu')(flat)
            flat = tf.keras.layers.Dropout(params['dropout'])(flat)

    visual_repr = tf.keras.layers.Dense(params['repr_size'], activation='relu')(flat)

    # Classifier (combine text + image)
    combine_repr = tf.keras.layers.concatenate([text_repr, visual_repr])
    com_drop = tf.keras.layers.Dropout(params['dropout'])(combine_repr)

    if params['final_no_hidden_layer'] > 0:
        for i in range(params['final_no_hidden_layer']):
            com_drop = tf.keras.layers.Dense(params['final_hidden_neurons'], activation='relu')(com_drop)
            com_drop = tf.keras.layers.Dropout(params['dropout'])(com_drop)

    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(com_drop)

    model = tf.keras.models.Model(inputs=[in_id, in_mask, in_segment, input_image], outputs=prediction)
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'](), metrics=['accuracy'])
    
    return model

print("✓ Model definition ready (multi-GPU compatible)")
```

### Cell 4: Build Model with Multi-GPU

```python
# Hyperparameters
params_final = {
    'text_no_hidden_layer': 1,
    'text_hidden_neurons': 768,
    'dropout': 0.4,
    'repr_size': 32,
    'vis_no_hidden_layer': 1,
    'vis_hidden_neurons': 2742,
    'final_no_hidden_layer': 1,
    'final_hidden_neurons': 35,
    'optimizer': tf.keras.optimizers.Adam
}

# Clear any previous models
tf.keras.backend.clear_session()
gc.collect()

# Build model inside strategy scope (CRITICAL for multi-GPU)
print("Building model on multi-GPU...")
with strategy.scope():
    model = get_news_model(params_final)
    model.optimizer.learning_rate.assign(0.0005)

print("\n✓ Model created successfully on both GPUs")
print(f"Model will use {strategy.num_replicas_in_sync} GPUs for training")

# Display model summary
model.summary()
```

### Cell 5: Train with Multi-GPU

```python
# Setup callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Optional: TensorBoard for monitoring
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    profile_batch='500,520'
)

print(f"{'='*60}")
print("TRAINING CONFIGURATION")
print(f"{'='*60}")
print(f"GPUs in use: {strategy.num_replicas_in_sync}")
print(f"Global batch size: {GLOBAL_BATCH_SIZE}")
print(f"Batch size per GPU: {GLOBAL_BATCH_SIZE // strategy.num_replicas_in_sync}")
print(f"Training samples: {len(train_input_ids)}")
print(f"Validation samples: {len(test_input_ids)}")
print(f"Steps per epoch: {len(train_input_ids) // GLOBAL_BATCH_SIZE}")
print(f"{'='*60}\n")

# Train the model
print("Starting training...\n")

history = model.fit(
    [train_input_ids, train_input_masks, train_segment_ids, train_imagesX],
    trainY_processed,
    batch_size=GLOBAL_BATCH_SIZE,  # 512 total (256 per GPU)
    epochs=20,
    verbose=1,
    shuffle=True,
    validation_data=(
        [test_input_ids, test_input_masks, test_segment_ids, test_imagesX],
        testY_processed
    ),
    callbacks=[early_stop]  # Add tensorboard_callback if needed
)

print("\n✓ Training completed!")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
```

## Performance Tips for 2x T4 GPUs

### 1. Batch Size Optimization

```python
# Recommended batch sizes for 2x T4 (15GB each)
# For your model (BERT + VGG19):
BASE_BATCH_SIZE = 256   # Good balance
# Or try:
BASE_BATCH_SIZE = 384   # More aggressive (may need to watch memory)
```

### 2. Memory Management

```python
# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("✓ Mixed precision enabled (2x faster, less memory)")

# Apply in strategy scope:
with strategy.scope():
    model = get_news_model(params_final)
```

### 3. Monitor GPU Usage

```python
# Check GPU utilization during training
!nvidia-smi --loop=2

# Or programmatically:
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

### 4. Data Pipeline Optimization (Advanced)

For faster data loading with multi-GPU:

```python
def create_tf_dataset(input_ids, masks, segments, images, labels, batch_size):
    """Create optimized TF dataset for multi-GPU"""
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'input_masks': masks,
            'segment_ids': segments,
            'images': images
        },
        labels
    ))
    
    # Optimize for multi-GPU
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Use with strategy:
train_dataset = create_tf_dataset(
    train_input_ids, train_input_masks, train_segment_ids,
    train_imagesX, trainY_processed,
    GLOBAL_BATCH_SIZE
)

val_dataset = create_tf_dataset(
    test_input_ids, test_input_masks, test_segment_ids,
    test_imagesX, testY_processed,
    GLOBAL_BATCH_SIZE
)

# Train with datasets
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=[early_stop]
)
```

## Expected Performance Gains

| Configuration | Training Speed | Memory Usage |
|---------------|----------------|--------------|
| **1x T4 GPU** | ~2.5 min/epoch | ~12GB |
| **2x T4 GPU** | ~1.3 min/epoch | ~12GB × 2 |
| **Speedup** | **~1.9x faster** | Parallel processing |

## Troubleshooting

### Issue 1: "Out of Memory" on both GPUs
**Solution**: Reduce batch size
```python
BASE_BATCH_SIZE = 128  # Instead of 256
```

### Issue 2: Only 1 GPU being used
**Solution**: Verify strategy initialization
```python
print(f"Replicas: {strategy.num_replicas_in_sync}")  # Should be 2
```

### Issue 3: Slower than single GPU
**Solution**: 
- Ensure model is built inside `strategy.scope()`
- Use larger batch sizes (256+ per GPU)
- Check if data loading is the bottleneck

### Issue 4: GPU memory imbalance
**Solution**: Use `AUTO` sharding
```python
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
```

## Key Differences: Single GPU vs Multi-GPU

| Aspect | Single GPU | Multi-GPU (2x T4) |
|--------|-----------|-------------------|
| **Strategy** | Default | `MirroredStrategy()` |
| **Model Creation** | Direct | Inside `strategy.scope()` |
| **Batch Size** | 256 | 512 (256 × 2) |
| **Training Speed** | 1x baseline | ~1.9x faster |
| **Code Changes** | Minimal | Wrap in scope |

## Summary Checklist

- ✅ Enable GPU T4 x2 in Kaggle settings
- ✅ Import and initialize `MirroredStrategy`
- ✅ Verify 2 GPUs detected
- ✅ Scale batch size by `num_replicas_in_sync`
- ✅ Build model inside `strategy.scope()`
- ✅ Use `GLOBAL_BATCH_SIZE` in training
- ✅ Enable mixed precision (optional, 2x speedup)
- ✅ Monitor both GPUs with `nvidia-smi`

Happy multi-GPU training! 🚀
