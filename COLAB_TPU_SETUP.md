# Google Colab TPU Setup Guide

## Quick Setup for Google Colab with TPU

### 1. Enable TPU Runtime

1. In Google Colab, go to **Runtime** → **Change runtime type**
2. Select **Hardware accelerator**: `TPU`
3. Click **Save**

### 2. Install Required Packages

```python
# In the first cell of your Colab notebook
!pip install -q -r requirements_colab.txt
```

Or install individually:
```python
!pip install -q tensorflow>=2.15.0 tensorflow-hub>=0.16.0 transformers>=4.35.0
!pip install -q opencv-python scikit-learn seaborn
```

### 3. Mount Google Drive (Optional - for dataset storage)

```python
from google.colab import drive
drive.mount('/content/drive')

# Set your working directory
import os
os.chdir('/content/drive/MyDrive/SpotFake02')
```

### 4. Upload Dataset

**Option A: From Google Drive**
```python
# After mounting drive
!ls /content/drive/MyDrive/SpotFake02/dataset/
```

**Option B: Upload directly to Colab**
```python
from google.colab import files
uploaded = files.upload()  # Upload files one by one
```

**Option C: Download from URL**
```python
!wget https://your-dataset-url.com/dataset.zip
!unzip dataset.zip
```

### 5. Initialize TPU

Add this at the beginning of your training notebook:

```python
import tensorflow as tf

# TPU initialization
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("✓ TPU initialized successfully!")
    print(f"Number of TPU cores: {strategy.num_replicas_in_sync}")
except ValueError:
    # Fallback to GPU/CPU
    strategy = tf.distribute.get_strategy()
    print("✗ TPU not available, using default strategy")
```

### 6. Model Training with TPU

Wrap your model creation inside the strategy scope:

```python
with strategy.scope():
    # Build model
    model = get_news_model(params_final)
    
    # Set learning rate
    model.optimizer.learning_rate.assign(0.0005)
    
    print("✓ Model built on TPU")

# Training (outside strategy scope)
history = model.fit(
    [train_input_ids, train_input_masks, train_segment_ids, train_imagesX],
    trainY_processed,
    batch_size=256 * strategy.num_replicas_in_sync,  # Scale batch size
    epochs=20,
    validation_data=(
        [test_input_ids, test_input_masks, test_segment_ids, test_imagesX],
        testY_processed
    ),
    callbacks=[early_stop]
)
```

## Important Notes for TPU

### ✅ Do's

1. **Scale batch size**: Multiply your batch size by `strategy.num_replicas_in_sync` (usually 8 for TPU v2/v3)
2. **Build model in strategy scope**: Always wrap model creation with `strategy.scope()`
3. **Use TensorFlow data pipeline**: For large datasets, use `tf.data.Dataset` for better performance
4. **Save to Google Drive**: Save models to Drive to persist after session ends

### ❌ Don'ts

1. **Don't use Keras ImageDataGenerator**: TPUs work better with `tf.data`
2. **Don't use small batch sizes**: TPUs are optimized for large batches (256+)
3. **Don't load all data at once**: Use data pipelines for memory efficiency
4. **Avoid Python loops in model**: Use TensorFlow operations for TPU optimization

## Optimal Batch Sizes for TPU

```python
# TPU v2/v3 (8 cores)
batch_size = 256 * 8  # = 2048

# For your model (if memory allows)
batch_size_per_replica = 128  # Adjust based on model size
total_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
```

## Data Pipeline for TPU (Recommended)

For better performance with large datasets:

```python
def create_tf_dataset(input_ids, input_masks, segment_ids, images, labels, batch_size):
    """Create TensorFlow dataset for TPU"""
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'input_masks': input_masks,
            'segment_ids': segment_ids,
            'images': images
        },
        labels
    ))
    
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Create datasets
train_dataset = create_tf_dataset(
    train_input_ids, train_input_masks, train_segment_ids, 
    train_imagesX, trainY_processed,
    batch_size=256 * strategy.num_replicas_in_sync
)

val_dataset = create_tf_dataset(
    test_input_ids, test_input_masks, test_segment_ids,
    test_imagesX, testY_processed,
    batch_size=256 * strategy.num_replicas_in_sync
)

# Train with datasets
with strategy.scope():
    model = get_news_model(params_final)
    
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=[early_stop]
)
```

## Saving Models in Colab

```python
# Save to Google Drive (persists after session)
model.save_weights('/content/drive/MyDrive/SpotFake02/model_weights_final.weights.h5')

# Or download directly
from google.colab import files
model.save_weights('model_weights_final.weights.h5')
files.download('model_weights_final.weights.h5')
```

## Troubleshooting

### Issue: "Failed to initialize TPU system"
**Solution**: Restart runtime and try again. Sometimes TPU needs a fresh start.

### Issue: "Out of memory"
**Solution**: 
- Reduce batch size
- Use gradient accumulation
- Clear memory: `tf.keras.backend.clear_session()`

### Issue: "Slow training"
**Solution**:
- Use `tf.data.Dataset` instead of numpy arrays
- Enable prefetching: `dataset.prefetch(tf.data.AUTOTUNE)`
- Increase batch size

## Performance Tips

1. **Monitor TPU usage**: Check Cloud Console to ensure TPU is being utilized
2. **Profile training**: Use TensorBoard profiler to identify bottlenecks
3. **Mixed precision**: Enable for faster training:
   ```python
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy('mixed_bfloat16')
   ```

## Cost & Limits

- **Free tier**: Limited TPU hours per month in Colab
- **Colab Pro**: More TPU hours and better availability
- **Session timeout**: 12 hours (free), 24 hours (Pro)
- **Idle timeout**: 90 minutes (free), 24 hours (Pro)

## Example: Complete Colab Notebook Structure

```python
# Cell 1: Setup
!pip install -q -r requirements_colab.txt

# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: TPU Initialization
import tensorflow as tf
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

# Cell 4: Load Data
# ... (data loading code)

# Cell 5: Build Model with TPU
with strategy.scope():
    model = get_news_model(params_final)

# Cell 6: Train
history = model.fit(...)

# Cell 7: Save
model.save_weights('/content/drive/MyDrive/model_weights.h5')
```

Happy training on TPU! 🚀
