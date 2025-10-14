# SpotFake Training & Prediction Notebook

## Overview

`Train_and_Predict.ipynb` is a clean, organized notebook for training and using the SpotFake fake news detection model. It combines text (BERT) and image (VGG19) features to classify news posts as REAL or FAKE.

## Features

✅ **Complete Pipeline**: Data loading → Preprocessing → Training → Evaluation → Inference  
✅ **Clean Structure**: Well-organized sections with clear explanations  
✅ **Easy Inference**: Simple functions to predict on new text + image pairs  
✅ **Reusable**: Load trained weights without retraining  
✅ **Batch Processing**: Predict on multiple inputs at once

## Quick Start

### 1. Train the Model

Run all cells sequentially to:
- Load and preprocess the Twitter dataset
- Build the multimodal model (BERT + VGG19)
- Train for 20 epochs
- Evaluate on test set
- Save the best model

### 2. Make Predictions

#### Single Prediction

```python
# Your input
my_text = "Breaking news: Major event happening now!"
my_image = "path/to/image.jpg"

# Predict
result = predict_fake_news(my_text, my_image, model, tokenizer)

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### Batch Prediction

```python
texts = ["Text 1", "Text 2", "Text 3"]
images = ["img1.jpg", "img2.jpg", "img3.jpg"]

results = predict_batch(texts, images, model, tokenizer)
```

## Model Architecture

**Text Branch (BERT)**:
- BERT base uncased (768-dim embeddings)
- 1 hidden layer (768 neurons)
- Dropout (0.4)
- Output: 32-dim representation

**Image Branch (VGG19)**:
- VGG19 pretrained on ImageNet
- 1 hidden layer (2742 neurons)
- Dropout (0.4)
- Output: 32-dim representation

**Classifier**:
- Concatenate text + image representations (64-dim)
- 1 hidden layer (35 neurons)
- Dropout (0.4)
- Sigmoid output (binary classification)

## Hyperparameters

```python
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
```

- Learning rate: 0.0005
- Batch size: 256
- Epochs: 20

## File Structure

```
SpotFake02/
├── Train_and_Predict.ipynb       # Main notebook (NEW!)
├── Twitter_SpotFake.ipynb         # Original exploration notebook
├── dataset/
│   └── twitter/
│       ├── train_posts.txt
│       ├── test_posts.txt
│       ├── images_train/
│       └── images_test/
├── train_imagesX.npy              # Preprocessed training images
├── test_imagesX.npy               # Preprocessed test images
└── model-XXX-X.XXXXXX.h5          # Saved model checkpoints
```

## Notebook Sections

1. **Imports and Setup** - Load libraries and configure parameters
2. **Helper Functions** - Text/image preprocessing utilities
3. **Model Definition** - Build the multimodal architecture
4. **Load and Preprocess Data** - Load Twitter dataset
5. **Model Training** - Train with validation
6. **Model Evaluation** - Test set metrics
7. **Inference on New Input** - Predict on custom data
8. **Save/Load Model** - Persist trained models

## Usage Tips

### Skip Training (Use Existing Model)

If you already have a trained model:

```python
# After running sections 1-3 (imports, functions, model definition)
model = get_news_model(params_final)
model.load_weights('model-010-0.776923.h5')  # Your best checkpoint

# Jump to section 7 for inference
```

### Preprocessed Images

Images are saved as `.npy` files after first processing. This saves time on subsequent runs:
- `train_imagesX.npy` (~1.5GB)
- `test_imagesX.npy` (~500MB)

### Custom Dataset

To use your own data:

1. Prepare DataFrame with columns: `post_text`, `image_id`, `label`
2. Update the data loading section (Section 4)
3. Ensure images are in the correct folder structure

## Performance

Expected results on Twitter dataset:
- **Accuracy**: ~77-78%
- **Training time**: ~2-3 hours on GPU (20 epochs)
- **Inference time**: ~0.5-1 second per sample

## Requirements

```
tensorflow>=2.x
tensorflow-hub
transformers
opencv-python
pandas
numpy
tqdm
matplotlib
scikit-learn
```

Install with:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Out of Memory
- Reduce batch size from 256 to 128 or 64
- Process images in smaller batches

### Missing Images
- The notebook automatically filters out missing images
- Check `images_train_not_available` and `images_test_not_available` sets

### BERT Loading Issues
- Ensure internet connection for first run (downloads BERT weights)
- Cached in `~/.cache/huggingface/` and `~/.keras/`

## Key Differences from Original Notebook

✅ **Organized**: Clear sections vs. scattered cells  
✅ **Focused**: Training + inference only (no hyperparameter search)  
✅ **Documented**: Comments and markdown explanations  
✅ **Inference-ready**: Easy-to-use prediction functions  
✅ **Production-ready**: Save/load functionality

## Next Steps

1. **Run the notebook** - Execute all cells to train
2. **Test predictions** - Try the inference examples
3. **Customize** - Adapt for your use case
4. **Deploy** - Export model for production

## Credits

Based on the SpotFake multimodal fake news detection approach combining BERT text embeddings with VGG19 image features.
