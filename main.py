import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from typing import Optional, List, Dict
import logging
from datetime import datetime
import uuid
from google import genai
from google.genai import types

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Suppress warnings before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suppress transformers warnings
logging.getLogger('transformers').setLevel(logging.ERROR)

# Suppress specific warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow_hub as hub
from transformers import BertTokenizer
import shap

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

app = FastAPI(title="SpotFake API", description="Fake News Detection with Explainability", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model and utilities
model = None
tokenizer = None
gradcam = None
shap_explainer = None

# Model configuration
MODEL_CONFIG = {
    'max_seq_length': 23,
    'bert_path': "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
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

WEIGHTS_FILE = 'spotfake_resnet50_multi_gpu_final.weights.h5'
IMG_SIZE = 224

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Set via environment variable
gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✓ Gemini API configured")
    except Exception as e:
        print(f"⚠ Gemini API configuration error: {e}")
        gemini_client = None
else:
    print("⚠ Gemini API key not set (set GEMINI_API_KEY environment variable)")


# ========== HELPER CLASSES AND FUNCTIONS ==========

class InputExample:
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def convert_single_example(tokenizer, example, max_seq_length=23):
    """Tokenize a single text example"""
    encoding = tokenizer(
        example.text_a,
        truncation=True,
        padding='max_length',
        max_length=max_seq_length,
        return_tensors='tf'
    )
    input_ids = encoding['input_ids'][0].numpy().tolist()
    input_mask = encoding['attention_mask'][0].numpy().tolist()
    segment_ids = [0] * max_seq_length
    return input_ids, input_mask, segment_ids


def process_uploaded_image(image_file: bytes) -> np.ndarray:
    """Process uploaded image file to model input format"""
    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(image_file))
        img = img.convert('RGB')
        
        # Resize to 224x224
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Transpose to NCHW format (channels first)
        img_array = img_array.transpose(2, 0, 1)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


def get_news_model_resnet50(params):
    """Build multimodal model (BERT + ResNet50)"""
    max_seq_length = params['max_seq_length']
    bert_path = params['bert_path']
    
    # BERT encoder
    def bert_encode(input_ids, input_mask, segment_ids):
        bert_layer = hub.KerasLayer(
            bert_path,
            trainable=False,
            signature="tokens",
            signature_outputs_as_dict=True,
        )
        bert_outputs = bert_layer({
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids
        })
        return bert_outputs["pooled_output"]
    
    # Text branch
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids", dtype=tf.int32)
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks", dtype=tf.int32)
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids", dtype=tf.int32)
    
    bert_output = tf.keras.layers.Lambda(
        lambda inputs: bert_encode(inputs[0], inputs[1], inputs[2]),
        output_shape=(768,),
        name="bert_encoding"
    )([in_id, in_mask, in_segment])
    
    # Text hidden layers
    for i in range(params['text_no_hidden_layer']):
        bert_output = tf.keras.layers.Dense(params['text_hidden_neurons'], activation='relu')(bert_output)
        bert_output = tf.keras.layers.Dropout(params['dropout'])(bert_output)
    
    text_repr = tf.keras.layers.Dense(params['repr_size'], activation='relu')(bert_output)
    
    # Image branch - ResNet50
    conv_base = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    conv_base.trainable = False
    
    input_image = tf.keras.layers.Input(shape=(3, 224, 224))
    transposed = tf.keras.layers.Lambda(lambda x: tf.transpose(x, [0, 2, 3, 1]))(input_image)
    base_output = conv_base(transposed)
    
    flat = base_output
    
    # Visual hidden layers
    for i in range(params['vis_no_hidden_layer']):
        flat = tf.keras.layers.Dense(params['vis_hidden_neurons'], activation='relu')(flat)
        flat = tf.keras.layers.Dropout(params['dropout'])(flat)
    
    visual_repr = tf.keras.layers.Dense(params['repr_size'], activation='relu')(flat)
    
    # Classifier
    combine = tf.keras.layers.concatenate([text_repr, visual_repr])
    com_drop = tf.keras.layers.Dropout(params['dropout'])(combine)
    
    for i in range(params['final_no_hidden_layer']):
        com_drop = tf.keras.layers.Dense(params['final_hidden_neurons'], activation='relu')(com_drop)
        com_drop = tf.keras.layers.Dropout(params['dropout'])(com_drop)
    
    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(com_drop)
    
    model = tf.keras.models.Model(inputs=[in_id, in_mask, in_segment, input_image], outputs=prediction)
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'](), metrics=['accuracy'])
    
    return model


class GradCAM:
    """Grad-CAM for ResNet50-based multimodal model"""
    
    def __init__(self, model, target_layer_name='conv5_block3_out'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.conv_base = None
        self._find_conv_base()
    
    def _find_conv_base(self):
        """Find ResNet50 conv_base in the model"""
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):
                layer_names = [l.name for l in layer.layers]
                if any('conv5' in name for name in layer_names):
                    try:
                        layer.get_layer(self.target_layer_name)
                        self.conv_base = layer
                        print(f"✓ Found conv_base with target layer: {self.target_layer_name}")
                        return
                    except:
                        pass
        
        # Rebuild ResNet50 without pooling
        print(f"⚠ Rebuilding ResNet50 without pooling...")
        self.conv_base = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling=None
        )
        print(f"✓ Conv_base ready")
    
    def compute_heatmap(self, input_ids, input_mask, segment_ids, image, eps=1e-8):
        """Compute Grad-CAM heatmap"""
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        if len(input_ids.shape) == 1:
            input_ids = np.expand_dims(input_ids, axis=0)
            input_mask = np.expand_dims(input_mask, axis=0)
            segment_ids = np.expand_dims(segment_ids, axis=0)
        
        # Convert to tensors
        input_ids_tensor = tf.cast(input_ids, tf.int32)
        input_mask_tensor = tf.cast(input_mask, tf.int32)
        segment_ids_tensor = tf.cast(segment_ids, tf.int32)
        image_tensor = tf.cast(image, tf.float32)
        
        # Transpose image (NCHW → NHWC)
        transposed_image = tf.transpose(image_tensor, [0, 2, 3, 1])
        
        # Build gradient model
        target_layer = self.conv_base.get_layer(self.target_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=self.conv_base.input,
            outputs=[target_layer.output, self.conv_base.output]
        )
        
        # Compute with gradient tape
        with tf.GradientTape() as tape:
            tape.watch(transposed_image)
            conv_outputs_target, _ = grad_model(transposed_image)
            
            # Get final prediction
            predictions = self.model([
                input_ids_tensor,
                input_mask_tensor,
                segment_ids_tensor,
                image_tensor
            ])
            class_output = predictions[:, 0]
        
        # Compute gradients
        grads = tape.gradient(class_output, conv_outputs_target)
        
        if grads is None:
            # Fallback: activation-based CAM
            conv_outputs_np = conv_outputs_target.numpy()[0]
            heatmap = np.mean(conv_outputs_np, axis=-1)
        else:
            # True Grad-CAM
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs_np = conv_outputs_target.numpy()[0]
            pooled_grads_np = pooled_grads.numpy()
            heatmap = np.sum(conv_outputs_np * pooled_grads_np, axis=-1)
        
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap_max = heatmap.max()
        if heatmap_max > eps:
            heatmap = heatmap / heatmap_max
        
        # Resize to 224x224
        heatmap = cv2.resize(heatmap, (224, 224))
        return heatmap
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """Overlay heatmap on original image and return as base64"""
        # Convert to HWC if needed
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Normalize image to [0, 255]
        img = image.copy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Apply colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Convert to base64
        pil_img = Image.fromarray(overlayed)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64


class SHAPTextExplainer:
    """SHAP-based text explainer for multimodal model"""
    
    def __init__(self, model, tokenizer, max_seq_length=23):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def predict_text_only(self, texts, background_image):
        """Prediction function for SHAP"""
        # Tokenize texts
        examples = [InputExample(guid=None, text_a=text, label=0) for text in texts]
        
        input_ids_list = []
        input_masks_list = []
        segment_ids_list = []
        
        for example in examples:
            input_id, input_mask, segment_id = convert_single_example(
                self.tokenizer, example, self.max_seq_length
            )
            input_ids_list.append(input_id)
            input_masks_list.append(input_mask)
            segment_ids_list.append(segment_id)
        
        input_ids = np.array(input_ids_list)
        input_masks = np.array(input_masks_list)
        segment_ids = np.array(segment_ids_list)
        
        # Repeat background image
        images = np.repeat(np.expand_dims(background_image, 0), len(texts), axis=0)
        
        # Predict
        preds = self.model.predict(
            [input_ids, input_masks, segment_ids, images],
            verbose=0
        )
        
        return preds.flatten()
    
    def explain(self, text, image, num_samples=25):
        """Explain prediction using SHAP and return top tokens"""
        try:
            # Create prediction function
            def predict_fn(texts):
                if isinstance(texts, str):
                    texts = [texts]
                return self.predict_text_only(texts, image)
            
            # Create SHAP explainer
            explainer = shap.Explainer(
                predict_fn,
                masker=shap.maskers.Text(),
                algorithm='auto'
            )
            
            # Compute SHAP values
            shap_values = explainer([text], max_evals=num_samples)
            
            print(f"SHAP values type: {type(shap_values)}")
            print(f"SHAP values shape: {shap_values.values.shape if hasattr(shap_values, 'values') else 'N/A'}")
            
            # Extract token importance from SHAP
            # Get the actual tokens from the original text
            tokens = text.split()
            
            # Get SHAP values - they are returned as array
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0]  # First sample
                
                # Handle case where SHAP returns values for the whole text
                if isinstance(values, (np.ndarray, list)):
                    # If values length matches tokens, use directly
                    if len(values) >= len(tokens):
                        token_importance = []
                        for i, token in enumerate(tokens):
                            if i < len(values):
                                token_importance.append({
                                    "token": token,
                                    "importance": float(values[i])
                                })
                    else:
                        # If fewer values than tokens, distribute evenly
                        token_importance = []
                        for i, token in enumerate(tokens[:len(values)]):
                            token_importance.append({
                                "token": token,
                                "importance": float(values[i])
                            })
                else:
                    # Single value case
                    token_importance = [{
                        "token": tokens[0] if tokens else "text",
                        "importance": float(values)
                    }]
                
                # Sort by absolute importance and get top 10
                token_importance.sort(key=lambda x: abs(x['importance']), reverse=True)
                
                print(f"Extracted {len(token_importance)} tokens with SHAP values")
                print(f"Top 3 tokens: {token_importance[:3]}")
                
                return token_importance[:10]
            else:
                print("No values attribute in SHAP output")
                return []
                
        except Exception as e:
            print(f"SHAP Error: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty list if SHAP fails
            return []


# ========== GEMINI INTEGRATION ==========

def analyze_with_gemini(text: str, image_bytes: bytes) -> Dict[str, any]:
    """
    Use Gemini API to analyze news authenticity
    Returns analysis with reasoning points
    """
    if not gemini_client:
        return {
            "analysis": "Advanced AI analysis unavailable",
            "reasoning": ["Deep learning model analysis active"],
            "confidence_explanation": "Based on multimodal neural network patterns"
        }
    
    try:
        # Create image part from bytes (matching working testing.py approach)
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        )
        
        # Create prompt for news authenticity analysis
        prompt = f"""Analyze this news post for authenticity. The post contains an image and the following text:

Text: "{text}"

Provide a detailed analysis in the following JSON format:
{{
    "verdict": "REAL" or "FAKE",
    "confidence": 0.0 to 1.0,
    "reasoning": [
        "Point 1: Brief reason",
        "Point 2: Brief reason",
        "Point 3: Brief reason"
    ],
    "key_indicators": [
        "Specific element that indicates authenticity/fakeness"
    ],
    "summary": "One sentence summary of why this is real or fake"
}}

Focus on:
- Visual inconsistencies in the image
- Text credibility and language patterns
- Coherence between image and text
- Common fake news indicators

Be concise and specific. Provide 3-5 reasoning points."""

        # Generate analysis using stable Gemini model (matching testing.py)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                image_part
            ]
        )
        
        # Parse response
        response_text = response.text.strip()
        
        # Try to extract JSON from response
        import json
        import re
        
        # Find JSON in response (may be wrapped in markdown code blocks)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
            return {
                "analysis": analysis.get("summary", "Analysis complete"),
                "reasoning": analysis.get("reasoning", []),
                "key_indicators": analysis.get("key_indicators", []),
                "gemini_verdict": analysis.get("verdict", "UNKNOWN"),
                "gemini_confidence": analysis.get("confidence", 0.5)
            }
        else:
            # Fallback: parse unstructured response
            return {
                "analysis": response_text[:200],
                "reasoning": [response_text],
                "key_indicators": [],
                "gemini_verdict": "UNKNOWN",
                "gemini_confidence": 0.5
            }
            
    except Exception as e:
        error_msg = str(e)
        
        # User-friendly error messages for different scenarios
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            print("\n" + "="*70)
            print("⚠️  GEMINI API RATE LIMIT REACHED")
            print("="*70)
            print("The free tier has limited requests. Your ML model prediction")
            print("will still work - only the enhanced AI reasoning is skipped.")
            print("\nOptions:")
            print("  1. Wait ~1 minute and try again")
            print("  2. Use fewer requests per minute")
            print("  3. Upgrade to paid tier for higher limits")
            print("="*70 + "\n")
        elif "quota" in error_msg.lower():
            print("\n⚠️  Gemini API quota exceeded. Continuing with ML model only.")
        else:
            print(f"\n⚠️  Gemini API error: {error_msg[:150]}")
        
        # Return graceful fallback - prediction continues without Gemini
        return {
            "analysis": "ML model prediction complete (Enhanced AI reasoning unavailable)",
            "reasoning": ["Multimodal deep learning analysis active"],
            "key_indicators": [],
            "gemini_verdict": None,
            "gemini_confidence": None
        }


# ========== STARTUP EVENT ==========

@app.on_event("startup")
async def load_model():
    """Load model and initialize utilities on startup"""
    global model, tokenizer, gradcam, shap_explainer
    
    print("=" * 70)
    print("LOADING SPOTFAKE MODEL...")
    print("=" * 70)
    
    # Check if weights file exists
    if not os.path.exists(WEIGHTS_FILE):
        print(f"❌ ERROR: Weights file not found: {WEIGHTS_FILE}")
        print("Please ensure the trained model weights are in the current directory.")
        return
    
    # Build model
    print("\n1. Building model architecture...")
    model = get_news_model_resnet50(MODEL_CONFIG)
    
    # Load weights
    print(f"2. Loading weights from {WEIGHTS_FILE}...")
    try:
        model.load_weights(WEIGHTS_FILE)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading weights: {str(e)}")
        return
    
    # Initialize tokenizer
    print("\n3. Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("✓ Tokenizer loaded")
    
    # Initialize GradCAM
    print("\n4. Initializing GradCAM...")
    gradcam = GradCAM(model, target_layer_name='conv5_block3_out')
    
    # Initialize SHAP explainer
    print("\n5. Initializing SHAP explainer...")
    shap_explainer = SHAPTextExplainer(model, tokenizer, MODEL_CONFIG['max_seq_length'])
    print("✓ SHAP explainer ready")
    
    print("\n" + "=" * 70)
    print("✓ MODEL READY FOR PREDICTIONS")
    print("=" * 70)


# ========== API ENDPOINTS ==========

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.get("/api")
async def api_info():
    return {
        "message": "SpotFake API - Fake News Detection with Explainability",
        "version": "1.0",
        "endpoints": {
            "predict": "POST /predict - Upload image and text for prediction",
            "health": "GET /health - Check API health",
            "image": "GET /image/{filename} - Retrieve saved image",
            "cleanup": "DELETE /cleanup - Clean old temp files (>1 hour)"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Check if model is loaded and ready"""
    if model is None or tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded"}
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "weights_file": WEIGHTS_FILE
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Image file (JPEG/PNG)"),
    text: str = Form(..., description="Text content to analyze"),
    include_gradcam: bool = Query(False, description="Include Grad-CAM visualization"),
    include_shap: bool = Query(False, description="Include SHAP text explanation (slower)")
):
    """
    Predict if news is REAL or FAKE with optional explainability
    
    Parameters:
    - image: Image file associated with the news (upload via file picker)
    - text: Text content of the news
    - include_gradcam: Whether to include Grad-CAM visualization (default: False)
    - include_shap: Whether to include SHAP explanation (default: False, adds 20-40s)
    
    Returns:
    - verdict: REAL or FAKE
    - confidence: Prediction confidence score (0-1)
    - gradcam_image: Base64-encoded Grad-CAM overlay (if enabled)
    - shap_explanation: Top important tokens with scores (if enabled)
    """
    
    # Check if model is loaded
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # 1. Process image
        image_bytes = await image.read()
        
        # Save uploaded image to temp folder with unique name
        os.makedirs("temp", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_ext = os.path.splitext(image.filename)[1] if image.filename else ".jpg"
        temp_filename = f"temp/{timestamp}_{unique_id}{file_ext}"
        
        with open(temp_filename, "wb") as f:
            f.write(image_bytes)
        
        print(f"Saved uploaded image to: {temp_filename}")
        
        img_array = process_uploaded_image(image_bytes)
        
        # 2. Process text
        example = InputExample(guid=None, text_a=text, label=0)
        input_ids, input_mask, segment_ids = convert_single_example(
            tokenizer, example, MODEL_CONFIG['max_seq_length']
        )
        
        # Convert to numpy arrays
        input_ids = np.array([input_ids])
        input_mask = np.array([input_mask])
        segment_ids = np.array([segment_ids])
        img_array = np.expand_dims(img_array, axis=0)
        
        # 3. Make prediction
        prediction = model.predict(
            [input_ids, input_mask, segment_ids, img_array],
            verbose=0
        )[0, 0]
        
        verdict = "REAL" if prediction >= 0.5 else "FAKE"
        confidence = float(prediction) if prediction >= 0.5 else float(1 - prediction)
        
        # 3.5. Get Gemini analysis (runs in parallel with other operations)
        print("Getting AI-enhanced analysis...")
        gemini_analysis = analyze_with_gemini(text, image_bytes)
        
        # 4. Build response
        response = {
            "verdict": verdict,
            "confidence": confidence,
            "raw_score": float(prediction),
            "text": text[:100] + "..." if len(text) > 100 else text,
            "saved_image": temp_filename,
            "analysis": gemini_analysis.get("analysis", "Analysis complete"),
            "reasoning": gemini_analysis.get("reasoning", []),
            "key_indicators": gemini_analysis.get("key_indicators", [])
        }
        
        # 5. Generate Grad-CAM if requested
        if include_gradcam:
            print("Generating Grad-CAM...")
            heatmap = gradcam.compute_heatmap(
                input_ids[0], input_mask[0], segment_ids[0], img_array[0]
            )
            gradcam_base64 = gradcam.overlay_heatmap(img_array[0], heatmap)
            response["gradcam_image"] = gradcam_base64
            response["gradcam_note"] = "Base64-encoded PNG image"
        
        # 6. Generate SHAP explanation if requested
        if include_shap:
            print("Generating SHAP explanation (this may take 20-40 seconds)...")
            token_importance = shap_explainer.explain(text, img_array[0], num_samples=25)
            response["shap_explanation"] = token_importance
            response["shap_note"] = "Top 10 influential tokens with importance scores"
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/image/{filename}")
async def get_saved_image(filename: str):
    """
    Retrieve a saved image from the temp folder
    
    Parameters:
    - filename: Name of the saved image file
    """
    file_path = f"temp/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(file_path)


@app.delete("/cleanup")
async def cleanup_temp_folder():
    """
    Clean up old images from temp folder (older than 1 hour)
    """
    try:
        if not os.path.exists("temp"):
            return {"message": "Temp folder does not exist"}
        
        deleted_count = 0
        current_time = datetime.now()
        
        for filename in os.listdir("temp"):
            file_path = os.path.join("temp", filename)
            if os.path.isfile(file_path):
                # Check file age
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                if age_hours > 1:
                    os.remove(file_path)
                    deleted_count += 1
        
        return {
            "message": f"Cleanup complete",
            "deleted_files": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")