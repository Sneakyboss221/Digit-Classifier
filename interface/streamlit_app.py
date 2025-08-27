"""
Streamlit web application for MNIST digit recognition
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

# Import our modules
from data.data_loader import MNISTDataLoader, Preprocessor
from models.mlp_model import create_mlp_model, predict_mlp
from models.lenet5_model import create_lenet5_model, predict_lenet5
from models.resnet_model import create_resnet_model, predict_resnet
from models.ensemble_model import create_ensemble_model
from utils.visualization import Visualizer
from config import TRAINING_CONFIG, PATHS

# Set page config
st.set_page_config(
    page_title="Ultimate MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models"""
    device = TRAINING_CONFIG['device']
    models = {}
    
    try:
        # Load MLP model
        mlp_model = create_mlp_model()
        mlp_checkpoint = torch.load(f"{PATHS['saved_models']}/mlp_model.pth", map_location=device)
        mlp_model.load_state_dict(mlp_checkpoint['model_state_dict'])
        models['mlp'] = mlp_model.to(device)
        st.success("✅ MLP model loaded successfully")
    except Exception as e:
        st.warning(f"⚠️ Could not load MLP model: {str(e)}")
        models['mlp'] = None
    
    try:
        # Load LeNet-5 model
        lenet5_model = create_lenet5_model()
        lenet5_checkpoint = torch.load(f"{PATHS['saved_models']}/lenet5_model.pth", map_location=device)
        lenet5_model.load_state_dict(lenet5_checkpoint['model_state_dict'])
        models['lenet5'] = lenet5_model.to(device)
        st.success("✅ LeNet-5 model loaded successfully")
    except Exception as e:
        st.warning(f"⚠️ Could not load LeNet-5 model: {str(e)}")
        models['lenet5'] = None
    
    try:
        # Load ResNet model
        resnet_model = create_resnet_model()
        resnet_checkpoint = torch.load(f"{PATHS['saved_models']}/resnet_model.pth", map_location=device)
        resnet_model.load_state_dict(resnet_checkpoint['model_state_dict'])
        models['resnet'] = resnet_model.to(device)
        st.success("✅ ResNet model loaded successfully")
    except Exception as e:
        st.warning(f"⚠️ Could not load ResNet model: {str(e)}")
        models['resnet'] = None
    
    return models

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔢 Ultimate MNIST Digit Recognition</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    model_options = ["MLP", "LeNet-5", "ResNet", "Ensemble"]
    selected_model = st.sidebar.selectbox("Choose a model:", model_options)
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎨 Draw Your Digit")
        
        # Drawing canvas
        try:
            from streamlit_canvas import st_canvas
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=15,
                stroke_color="black",
                background_color="white",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            # Clear button
            if st.button("🗑️ Clear Canvas"):
                st.rerun()
            
        except ImportError:
            st.error("""
            **streamlit-canvas not installed!**
            
            Please install it with:
            ```bash
            pip install streamlit-canvas
            ```
            """)
            
            # Fallback: file upload
            st.subheader("📁 Upload Image Instead")
            uploaded_file = st.file_uploader("Upload a digit image", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('L')
                st.image(image, caption="Uploaded Image", use_column_width=True)
                canvas_result = type('obj', (object,), {
                    'image_data': np.array(image)
                })
            else:
                canvas_result = None
    
    with col2:
        st.subheader("🤖 Prediction Results")
        
        if canvas_result is not None and canvas_result.image_data is not None:
            # Process the drawing
            image_data = canvas_result.image_data
            
            # Convert to grayscale if needed
            if len(image_data.shape) == 3:
                gray = np.dot(image_data[..., :3], [0.299, 0.587, 0.114])
                drawing_array = gray.astype(np.uint8)
            else:
                drawing_array = image_data.astype(np.uint8)
            
            # Preprocess for prediction
            try:
                preprocessed = Preprocessor.preprocess_drawn_image(drawing_array)
                
                # Make predictions
                device = TRAINING_CONFIG['device']
                predictions = {}
                
                if selected_model == "MLP" and models['mlp'] is not None:
                    pred_class, probs = predict_mlp(models['mlp'], preprocessed, device)
                    predictions['MLP'] = {'class': pred_class, 'probabilities': probs}
                
                elif selected_model == "LeNet-5" and models['lenet5'] is not None:
                    pred_class, probs = predict_lenet5(models['lenet5'], preprocessed, device)
                    predictions['LeNet-5'] = {'class': pred_class, 'probabilities': probs}
                
                elif selected_model == "ResNet" and models['resnet'] is not None:
                    pred_class, probs = predict_resnet(models['resnet'], preprocessed, device)
                    predictions['ResNet'] = {'class': pred_class, 'probabilities': probs}
                
                elif selected_model == "Ensemble":
                    # Create ensemble model
                    ensemble_models = {k: v for k, v in models.items() if v is not None}
                    if ensemble_models:
                        ensemble = create_ensemble_model(
                            ensemble_models.get('mlp'),
                            ensemble_models.get('lenet5'),
                            ensemble_models.get('resnet')
                        )
                        pred_class, ensemble_probs, individual_preds = ensemble.predict(preprocessed, device)
                        predictions['Ensemble'] = {'class': pred_class, 'probabilities': ensemble_probs}
                        
                        # Add individual predictions
                        for model_name, pred in individual_preds.items():
                            predictions[model_name.upper()] = {
                                'class': pred['predicted_class'],
                                'probabilities': pred['probabilities']
                            }
                
                # Display predictions
                if predictions:
                    for model_name, pred in predictions.items():
                        with st.container():
                            st.markdown(f'<div class="model-card">', unsafe_allow_html=True)
                            st.markdown(f"**{model_name} Prediction:**")
                            st.markdown(f"<div class='prediction-box'>", unsafe_allow_html=True)
                            st.markdown(f"**Predicted Digit: {pred['class']}**")
                            st.markdown(f"**Confidence: {pred['probabilities'][pred['class']]:.3f}**")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Probability bar chart
                            fig, ax = plt.subplots(figsize=(8, 4))
                            digits = range(10)
                            bars = ax.bar(digits, pred['probabilities'], color='skyblue')
                            bars[pred['class']].set_color('red')
                            ax.set_xlabel('Digit')
                            ax.set_ylabel('Probability')
                            ax.set_title(f'{model_name} Probabilities')
                            ax.set_xticks(digits)
                            st.pyplot(fig)
                            st.markdown('</div>', unsafe_allow_html=True)
                
                else:
                    st.error("No models available for prediction. Please train the models first.")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        
        else:
            st.info("👆 Draw a digit in the canvas to get predictions!")
    
    # Additional features
    st.markdown("---")
    
    # Model information
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.subheader("📊 Model Performance")
        performance_data = {
            'MLP': '~98.5%',
            'LeNet-5': '~99.2%',
            'ResNet': '~99.4%',
            'Ensemble': '~99.6%'
        }
        
        for model, acc in performance_data.items():
            st.metric(model, acc)
    
    with col4:
        st.subheader("⚡ Model Characteristics")
        characteristics = {
            'MLP': 'Fast, lightweight',
            'LeNet-5': 'Classic CNN',
            'ResNet': 'Advanced, deep',
            'Ensemble': 'Combined predictions'
        }
        
        for model, char in characteristics.items():
            st.write(f"**{model}:** {char}")
    
    with col5:
        st.subheader("🔧 Quick Actions")
        if st.button("📈 View Training Curves"):
            st.info("Training curves will be displayed here")
        
        if st.button("📋 View Confusion Matrix"):
            st.info("Confusion matrix will be displayed here")
        
        if st.button("🎯 Test on Sample Data"):
            st.info("Sample testing will be performed here")

if __name__ == "__main__":
    main()
