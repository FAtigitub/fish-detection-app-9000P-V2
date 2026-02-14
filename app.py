import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import io
import os
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Fish Detection AI - YOLOv8",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, interactive, and attractive design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Hero Section - Enhanced with animation */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
        animation: fadeInDown 0.6s ease-out;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        line-height: 1.2;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
        animation: slideIn 0.7s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .hero-subtitle {
        font-size: 0.9rem;
        font-weight: 400;
        opacity: 0.95;
        margin-top: 0.1rem;
        line-height: 1.3;
        position: relative;
        z-index: 1;
        animation: slideIn 0.8s ease-out;
    }
    
    /* Stats Cards - Enhanced with attractive animations */
    .stats-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 0.4rem 0;
        position: relative;
        overflow: hidden;
        cursor: pointer;
        animation: fadeInUp 0.5s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stats-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .stats-card:hover::before {
        left: 100%;
    }
    
    .stats-card:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1;
        animation: countUp 0.7s ease-out;
    }
    
    @keyframes countUp {
        from {
            opacity: 0;
            transform: scale(0.5);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #6c757d;
        font-weight: 600;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Fish Alert */
    .fish-alert {
        background: linear-gradient(135deg, #ffd89b 0%, #fb923c 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(251, 146, 60, 0.3);
        margin: 0.8rem 0;
        font-weight: 600;
        font-size: 0.9rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.9; }
    }
    
    /* Detection Box */
    .detection-box {
        background: white;
        padding: 0.7rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin: 0.4rem 0;
        transition: all 0.3s ease;
    }
    
    .detection-box:hover {
        box-shadow: 0 3px 12px rgba(0,0,0,0.1);
        transform: translateX(3px);
    }
    
    .detection-title {
        font-weight: 600;
        color: #2d3748;
        font-size: 0.85rem;
    }
    
    .detection-value {
        color: #667eea;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Buttons - Enhanced with Ripple Effect */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 3px 12px rgba(102, 126, 234, 0.35);
        text-transform: none;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px) scale(1);
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.4);
    }
    
    /* File Uploader - Enhanced */
    .uploadedFile {
        border-radius: 8px;
        border: 2px solid #667eea;
        padding: 0.7rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        transition: all 0.3s ease;
        animation: fadeInUp 0.4s ease-out;
    }
    
    .uploadedFile:hover {
        transform: translateY(-2px);
        box-shadow: 0 3px 15px rgba(102, 126, 234, 0.2);
        border-color: #764ba2;
    }
    
    /* Expander - Enhanced */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        font-size: 0.85rem;
        padding: 0.5rem;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.12);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 3px 12px rgba(16, 185, 129, 0.35);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 5px 18px rgba(16, 185, 129, 0.5);
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Progress Bar - Enhanced with Animation */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        background-size: 200% 100%;
        animation: progressShimmer 2s linear infinite;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    @keyframes progressShimmer {
        0% {
            background-position: 200% 0;
        }
        100% {
            background-position: -200% 0;
        }
    }
    
    /* Info Box - Enhanced */
    .info-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #bfdbfe 100%);
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.6rem 0;
        animation: slideInRight 0.4s ease-out;
        transition: all 0.3s ease;
        font-size: 0.85rem;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .info-box:hover {
        transform: translateX(-3px);
        box-shadow: 0 3px 12px rgba(59, 130, 246, 0.25);
    }
    
    /* 2026 Clean Design Standards - Enhanced Interactive */
    .upload-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            border-color: #667eea;
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4);
        }
        50% {
            border-color: #764ba2;
            box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);
        }
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        transform: scale(1.01);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.15);
        animation: none;
    }
    
    /* Minimalist Headings */
    h3 {
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 0.4rem;
        font-size: 1.1rem;
        line-height: 1.3;
    }
    
    /* Success Box - Enhanced */
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 0.7rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 0.4rem 0;
        animation: slideInLeft 0.4s ease-out;
        transition: all 0.3s ease;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .success-box:hover {
        transform: translateX(3px);
        box-shadow: 0 3px 12px rgba(16, 185, 129, 0.2);
    }
    
    /* Image Container - Enhanced */
    .image-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.6rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInScale 0.5s ease-out;
        position: relative;
    }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .image-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0) 0%, rgba(118, 75, 162, 0) 100%);
        transition: background 0.3s ease;
        pointer-events: none;
    }
    
    .image-container:hover {
        transform: scale(1.01);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    .image-container:hover::before {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }
    
    /* Metric Enhancement */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    /* Smooth Scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Alert Boxes Enhancement */
    .stAlert {
        border-radius: 8px;
        animation: slideInLeft 0.4s ease-out;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    .stAlert:hover {
        transform: translateX(3px);
    }
    
    /* Detection Box Styles - Color Coded and Animated */
    .detection-high {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        padding: 0.7rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        animation: bounceIn 0.5s ease-out;
    }
    
    .detection-medium {
        background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%);
        border-left: 4px solid #fb923c;
        padding: 0.7rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        animation: bounceIn 0.5s ease-out;
    }
    
    .detection-low {
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
        border-left: 4px solid #ef4444;
        padding: 0.7rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        animation: bounceIn 0.5s ease-out;
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            transform: scale(1);
        }
    }
    
    /* Stagger Animation Delays for Stats Cards */
    .stats-card:nth-child(1) {
        animation-delay: 0.1s;
    }
    
    .stats-card:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .stats-card:nth-child(3) {
        animation-delay: 0.3s;
    }
    
    /* Loading Spinner Enhancement */
    .stSpinner > div {
        border-color: #667eea !important;
        border-top-color: transparent !important;
    }
    
    /* Floating Animation for Icons */
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    /* Section Dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #667eea 50%, transparent 100%);
        margin: 1rem 0;
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Radio Buttons - Enhanced */
    .stRadio > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 0.6rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    .stRadio > div:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.12);
    }
    
    /* Camera Input */
    [data-testid="stCameraInput"] {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stCameraInput"]:hover {
        box-shadow: 0 3px 15px rgba(102, 126, 234, 0.18);
    }
</style>
""", unsafe_allow_html=True)

# Load model (cached for performance)
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main app
def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🐟 Fish Detection AI</div>
        <div class="hero-subtitle">Advanced Fish Detection System | Powered by YOLOv8 | 9 Species | 99.5% mAP50 Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner('Loading AI model...'):
        model = load_model()
    
    if model is None:
        st.error("❌ Failed to load the model. Please ensure 'best.pt' is in the app directory.")
        return
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## Detection Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Lower values detect more fish but may include false positives. Recommended: 0.25 for balanced results"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Intersection over Union threshold for removing duplicate detections"
        )
        
        img_size = st.selectbox(
            "Image Size",
            options=[320, 480, 640, 800, 1024],
            index=2,
            help="Larger sizes may detect smaller fish but are slower"
        )
        
        max_det = st.number_input(
            "Max Detections",
            min_value=10,
            max_value=1000,
            value=300,
            step=50,
            help="Maximum number of detections per image"
        )
        
        st.markdown("---")
        st.markdown("### Advanced Options")
        
        use_tta = st.checkbox(
            "Test Time Augmentation (TTA)",
            value=True,
            help="✓ Enabled by default - Improves detection accuracy with multiple augmented predictions"
        )
        
        st.markdown("---")
        st.markdown("### Display Options")
        show_labels = st.checkbox("Show Labels", value=True)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        
        st.markdown("---")
        st.markdown("## Model Information")
        
        with st.expander("🔧 Model Details", expanded=False):
            st.markdown("""
            **Architecture:** YOLOv8 Medium  
            **Parameters:** ~25M  
            **Input Size:** 640×640  
            **Classes:** 9 fish species  
            **Framework:** PyTorch + Ultralytics  
            **Inference Speed:** 22.5ms per image  
            """)
        
        with st.expander("📊 Training Statistics", expanded=False):
            st.markdown("""
            **Dataset:** Fish Detection Dataset  
            **Total Images:** 9,000  
            - Training: 6,295 images (6,295 labels)  
            - Validation: 1,796 images (1,796 labels)  
            - Testing: 909 images (909 labels)  
            
            **Training Configuration:**  
            - Epochs: 80  
            - Batch Size: 32  
            - Image Size: 640×640  
            - Hardware: Tesla T4 x2 GPU  
            
            **Speed Performance:**  
            - Preprocess: 0.8ms  
            - Inference: 22.5ms  
            - Postprocess: 0.8ms  
            """)
        
        with st.expander("🎯 Model Performance", expanded=True):
            st.markdown("""
            <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                        padding: 1rem; border-radius: 10px; border-left: 4px solid #10b981;'>
                <h4 style='color: #065f46; margin-top: 0; font-size: 1rem;'>🏆 Excellent Detection Accuracy</h4>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-top: 0.8rem;'>
                    <div>
                        <p style='margin: 0.3rem 0; color: #047857; font-size: 0.85rem;'><strong>mAP50:</strong> <span style='font-size: 1.1rem; color: #059669;'>99.50%</span></p>
                        <p style='margin: 0.3rem 0; color: #047857; font-size: 0.85rem;'><strong>mAP50-95:</strong> <span style='font-size: 1.1rem; color: #059669;'>83.23%</span></p>
                    </div>
                    <div>
                        <p style='margin: 0.3rem 0; color: #047857; font-size: 0.85rem;'><strong>Precision:</strong> <span style='font-size: 1.1rem; color: #059669;'>99.92%</span></p>
                        <p style='margin: 0.3rem 0; color: #047857; font-size: 0.85rem;'><strong>Recall:</strong> <span style='font-size: 1.1rem; color: #059669;'>100.00%</span></p>
                    </div>
                </div>
                <p style='margin-top: 0.8rem; margin-bottom: 0; color: #065f46; font-size: 0.8rem;'>
                    ✓ Target mAP50 >85%: <strong>EXCEEDED</strong> (99.50%)<br>
                    ✓ Target mAP50-95 >70%: <strong>EXCEEDED</strong> (83.23%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("� Per-Species Performance", expanded=False):
            st.markdown("""
            **Individual Species Metrics:**
            
            | Species | Images | Precision | Recall | mAP50 | mAP50-95 |
            |---------|--------|-----------|--------|-------|----------|
            | Gilt-Head Bream | 188 | 99.9% | 100% | 99.5% | 84.8% |
            | Red Sea Bream | 195 | 99.9% | 100% | 99.5% | 86.8% |
            | Striped Red Mullet | 199 | 99.9% | 100% | 99.5% | 78.2% |
            | Black Sea Sprat | 196 | 100% | 100% | 99.5% | 79.7% |
            | House Mackerel | 218 | 99.9% | 100% | 99.5% | 87.4% |
            | Red Mullet | 214 | 99.9% | 100% | 99.5% | 83.1% |
            | Sea Bass | 176 | 99.9% | 100% | 99.5% | 86.7% |
            | Shrimp | 205 | 99.9% | 100% | 99.5% | 76.4% |
            | Trout | 205 | 99.9% | 100% | 99.5% | 85.8% |
            
            **Key Insights:**
            - ✓ All species achieve >99.5% mAP50
            - ✓ Perfect 100% recall across all classes
            - ✓ Consistent >99.9% precision
            - ✓ mAP50-95 ranges from 76.4% to 87.4%
            - 🏆 Best performer: House Mackerel (87.4% mAP50-95)
            """)
        
        with st.expander("�💡 Detection Tips", expanded=False):
            st.markdown("""
            **For better detection results:**
            
            1. **Use clear, well-lit images** - Good lighting is essential
            2. **Optimal confidence threshold:** 0.20-0.30 for balanced results
            3. **Enable TTA** (✓ default) - Improves accuracy by ~2-5%
            4. **Image size:** 640 (default) works well, use 800-1024 for small fish
            5. **Ensure fish are visible** and not heavily overlapping
            6. **Good contrast** between fish and background
            
            **📸 Camera Feature:**
            - Works on **Streamlit Cloud** (deployed app)
            - Works on **HTTPS** connections
            - May not work on local HTTP (use upload instead)
            
            **⚡ Model Status:**
            - ✓ Fully trained on 9,000 images
            - ✓ 99.5% mAP50 accuracy
            - ✓ All 9 species supported
            """)
    
    # Main Content Area - Clean 2026 Design
    st.markdown("---")
    
    # Check model file size and warn if it seems untrained
    if os.path.exists('best.pt'):
        model_size_mb = os.path.getsize('best.pt') / (1024 * 1024)
        if model_size_mb < 10:  # Untrained models are usually smaller
            st.warning("""
            ⚠️ **Model May Not Be Trained**
            
The model file size is small ({:.1f} MB), suggesting it may not be fully trained. 
            
**To train your model:**
1. Upload the training notebook to Kaggle
2. Activate Tesla T4 x2 GPU
3. Update the dataset path
4. Run all cells (~2-3 hours)
5. Download the trained `best.pt` and replace this file
            
**For now:** Try lowering confidence threshold to 0.1-0.15 and increasing image size to 800-1024.
            """.format(model_size_mb))
    
    # Create two columns for input display
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📸 Get Fish Image")
        
        # Input method selector
        input_method = st.radio(
            "Choose input method:",
            options=["📤 Upload Image", "📸 Take Photo"],
            horizontal=True,
            label_visibility="visible"
        )
        
        image = None
        
        if input_method == "📤 Upload Image":
            st.markdown("<p style='color: #6c757d; margin-bottom: 1rem; font-size: 0.85rem;'>Drag and drop or click to upload JPG, JPEG, or PNG images</p>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a fish image...",
                type=["jpg", "jpeg", "png"],
                help="Upload clear, well-lit images of fish for best detection results",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        
        else:  # Take Photo
            st.markdown("<p style='color: #6c757d; margin-bottom: 1rem; font-size: 0.85rem;'>Click the button below to capture an image from your camera</p>", unsafe_allow_html=True)
            
            camera_photo = st.camera_input(
                "Take a photo",
                help="Capture a clear, well-lit image of fish",
                label_visibility="collapsed"
            )
            
            if camera_photo is not None:
                image = Image.open(camera_photo)
        
        # Display image if available
        if image is not None:
            st.markdown("### 📷 Original Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Detection button with modern styling - Cleaner spacing
            st.markdown("---")
            detect_button = st.button("🔍 Detect Fish", type="primary", use_container_width=True)
            
            if detect_button:
                # Progress bar for detection
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate loading stages
                status_text.text("Preprocessing image...")
                progress_bar.progress(20)
                time.sleep(0.3)
                
                status_text.text("Running AI detection...")
                progress_bar.progress(50)
                
                # Convert PIL to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Run prediction with enhanced parameters
                results = model.predict(
                    source=image_cv,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    imgsz=img_size,
                    max_det=max_det,
                    save=False,
                    verbose=False,
                    agnostic_nms=False,
                    retina_masks=False,
                    augment=use_tta
                )
                
                progress_bar.progress(80)
                status_text.text("Processing results...")
                time.sleep(0.2)
                
                # Get result
                result = results[0]
                
                # Process detections
                detections = []
                
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'class_id': class_id
                    })
                
                progress_bar.progress(100)
                status_text.text("Detection complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Display results in col2
                with col2:
                    st.markdown("### 🎯 Detection Results")
                    
                    # Annotated image
                    annotated_img = result.plot(
                        conf=show_confidence,
                        labels=show_labels,
                        line_width=2,
                        boxes=show_boxes
                    )
                    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(annotated_img_rgb, use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Statistics Cards
                    st.markdown("---")
                    st.markdown("### 📊 Detection Statistics")
                    
                    col_stat1, col_stat2 = st.columns(2)
                    
                    with col_stat1:
                        st.markdown("""
                        <div class="stats-card">
                            <div class="stat-value">{}</div>
                            <div class="stat-label">Total Fish</div>
                        </div>
                        """.format(len(detections)), unsafe_allow_html=True)
                    
                    with col_stat2:
                        unique_species = len(set([d['class'] for d in detections]))
                        st.markdown("""
                        <div class="stats-card">
                            <div class="stat-value">{}</div>
                            <div class="stat-label">Species Found</div>
                        </div>
                        """.format(unique_species), unsafe_allow_html=True)
                    
                    # Detailed Detections
                    if detections:
                        st.markdown("---")
                        st.markdown("### 📋 Detailed Detection List")
                        
                        # Calculate for report (not displayed)
                        avg_confidence = np.mean([d['confidence'] for d in detections])
                        
                        for i, det in enumerate(detections, 1):
                            
                            with st.expander(
                                f"Fish #{i}: {det['class']} ({det['confidence']:.1%})",
                                expanded=False
                            ):
                                col_det1, col_det2 = st.columns(2)
                                
                                with col_det1:
                                    st.markdown(f"""
                                    <div class="detection-box">
                                        <div class="detection-title">Species</div>
                                        <div class="detection-value">{det['class']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown(f"""
                                    <div class="detection-box">
                                        <div class="detection-title">Class ID</div>
                                        <div class="detection-value">{det['class_id']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col_det2:
                                    confidence_color = "#10b981" if det['confidence'] > 0.7 else "#fb923c" if det['confidence'] > 0.4 else "#ef4444"
                                    st.markdown(f"""
                                    <div class="detection-box">
                                        <div class="detection-title">Confidence</div>
                                        <div class="detection-value" style="color: {confidence_color}">
                                            {det['confidence']:.2%}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("No fish detected. Try adjusting the confidence threshold in the sidebar.")
                    
                    # Download Section
                    st.markdown("---")
                    st.markdown("### 💾 Download Results")
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        result_bytes = cv2.imencode('.jpg', annotated_img)[1].tobytes()
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="Download Annotated Image",
                            data=result_bytes,
                            file_name=f"fish_detection_{timestamp}.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        # Create detection report
                        report = f"""FISH DETECTION - REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*50}

SUMMARY:
- Total Fish Detected: {len(detections)}
- Unique Species: {unique_species}
- Average Confidence: {avg_confidence:.2%}

DETECTIONS:
"""
                        for i, det in enumerate(detections, 1):
                            report += f"\n{i}. {det['class']}\n"
                            report += f"   Class ID: {det['class_id']}\n"
                            report += f"   Confidence: {det['confidence']:.2%}\n"
                        
                        st.download_button(
                            label="Download Report (TXT)",
                            data=report,
                            file_name=f"detection_report_{timestamp}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
    
    # Footer with modern design
    st.markdown("---")
    
    # Quick Tips Section
    with st.expander("Quick Tips for Best Results", expanded=False):
        col_tip1, col_tip2 = st.columns(2)
        
        with col_tip1:
            st.markdown("""
            **Image Quality:**
            - Use well-lit, clear images
            - Ensure fish are clearly visible
            - Avoid blurry or dark photos
            - Front or side view is optimal
            """)
        
        with col_tip2:
            st.markdown("""
            **Settings:**
            - Lower threshold (0.1-0.3): More detections
            - Higher threshold (0.6-0.8): Fewer, confident ones
            - Default (0.25): Balanced approach
            - Adjust based on your needs
            """)
    
    # Species Reference
    with st.expander("Complete Species List (9 Total)", expanded=False):
        col_sp1, col_sp2 = st.columns(2)
        
        with col_sp1:
            st.markdown("""
            1. Gilt-Head Bream
            2. Red Sea Bream
            3. Striped Red Mullet
            4. Black Sea Sprat
            5. House Mackerel
            """)
        
        with col_sp2:
            st.markdown("""
            6. Red Mullet
            7. Sea Bass
            8. Shrimp
            9. Trout
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0; color: #6c757d;'>
        <h3 style='color: #667eea; margin-bottom: 0.5rem; font-size: 1.2rem;'>🐟 Fish Detection AI</h3>
        <p style='font-size: 0.85rem; margin-bottom: 0.3rem;'>
            <strong>Advanced Fish Detection System</strong> | Powered by YOLOv8
        </p>
        <p style='font-size: 0.8rem;'>
            Detection & Counting | Trained on 9,000 images | 9 Fish Species
        </p>
        <p style='font-size: 0.8rem; margin-top: 0.4rem;'>
            <strong style='color: #10b981;'>mAP50: 99.5%</strong> • 
            <strong style='color: #10b981;'>mAP50-95: 83.2%</strong> • 
            <strong style='color: #10b981;'>Precision: 99.9%</strong> • 
            <strong style='color: #10b981;'>Recall: 100%</strong>
        </p>
        <p style='font-size: 0.75rem; margin-top: 0.6rem; color: #9ca3af;'>
            Built with Streamlit • Ultralytics YOLOv8 • OpenCV • PyTorch
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
