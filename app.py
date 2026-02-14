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
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
        animation: fadeInDown 0.8s ease-out;
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
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
        animation: slideIn 1s ease-out;
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
        font-size: 1.3rem;
        font-weight: 400;
        opacity: 0.95;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
        animation: slideIn 1.2s ease-out;
    }
    
    /* Stats Cards - Enhanced with attractive animations */
    .stats-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        cursor: pointer;
        animation: fadeInUp 0.6s ease-out;
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
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.25);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .stat-value {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1;
        animation: countUp 1s ease-out;
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
        font-size: 1rem;
        color: #6c757d;
        font-weight: 600;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Fish Alert */
    .fish-alert {
        background: linear-gradient(135deg, #ffd89b 0%, #fb923c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(251, 146, 60, 0.3);
        margin: 1.5rem 0;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.9; }
    }
    
    /* Detection Box */
    .detection-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .detection-box:hover {
        box-shadow: 0 5px 20px rgba(0,0,0,0.12);
        transform: translateX(5px);
    }
    
    .detection-title {
        font-weight: 600;
        color: #2d3748;
        font-size: 1.1rem;
    }
    
    .detection-value {
        color: #667eea;
        font-weight: 700;
        font-size: 1rem;
    }
    
    /* Buttons - Enhanced with Ripple Effect */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1.5px;
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
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* File Uploader - Enhanced */
    .uploadedFile {
        border-radius: 15px;
        border: 2px solid #667eea;
        padding: 1rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .uploadedFile:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
        border-color: #764ba2;
    }
    
    /* Expander - Enhanced */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-color: #667eea;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.15);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.6);
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
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4);
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
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
        margin: 1rem 0;
        animation: slideInRight 0.5s ease-out;
        transition: all 0.3s ease;
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
        transform: translateX(-5px);
        box-shadow: 0 5px 20px rgba(59, 130, 246, 0.3);
    }
        margin: 1rem 0;
    }
    
    /* 2026 Clean Design Standards - Enhanced Interactive */
    .upload-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 3rem;
        border-radius: 20px;
        border: 2px dashed #667eea;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
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
        transform: scale(1.02);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        animation: none;
    }
    
    /* Minimalist Headings */
    h3 {
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 0.5rem;
        font-size: 1.5rem;
    }
    
    /* Success Box - Enhanced */
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #10b981;
        margin: 1rem 0;
        animation: slideInLeft 0.5s ease-out;
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
        transform: translateX(5px);
        box-shadow: 0 5px 20px rgba(16, 185, 129, 0.3);
    }
    
    /* Image Container - Enhanced */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInScale 0.6s ease-out;
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
        transform: scale(1.02);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.3);
    }
    
    .image-container:hover::before {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }
    
    /* Metric Enhancement */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeIn 0.8s ease-out;
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
        border-radius: 15px;
        animation: slideInLeft 0.5s ease-out;
        transition: all 0.3s ease;
    }
    
    .stAlert:hover {
        transform: translateX(5px);
    }
    
    /* Detection Box Styles - Color Coded and Animated */
    .detection-high {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: bounceIn 0.6s ease-out;
    }
    
    .detection-medium {
        background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%);
        border-left: 5px solid #fb923c;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: bounceIn 0.6s ease-out;
    }
    
    .detection-low {
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: bounceIn 0.6s ease-out;
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
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #667eea 50%, transparent 100%);
        margin: 2rem 0;
        animation: fadeIn 1s ease-out;
    }
    
    /* Radio Buttons - Enhanced */
    .stRadio > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stRadio > div:hover {
        border-color: #667eea;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.15);
    }
    
    /* Camera Input */
    [data-testid="stCameraInput"] {
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stCameraInput"]:hover {
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Advanced Animations for Interactive Statistics */
    @keyframes slideIn {
        from {
            width: 0%;
            opacity: 0;
        }
        to {
            width: var(--target-width);
            opacity: 1;
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    @keyframes scaleIn {
        from {
            transform: scale(0.8);
            opacity: 0;
        }
        to {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 5px rgba(102, 126, 234, 0.5);
        }
        50% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.8);
        }
    }
    
    /* Enhanced Statistics Cards with Glassmorphism */
    .stats-card-enhanced {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stats-card-enhanced:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 15px 45px rgba(102, 126, 234, 0.3);
    }
    
    /* Animated Progress Bars */
    .progress-bar-animated {
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar-animated::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer 2s infinite;
    }
    
    /* Interactive Hover Effects */
    .interactive-card {
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .interactive-card:hover {
        transform: translateX(8px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Confidence Badge Animations */
    .confidence-badge {
        animation: scaleIn 0.5s ease-out;
        transition: all 0.3s ease;
    }
    
    .confidence-badge:hover {
        transform: scale(1.1);
        filter: brightness(1.1);
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
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        
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
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("### Advanced Options")
        
        use_tta = st.checkbox(
            "Test Time Augmentation (TTA)",
            value=True,
            help="✓ Enabled by default - Improves detection accuracy with multiple augmented predictions"
        )
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("### Display Options")
        show_labels = st.checkbox("Show Labels", value=True)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        
        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
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
                        padding: 1.5rem; border-radius: 15px; border-left: 5px solid #10b981;'>
                <h4 style='color: #065f46; margin-top: 0;'>🏆 Excellent Detection Accuracy</h4>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
                    <div>
                        <p style='margin: 0.5rem 0; color: #047857;'><strong>mAP50:</strong> <span style='font-size: 1.3rem; color: #059669;'>99.50%</span></p>
                        <p style='margin: 0.5rem 0; color: #047857;'><strong>mAP50-95:</strong> <span style='font-size: 1.3rem; color: #059669;'>83.23%</span></p>
                    </div>
                    <div>
                        <p style='margin: 0.5rem 0; color: #047857;'><strong>Precision:</strong> <span style='font-size: 1.3rem; color: #059669;'>99.92%</span></p>
                        <p style='margin: 0.5rem 0; color: #047857;'><strong>Recall:</strong> <span style='font-size: 1.3rem; color: #059669;'>100.00%</span></p>
                    </div>
                </div>
                <p style='margin-top: 1rem; margin-bottom: 0; color: #065f46; font-size: 0.9rem;'>
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
    st.markdown("<br>", unsafe_allow_html=True)
    
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
        st.markdown("### Get Fish Image")
        
        # Input method selector
        input_method = st.radio(
            "Choose input method:",
            options=["📤 Upload Image", "📸 Take Photo"],
            horizontal=True,
            label_visibility="visible"
        )
        
        image = None
        
        if input_method == "📤 Upload Image":
            st.markdown("<p style='color: #6c757d; margin-bottom: 1.5rem;'>Drag and drop or click to upload JPG, JPEG, or PNG images</p>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a fish image...",
                type=["jpg", "jpeg", "png"],
                help="Upload clear, well-lit images of fish for best detection results",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        
        else:  # Take Photo
            st.markdown("<p style='color: #6c757d; margin-bottom: 1.5rem;'>Click the button below to capture an image from your camera</p>", unsafe_allow_html=True)
            
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
            st.markdown("<br>", unsafe_allow_html=True)
            detect_button = st.button("Detect Fish", type="primary", use_container_width=True)
            
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
                    st.markdown("### Detection Results")
                    
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
                    
                    # Statistics Cards - Enhanced Interactive Design
                    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
                    st.markdown("### 📊 Detection Analytics")
                    
                    # Top-level metrics with animated cards
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.markdown(f"""
                        <div class="stats-card">
                            <div class="stat-value">{len(detections)}</div>
                            <div class="stat-label">Total Fish</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_stat2:
                        unique_species = len(set([d['class'] for d in detections]))
                        st.markdown(f"""
                        <div class="stats-card">
                            <div class="stat-value">{unique_species}</div>
                            <div class="stat-label">Species Found</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_stat3:
                        avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
                        st.markdown(f"""
                        <div class="stats-card">
                            <div class="stat-value">{avg_confidence:.0%}</div>
                            <div class="stat-label">Avg Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed Detections with Interactive Elements
                    if detections:
                        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
                        
                        # Species Distribution Analysis
                        st.markdown("#### 🐠 Species Distribution")
                        species_counts = {}
                        for det in detections:
                            species_counts[det['class']] = species_counts.get(det['class'], 0) + 1
                        
                        # Create interactive progress bars for each species
                        max_count = max(species_counts.values())
                        for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
                            percentage = (count / len(detections)) * 100
                            bar_width = (count / max_count) * 100
                            
                            # Color coding based on count
                            if count >= max_count * 0.7:
                                color = "#10b981"  # Green
                                bg_color = "#d1fae5"
                            elif count >= max_count * 0.4:
                                color = "#3b82f6"  # Blue
                                bg_color = "#dbeafe"
                            else:
                                color = "#f59e0b"  # Orange
                                bg_color = "#fef3c7"
                            
                            st.markdown(f"""
                            <div style='margin: 0.8rem 0; padding: 1rem; background: {bg_color}; 
                                        border-radius: 12px; border-left: 4px solid {color};
                                        transition: all 0.3s ease; cursor: pointer;'
                                 onmouseover="this.style.transform='translateX(5px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.1)';"
                                 onmouseout="this.style.transform='translateX(0)'; this.style.boxShadow='none';">
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                                    <span style='font-weight: 600; color: #1f2937; font-size: 1rem;'>{species}</span>
                                    <span style='font-weight: 700; color: {color}; font-size: 1.1rem;'>{count} fish ({percentage:.1f}%)</span>
                                </div>
                                <div style='background: white; height: 8px; border-radius: 10px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);'>
                                    <div style='background: linear-gradient(90deg, {color}, {color}dd); 
                                                height: 100%; width: {bar_width}%; 
                                                border-radius: 10px;
                                                animation: slideIn 0.8s ease-out;
                                                box-shadow: 0 0 10px {color}66;'></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Confidence Level Breakdown
                        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
                        st.markdown("#### 📈 Confidence Analysis")
                        
                        # Calculate confidence distribution
                        high_conf = len([d for d in detections if d['confidence'] >= 0.7])
                        medium_conf = len([d for d in detections if 0.4 <= d['confidence'] < 0.7])
                        low_conf = len([d for d in detections if d['confidence'] < 0.4])
                        
                        total = len(detections)
                        
                        # Interactive confidence breakdown with animated bars
                        confidence_data = [
                            ("High (≥70%)", high_conf, "#10b981", "#d1fae5"),
                            ("Medium (40-70%)", medium_conf, "#f59e0b", "#fef3c7"),
                            ("Low (<40%)", low_conf, "#ef4444", "#fee2e2")
                        ]
                        
                        for label, count, color, bg_color in confidence_data:
                            percentage = (count / total * 100) if total > 0 else 0
                            st.markdown(f"""
                            <div style='margin: 0.8rem 0; padding: 1.2rem; 
                                        background: linear-gradient(135deg, {bg_color} 0%, white 100%);
                                        border-radius: 15px; border: 2px solid {color}33;
                                        transition: all 0.3s ease;'
                                 onmouseover="this.style.borderColor='{color}'; this.style.transform='scale(1.02)';"
                                 onmouseout="this.style.borderColor='{color}33'; this.style.transform='scale(1)';">
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;'>
                                    <span style='font-weight: 600; color: #1f2937; font-size: 1.05rem;'>
                                        <span style='display: inline-block; width: 12px; height: 12px; 
                                                     background: {color}; border-radius: 50%; margin-right: 8px;'></span>
                                        {label}
                                    </span>
                                    <span style='font-weight: 700; color: {color}; font-size: 1.2rem;'>{count}</span>
                                </div>
                                <div style='background: #f3f4f6; height: 12px; border-radius: 10px; overflow: hidden;'>
                                    <div style='background: linear-gradient(90deg, {color}, {color}cc); 
                                                height: 100%; width: {percentage}%; 
                                                border-radius: 10px;
                                                animation: slideIn 1s ease-out;
                                                position: relative;
                                                box-shadow: 0 2px 8px {color}44;'>
                                        <div style='position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
                                                    color: white; font-size: 0.75rem; font-weight: 600;'>
                                            {percentage:.1f}%
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Overall Statistics Summary Box
                        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
                        min_conf = min([d['confidence'] for d in detections])
                        max_conf = max([d['confidence'] for d in detections])
                        
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    padding: 1.5rem; border-radius: 20px; color: white;
                                    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
                                    margin: 1rem 0;'>
                            <h4 style='margin: 0 0 1rem 0; font-size: 1.2rem; font-weight: 700;'>
                                📊 Summary Statistics
                            </h4>
                            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
                                <div style='background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 12px;
                                            backdrop-filter: blur(10px);'>
                                    <div style='font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.3rem;'>Average Confidence</div>
                                    <div style='font-size: 1.8rem; font-weight: 800;'>{avg_confidence:.1%}</div>
                                </div>
                                <div style='background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 12px;
                                            backdrop-filter: blur(10px);'>
                                    <div style='font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.3rem;'>Confidence Range</div>
                                    <div style='font-size: 1.4rem; font-weight: 700;'>{min_conf:.0%} - {max_conf:.0%}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detailed Detection List
                        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
                        st.markdown("#### 🔍 Individual Detections")
                        
                        for i, det in enumerate(detections, 1):
                            # Color based on confidence
                            if det['confidence'] >= 0.7:
                                badge_color = "#10b981"
                                badge_bg = "#d1fae5"
                            elif det['confidence'] >= 0.4:
                                badge_color = "#f59e0b"
                                badge_bg = "#fef3c7"
                            else:
                                badge_color = "#ef4444"
                                badge_bg = "#fee2e2"
                            
                            with st.expander(
                                f"🐟 Fish #{i}: {det['class']} - {det['confidence']:.1%}",
                                expanded=False
                            ):
                                st.markdown(f"""
                                <div style='padding: 0.5rem;'>
                                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;'>
                                        <div style='background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                                                    padding: 1rem; border-radius: 12px; border-left: 4px solid #667eea;'>
                                            <div style='color: #6c757d; font-size: 0.85rem; margin-bottom: 0.3rem;'>Species</div>
                                            <div style='color: #667eea; font-weight: 700; font-size: 1.1rem;'>{det['class']}</div>
                                        </div>
                                        <div style='background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                                                    padding: 1rem; border-radius: 12px; border-left: 4px solid #667eea;'>
                                            <div style='color: #6c757d; font-size: 0.85rem; margin-bottom: 0.3rem;'>Class ID</div>
                                            <div style='color: #667eea; font-weight: 700; font-size: 1.1rem;'>{det['class_id']}</div>
                                        </div>
                                    </div>
                                    <div style='background: {badge_bg}; padding: 1.2rem; border-radius: 12px;
                                                border-left: 4px solid {badge_color};'>
                                        <div style='color: #1f2937; font-size: 0.85rem; margin-bottom: 0.5rem;'>Confidence Score</div>
                                        <div style='display: flex; align-items: center; gap: 1rem;'>
                                            <div style='flex: 1; background: white; height: 10px; border-radius: 10px; overflow: hidden;'>
                                                <div style='background: {badge_color}; height: 100%; width: {det['confidence']*100}%;
                                                            border-radius: 10px; transition: width 0.5s ease;'></div>
                                            </div>
                                            <div style='color: {badge_color}; font-weight: 800; font-size: 1.3rem; min-width: 60px;'>
                                                {det['confidence']:.1%}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No fish detected. Try adjusting the confidence threshold in the sidebar.")
                    
                    # Download Section
                    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
                    st.markdown("### Download Results")
                    
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
    st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)
    
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
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #6c757d; border-top: 1px solid #e2e8f0;'>
        <h3 style='color: #667eea; margin-bottom: 1rem;'>🐟 Fish Detection AI</h3>
        <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
            <strong>Advanced Fish Detection System</strong> | Powered by YOLOv8
        </p>
        <p style='font-size: 0.95rem;'>
            Detection & Counting | Trained on 9,000 images | 9 Fish Species
        </p>
        <p style='font-size: 0.9rem; margin-top: 0.5rem;'>
            <strong style='color: #10b981;'>mAP50: 99.5%</strong> • 
            <strong style='color: #10b981;'>mAP50-95: 83.2%</strong> • 
            <strong style='color: #10b981;'>Precision: 99.9%</strong> • 
            <strong style='color: #10b981;'>Recall: 100%</strong>
        </p>
        <p style='font-size: 0.85rem; margin-top: 1rem; color: #9ca3af;'>
            Built with Streamlit • Ultralytics YOLOv8 • OpenCV • PyTorch
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
