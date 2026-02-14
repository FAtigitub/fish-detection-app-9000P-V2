@echo off
echo ===================================
echo   SardineVision AI - Quick Start
echo ===================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo ===================================
echo Starting Streamlit app...
echo ===================================
echo.
echo App will open at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run app.py
