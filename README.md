# 🐟 SardineVision AI - Fish Detection System

A **modern, professional, and interactive** fish detection web application powered by YOLOv8, specifically trained to detect and count Indonesian fish species including **Sardines**.

## ✨ Key Features

### 🎨 Modern UI/UX Design
- **Gradient hero section** with professional branding
- **Interactive cards** with hover effects and animations
- **Real-time progress indicators** for detection process
- **Responsive layout** optimized for all devices
- **Custom color scheme** with purple-blue gradients

### 🔍 Detection Capabilities
- **Image Upload**: Upload fish images in JPG, JPEG, or PNG format
- **Live Camera**: Capture photos directly from camera (requires HTTPS/Cloud)
- **Real-time Detection**: Instant fish detection and counting with visual feedback
- **Sardine Tracking**: Special highlighting and alerts for sardine detections
- **Multi-Species**: Detects 19 different Indonesian fish species
- **Confidence Adjustment**: Customize detection sensitivity via interactive slider

### 📊 Analytics & Reporting
- **Real-time Statistics**: Total fish count, sardine count, species diversity
- **Detailed Detection List**: Expandable cards for each detected fish
- **Visual Confidence Indicators**: Color-coded confidence scores
- **Average Confidence Metrics**: Overall detection quality assessment
- **Downloadable Reports**: Export annotated images and text reports

### ⚙️ Advanced Controls
- **Adjustable Confidence Threshold**: Fine-tune detection sensitivity
- **Display Options**: Toggle labels, confidence scores, bounding boxes
- **Model Information**: Expandable sections with training details
- **Species Reference**: Complete list of 19 detectable species

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this directory**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Ensure model file exists:**
   - Make sure `best.pt` is in the same directory as `app.py`

4. **Run the app:**
```bash
streamlit run app.py
```

5. **Open browser:**
   - The app will automatically open at `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud (FREE)

### Step 1: Prepare GitHub Repository

1. **Create a new GitHub repository:**
   - Go to https://github.com/new
   - Name it: `sardine-detection-app`
   - Make it Public
   - Don't initialize with README (we have one)

2. **Upload files to GitHub:**
   ```bash
   cd sardine_detection_app
   git init
   git add .
   git commit -m "Initial commit - SardineVision AI"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/sardine-detection-app.git
   git push -u origin main
   ```

   **OR** manually upload via GitHub web interface:
   - Click "uploading an existing file"
   - Drag and drop: `app.py`, `requirements.txt`, `README.md`, `best.pt`
   - Commit changes

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit: https://share.streamlit.io/

2. **Sign in with GitHub:**
   - Click "Sign in with GitHub"
   - Authorize Streamlit

3. **Deploy new app:**
   - Click "New app"
   - **Repository:** Select `YOUR_USERNAME/sardine-detection-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - Click "Deploy!"

4. **Wait for deployment:**
   - Initial deployment takes 5-10 minutes
   - Streamlit will install all dependencies

5. **Get your public URL:**
   - Format: `https://YOUR_USERNAME-sardine-detection-app-XXXXX.streamlit.app`
   - Share this URL with anyone!

## 📦 Model Information

- **Framework:** YOLOv8 Medium (yolov8m.pt)
- **Training Dataset:** Indonesian Fish Dataset (4,645 images)
- **Classes:** 19 fish species
- **Special Feature:** Sardine detection (Class 17: Tribus Sardini)
- **Training Stats:**
  - Train: 3,480 images
  - Valid: 584 images
  - Test: 581 images
  - Epochs: 100

## 🐟 Detected Fish Species

1. Alepes Djedaba (Round Scad)
2. Atropus Atropos (Clupea)
3. Caranx Ignobilis (Giant Trevally)
4. Chanos Chanos (Milkfish)
5. Decapterus Macarellus (Mackerel Scad)
6. Euthynnus Affinis (Kawakawa Bonito)
7. Katsuwonus Pelamis (Skipjack Tuna)
8. Lutjanus Malabaricus (Malabar Red Snapper)
9. Parastromateus Niger (Black Pomfret)
10. Rastrelliger Kanagurta (Indian Mackerel)
11. Rastrelliger sp (Mackerel Species)
12. Scaridae (Parrotfish)
13. Scomber Japonicus (Chub Mackerel)
14. Scomberomorus Guttatus (Indo-Pacific King Mackerel)
15. Thunnus Alalunga (Albacore Tuna)
16. Thunnus Obesus (Bigeye Tuna)
17. Thunnus Tonggol (Longtail Tuna)
18. **⭐ Tribus Sardini (SARDINE)** - Target Species
19. Upeneus Moluccensis (Goldband Goatfish)

## 🛠️ Troubleshooting

### Camera capture not working
**Local environment:**
- Camera capture requires HTTPS or Streamlit Cloud
- **Solution:** Use "Upload Image" option for local testing
- Camera will work automatically once deployed to Streamlit Cloud

**On Streamlit Cloud:**
- Browser may block camera access
- Click "Allow" when prompted for camera permissions
- Check browser settings if camera still doesn't work

### Model file not found
- Ensure `best.pt` is in the same directory as `app.py`
- File size should be ~50MB

### Streamlit Cloud deployment fails
- Check if `best.pt` is uploaded to GitHub (max file size: 100MB)
- If too large, use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.pt"
  git add .gitattributes
  git add best.pt
  git commit -m "Add model with LFS"
  git push
  ```

### Low detection accuracy
- Adjust confidence threshold (sidebar)
- Use well-lit, clear images
- Ensure fish are clearly visible

## 📝 Usage Tips

1. **Best Results:**
   - Use clear, well-lit images
   - Fish should be clearly visible
   - Avoid blurry or dark images

2. **Confidence Threshold:**
   - Higher (0.5-0.8): Fewer but more confident detections
   - Lower (0.1-0.3): More detections, may include false positives
   - Default (0.25): Balanced

3. **Production Line Integration:**
   - Use camera capture for real-time detection
   - Set appropriate confidence threshold
   - Monitor sardine count for quality control

## 🎯 Next Steps (Roadmap)

- [ ] Task 2: Sardine classification by size/type
- [ ] Task 3: Quality control & defect detection
- [ ] Task 4: Production line camera integration
- [ ] Task 5: Real-time counting dashboard
- [ ] Task 6: Data logging and analytics

## 📄 License

This project is part of the SardineVision AI system for production line automation.

## 🙋 Support

For issues or questions, please refer to the main project documentation.

---

**🐟 SardineVision AI** | Powered by YOLOv8 | Built with Streamlit
