# CloudVision 🌤️
**Advanced Multi-Model Cloud Detection & Segmentation System**

CloudVision is a state-of-the-art satellite imagery analysis platform designed to detect and segment cloud formations with high precision. It leverages multiple deep learning models (Attention U-Net variants) to identify cloud types, shadows, and clear skies, providing actionable meteorological insights via Google Gemini AI.


## 🚀 Key Features

*   **Multi-Model Inference**: Choose between three specialized models for different analysis needs:
    *   **V1 (Best Performing)**: The most accurate model, successfully identifying clear skies, shadows, and cloud types with high precision. **Recommended for all tasks.**
    *   **V2 & V3 (Legacy)**: Previous versions with different weight configurations. Included for comparative purposes.
*   **Shadow Mitigation**: Automated removal of cloud shadows to recover ground details.
*   **AI Analysis**: Integrated **Google Gemini 2.5** to generate detailed meteorological reports based on segmentation metrics.
*   **Interactive Dashboard**: Real-time visualization of original imagery, segmentation masks, and class distribution.
*   **Grayscale Masks**: Exportable, standard-compliant segmentation masks.

## 🛠️ Tech Stack

*   **Frontend**: Next.js 15, React, Tailwind CSS, Chart.js
*   **Backend**: Django REST Framework, Python
*   **AI/ML**: TensorFlow/Keras, Google Gemini API
*   **Processing**: NumPy, Pillow, Scikit-Image

## 📦 Installation & Setup

### Prerequisites
*   Node.js (v18+)
*   Python (3.9+)
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/kazmisohail/CloudVision.git
cd CloudVision
```

### 2. Backend Setup
Navigate to the backend directory and set up the Python environment.

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
# source venv/bin/activate

pip install -r requirements.txt
```

**Configuration**:
Create a `.env` file in the `backend` folder:
```env
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_PATH_V1=models/Attention_UNet_Improved_2.keras
MODEL_PATH_V2=models/Attention_UNet_Advanced_1.keras
MODEL_PATH_V3=models/Attention_UNet_Balanced_Final.keras
```

```
Find the trained model here. Download it and save it in backend/models folder
https://huggingface.co/kazmisohail/Attention_UNet
```

**Run Server**:
```bash
python manage.py runserver
```

### 3. Frontend Setup
Open a new terminal and navigate to the frontend directory.

```bash
cd frontend
npm install
```

**Run Client**:
```bash
npm run dev
```

Visit `http://localhost:3000` to access the application.

## 📝 Usage Guide

1.  **Upload Image**: Drag and drop a satellite image into the upload zone.
2.  **Select Model**: Choose V1, V2, or V3 from the Control Panel.
3.  **Analyze**: Click "Analyze Cloud Formations".
4.  **View Results**:
    *   **Segmentation Mask**: Visualizes cloud types (Thick, Thin, Shadow, Clear).
    *   **Mitigated Image**: Shows the image with shadows removed.
    *   **AI Report**: Read the Gemini-generated weather analysis.

## 👥 Team
*   **Ms. Amna Iftikhar** - Supervisor
*   **Sohail Abbas** - Student
*   **Sofia Haider** - Student
*   **Emad Tariq** - Student

---
*Built for Final Year Project at Bahria University.*