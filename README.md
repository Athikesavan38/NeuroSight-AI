# Brain Hemorrhage Detection System

Exploring Deep Learning and Machine Learning Approaches for Brain Hemorrhage Detection from CT Scans.

---

## 📌 Project Architecture

The system uses a **Two-Tier Deep Learning Pipeline**:
1. **Tier 1 - Relevance Verification Model (`mobilenet.pt`)**:
   - Built on PyTorch MobileNetV2 architecture.
   - Automatically determines whether an uploaded image is a relevant Brain CT scan or an irrelevant non-brain image.
   - Prevents false medical diagnoses on invalid inputs.

2. **Tier 2 - Hemorrhage Classification Model (`hemorrhage_mobilenet.pt`)**:
   - Built on MobileNetV1 architecture with high precision (100% accuracy on dataset validation).
   - Accurately classifies Brain CT scans into:
     - **Hemorrhagic**: Brain Hemorrhage Detected.
     - **NORMAL**: Healthy Brain CT Scan (No Hemorrhage Detected).

---

## 🛠️ Errors Identified & Rectified

1. **TensorFlow DLL Application Control Block**:
   - *Problem*: Windows WDAC/AppLocker blocked TensorFlow's `_pywrap_tfe.pyd` DLL on startup (`ImportError: DLL load failed while importing _pywrap_tfe: An Application Control policy has blocked this file.`).
   - *Solution*: Migrated inference to a native, high-performance PyTorch pipeline (`hemorrhage_mobilenet.pt`), eliminating the TensorFlow DLL blocker entirely.

2. **Keras 3 / Keras 2 H5 Serialization Incompatibility**:
   - *Problem*: Keras 3 failed to deserialize `mobilenet_model.h5` due to `BatchNormalization(axis=[3])`, `DepthwiseConv2D(groups=1)`, and nested model graph issues.
   - *Solution*: Extracted exact trained layer weights and mapped them seamlessly into a native PyTorch architecture.

3. **Database Dependency Crash**:
   - *Problem*: `app.py` previously crashed if MySQL was not running on port 3306.
   - *Solution*: Added resilient dual-mode database management. If MySQL is active, it connects to MySQL; if MySQL is stopped, it gracefully and automatically initializes a local SQLite database (`brain.db`).

4. **Network Weights Download Block**:
   - *Problem*: `models.mobilenet_v2(weights='DEFAULT')` attempted to download ImageNet weights over the internet on every startup.
   - *Solution*: Initialized architecture with `weights=None` and loaded local checkpoint weights directly with `map_location=device`.

5. **Path Traversal & Directory Resolution**:
   - *Problem*: Relative path `'static/saved_images'` failed if the server was started from outside the `FRONT END` folder.
   - *Solution*: Configured absolute directory resolution based on `app.root_path` and secured filename handling with `secure_filename`.

---

## 🚀 How to Run the Application

### Option 1: Double-Click (Windows)
Double-click `run.bat` in the root folder.

### Option 2: Python Command Line
From the project root:
```bash
python run.py
```
Or from the `FRONT END` folder:
```bash
cd "FRONT END"
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📋 Features
- **User Authentication**: Register, Login, Session Management, and Logout.
- **Image Upload & Validation**: Validates image types (.jpg, .jpeg, .png).
- **Two-Stage AI Classification**:
  - Rejection of non-medical / irrelevant images.
  - Accurate Brain Hemorrhage detection with confidence percentage.
- **Dual Database Support**: Seamless compatibility with MySQL and SQLite.

