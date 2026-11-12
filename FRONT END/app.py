import os
import sqlite3
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, redirect, request, url_for, session

# -------------------------------------------------------------
# Base Directories and Paths
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'saved_images')
os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)

# -------------------------------------------------------------
# Flask Application Setup
# -------------------------------------------------------------
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'), template_folder=os.path.join(BASE_DIR, 'templates'))
app.secret_key = 'brain_hemorrhage_detection_secret_key_2026'

# -------------------------------------------------------------
# Database Setup: MySQL with Graceful SQLite Fallback
# -------------------------------------------------------------
USE_MYSQL = False
mysql_conn = None

def init_database():
    global USE_MYSQL, mysql_conn
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            port="3306"
        )
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS brain")
        conn.commit()
        cursor.close()
        conn.close()

        mysql_conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            port="3306",
            database="brain"
        )
        cursor = mysql_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(225),
                email VARCHAR(50),
                password VARCHAR(50)
            )
        """)
        mysql_conn.commit()
        cursor.close()
        USE_MYSQL = True
        print("[Database] Successfully connected to MySQL database 'brain'.")
    except Exception as e:
        USE_MYSQL = False
        print(f"[Database] MySQL connection unavailable ({e}). Falling back to SQLite.")
        sqlite_path = os.path.join(BASE_DIR, 'brain.db')
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                password TEXT
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[Database] SQLite initialized at: {sqlite_path}")

init_database()

def get_db_connection():
    global USE_MYSQL, mysql_conn
    if USE_MYSQL:
        try:
            mysql_conn.ping(reconnect=True, attempts=3, delay=1)
            return mysql_conn
        except Exception:
            try:
                import mysql.connector
                mysql_conn = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="",
                    port="3306",
                    database="brain"
                )
                return mysql_conn
            except Exception:
                USE_MYSQL = False
    return sqlite3.connect(os.path.join(BASE_DIR, 'brain.db'))

def executionquery(query, values=()):
    conn = get_db_connection()
    if not USE_MYSQL:
        query = query.replace("%s", "?")
    cursor = conn.cursor()
    cursor.execute(query, values)
    conn.commit()
    cursor.close()
    if not USE_MYSQL:
        conn.close()

def retrivequery1(query, values=()):
    conn = get_db_connection()
    if not USE_MYSQL:
        query = query.replace("%s", "?")
    cursor = conn.cursor()
    cursor.execute(query, values)
    data = cursor.fetchall()
    cursor.close()
    if not USE_MYSQL:
        conn.close()
    return data

def retrivequery2(query):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    if not USE_MYSQL:
        conn.close()
    return data

# -------------------------------------------------------------
# Deep Learning Models Setup (PyTorch Native)
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[AI Models] Running on device: {device}")

# 1. Relevance Model (MobileNetV2: 0=relevant, 1=irrelevant)
class MobileNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=None)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

rel_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

rel_model_path = os.path.join(BASE_DIR, "mobilenet.pt")
relevance_model = MobileNetModel(num_classes=2).to(device)
relevance_model.load_state_dict(torch.load(rel_model_path, map_location=device))
relevance_model.eval()
print("[AI Models] Relevance verification model loaded successfully.")

# 2. Hemorrhage Detection Model (MobileNetV1: 0=Hemorrhagic, 1=NORMAL)
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d((0, 1, 0, 1)) if stride == 2 else nn.Identity(),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1 if stride == 1 else 0, bias=False),
            nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.01),
            nn.ReLU6(inplace=True)
        )

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.pad = nn.ZeroPad2d((0, 1, 0, 1)) if stride == 2 else nn.Identity()
        self.dw = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride,
                            padding=1 if stride == 1 else 0, groups=in_planes, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_planes, eps=0.001, momentum=0.01)
        self.dw_relu = nn.ReLU6(inplace=True)
        self.pw = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.01)
        self.pw_relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.dw_relu(self.dw_bn(self.dw(self.pad(x))))
        x = self.pw_relu(self.pw_bn(self.pw(x)))
        return x

class HemorrhageMobileNet(nn.Module):
    def __init__(self, num_classes=2):
        super(HemorrhageMobileNet, self).__init__()
        self.conv1 = ConvBNReLU(3, 32, stride=2)
        configs = [
            (32, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2),
            (256, 256, 1), (256, 512, 2), (512, 512, 1), (512, 512, 1),
            (512, 512, 1), (512, 512, 1), (512, 512, 1), (512, 1024, 2),
            (1024, 1024, 1),
        ]
        self.blocks = nn.ModuleList([
            DepthwiseSeparableConv(inp, oup, s) for (inp, oup, s) in configs
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for b in self.blocks:
            x = b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

hem_model_path = os.path.join(BASE_DIR, "hemorrhage_mobilenet.pt")
hemorrhage_model = HemorrhageMobileNet(num_classes=2).to(device)
hemorrhage_model.load_state_dict(torch.load(hem_model_path, map_location=device))
hemorrhage_model.eval()
print("[AI Models] Brain Hemorrhage classification model loaded successfully.")

# -------------------------------------------------------------
# Prediction Pipeline Helpers
# -------------------------------------------------------------
def predict_image_pipeline(image_path):
    img = Image.open(image_path).convert('RGB')

    # Step 1: Relevance Check
    rel_tensor = rel_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        rel_logits = relevance_model(rel_tensor)
        rel_prob = torch.softmax(rel_logits, dim=1)[0].cpu().numpy()
        rel_pred = int(np.argmax(rel_prob))

    if rel_pred != 0:
        return {
            'is_relevant': False,
            'prediction': "The image is not relevant to this project.",
            'confidence': f"{rel_prob[1]*100:.2f}%",
            'type': 'warning'
        }

    # Step 2: Hemorrhage Classification
    hem_img = img.resize((256, 256))
    arr = np.array(hem_img, dtype=np.float32) / 255.0
    hem_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        hem_logits = hemorrhage_model(hem_tensor)
        hem_prob = torch.softmax(hem_logits, dim=1)[0].cpu().numpy()
        hem_pred = int(np.argmax(hem_prob))

    if hem_pred == 0:
        return {
            'is_relevant': True,
            'prediction': "Hemorrhagic",
            'confidence': f"{hem_prob[0]*100:.2f}%",
            'type': 'danger'
        }
    else:
        return {
            'is_relevant': True,
            'prediction': "NORMAL",
            'confidence': f"{hem_prob[1]*100:.2f}%",
            'type': 'success'
        }

# -------------------------------------------------------------
# Flask Web Routes
# -------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        c_password = request.form.get('c_password', '')

        if not name or not email or not password:
            return render_template('register.html', message="Please fill in all required fields!")

        if password != c_password:
            return render_template('register.html', message="Confirm password does not match!")

        email_data = retrivequery2("SELECT UPPER(email) FROM users")
        email_data_list = [row[0] for row in email_data]
        if email.upper() in email_data_list:
            return render_template('register.html', message="This email ID already exists!")

        executionquery(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, password)
        )
        return render_template('login.html', message="Successfully Registered! Please login.")

    return render_template('register.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        email_data = retrivequery2("SELECT UPPER(email) FROM users")
        email_data_list = [row[0] for row in email_data]

        if email.upper() in email_data_list:
            password_data = retrivequery1("SELECT password, name FROM users WHERE UPPER(email) = UPPER(%s)", (email,))
            if password_data and password == password_data[0][0]:
                session['user_email'] = email
                session['user_name'] = password_data[0][1]
                return redirect(url_for('home'))
            return render_template('login.html', message="Invalid Password!")
        return render_template('login.html', message="This email ID does not exist!")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login', message="Logged out successfully!"))

@app.route('/home')
def home():
    user = session.get('user_name', 'User')
    return render_template('home.html', user=user)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('prediction.html', message="No file part in the request!")

        myfile = request.files['file']
        if not myfile or myfile.filename == '':
            return render_template('prediction.html', message="Please select an image file first!")

        filename = secure_filename(myfile.filename)
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        _, ext = os.path.splitext(filename.lower())
        if ext not in allowed_extensions:
            return render_template('prediction.html', message="Invalid file format! Please upload a JPG or PNG image.")

        # Save uploaded image safely
        filepath = os.path.join(SAVED_IMAGES_DIR, filename)
        myfile.save(filepath)

        # Run AI detection pipeline
        try:
            result = predict_image_pipeline(filepath)
            return render_template(
                'prediction.html',
                prediction=result['prediction'],
                confidence=result['confidence'],
                prediction_type=result['type'],
                file_name=filename
            )
        except Exception as e:
            return render_template('prediction.html', message=f"Prediction error: {str(e)}", file_name=filename)

    return render_template('prediction.html')

# -------------------------------------------------------------
# Main Application Entrypoint
# -------------------------------------------------------------
if __name__ == '__main__':
    print(f"[Brain Hemorrhage Detection Server] Starting at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
