import os
import io
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torchvision import transforms
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import sqlite3
import secrets
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'deepfake_image_model.pth')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
DATABASE_PATH = os.path.join(BASE_DIR, 'users.db')
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES



def get_db():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection

def init_db():
    with get_db() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

def logged_in():
    return bool(session.get('user_id'))

init_db()

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(96, 2))
    def forward(self, x):
        return self.classifier(self.features(x))

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
model = None
classes = ['fake', 'real']
model_error = None
try:
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = SmallCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    classes = checkpoint.get('classes', classes)
except Exception as exc:
    model_error = str(exc)

@app.route('/')
def home():
    if not logged_in():
        return redirect(url_for('login'))
    return render_template('index.html', model_loaded=(model is not None), model_error=model_error, username=session.get('username'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if logged_in():
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        if not username or not email or not password:
            flash('Please fill in every field.', 'error')
        elif len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
        elif len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
        elif password != confirm:
            flash('Passwords do not match.', 'error')
        else:
            try:
                with get_db() as db:
                    cursor = db.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                                      (username, email, generate_password_hash(password)))
                    session.clear()
                    session['user_id'] = cursor.lastrowid
                    session['username'] = username
                return redirect(url_for('home'))
            except sqlite3.IntegrityError:
                flash('That username or email is already registered.', 'error')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if logged_in():
        return redirect(url_for('home'))
    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '')
        with get_db() as db:
            user = db.execute('SELECT * FROM users WHERE username = ? OR email = ?',
                              (identifier, identifier.lower())).fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('home'))
        flash('Incorrect username/email or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if not logged_in():
        return jsonify({'error': 'Please log in to use the detector.'}), 401

    if model is None:
        return jsonify({'error': 'Model could not be loaded. Check that deepfake_image_model.pth is beside app.py.'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'Please select an image first.'}), 400
    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'No file selected.'}), 400
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': 'Unsupported file type. Upload JPG, JPEG, PNG, WEBP, or BMP.'}), 400
    raw = file.read()
    if not raw:
        return jsonify({'error': 'The selected file is empty.'}), 400
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({'error': 'Image is larger than 10 MB.'}), 413
    try:
        image = Image.open(io.BytesIO(raw)).convert('RGB')
    except (UnidentifiedImageError, OSError):
        return jsonify({'error': 'This file is not a valid image.'}), 400
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    label = classes[pred_idx].lower()
    score = float(probs[pred_idx].item() * 100)
    return jsonify({
        'label': label,
        'display_label': 'FAKE / AI-GENERATED' if label == 'fake' else 'REAL',
        'confidence': round(score, 2),
        'scores': {classes[i]: round(float(probs[i].item()) * 100, 2) for i in range(len(classes))},
        'note': 'Demo model trained on only 10 images. Results are not reliable for real-world verification.'
    })

@app.errorhandler(413)
def too_large(_error):
    return jsonify({'error': 'File is larger than the 10 MB upload limit.'}), 413

if __name__ == '__main__':
    init_db()
    print('Open the website at http://127.0.0.1:5000')
    app.run(debug=False, host='127.0.0.1', port=5000)
