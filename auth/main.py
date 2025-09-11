from flask import Flask, redirect, session, jsonify, request
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import os
import uuid
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# --- CORS config ---
CORS(app,
     supports_credentials=True,
     origins=[
         "http://localhost:3001",
         "http://127.0.0.1:3001",
         "http://192.168.190.28:3001"  # <-- Add this line
     ])

# --- Flask session config ---
app.secret_key = os.getenv("SESSION_SECRET_KEY", "your-secret-key-here")
app.permanent_session_lifetime = timedelta(days=31)
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_DOMAIN=None  # For development, keep None
)

# --- SQLAlchemy config ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "sqlite:///app.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- User model ---
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(256))  # <-- Add this line
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    auth_provider = db.Column(db.String(50), default="google")

# --- OAuth config ---
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

ALLOWED_DOMAIN = os.getenv("ALLOWED_DOMAIN", "gmail.com")

# --- Routes ---

@app.route('/')
def home():
    return redirect("http://192.168.190.28:3001")

@app.route('/login')
def login():
    redirect_uri = "http://localhost:9095/callback"   # Flask running locally
    return google.authorize_redirect(
        redirect_uri,
        prompt="select_account"
    )

@app.route('/callback')
def callback():
    try:
        token = google.authorize_access_token()
        userinfo = google.userinfo()
    except Exception as e:
        print("Authorization error:", e)
        return f"<h3>Authorization failed: {str(e)}</h3><p><a href='/login'>Try again</a></p>", 400

    email = userinfo.get('email')
    if not email:
        return "<h3>No email found in Google response</h3><p><a href='/login'>Try again</a></p>", 400

    domain = email.split('@')[-1]
    if ALLOWED_DOMAIN and domain != ALLOWED_DOMAIN:
        session.clear()
        return f"""
        <h3>Access denied for domain: {domain}</h3>
        <p>Only {ALLOWED_DOMAIN} is allowed.</p>
        <p><a href="http://localhost:3001">Back to app</a></p>
        """, 403

    session.permanent = True
    session['user'] = {
        "email": email,
        "name": userinfo.get("name"),
        "sub": userinfo.get("sub"),
        "picture": userinfo.get("picture")
    }

    # Save user in DB
    try:
        existing_user = User.query.filter_by(email=email).first()
        if not existing_user:
            new_user = User(
                email=email,
                name=userinfo.get("name", "Unknown User"),
                auth_provider="google"
            )
            db.session.add(new_user)
            db.session.commit()
            print(f"Created new user: {email}")
        else:
            print(f"User already exists: {email}")
    except Exception as e:
        print(f"Database error: {e}")

    return redirect("http://localhost:3001")   # React app

@app.route('/api/user/me')
def get_current_user_api():
    user = session.get('user')
    if not user:
        return jsonify({"message": "Unauthorized", "authenticated": False}), 401

    return jsonify({
        "authenticated": True,
        "email": user.get("email"),
        "name": user.get("name"),
        "google_id": user.get("sub"),
        "picture": user.get("picture")
    })

@app.route('/logout')
def logout():
    session.clear()
    return redirect("http://localhost:3001")

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})

@app.route('/debug/session')
def debug_session():
    if app.debug:
        return jsonify({
            "session_data": dict(session),
            "has_user": 'user' in session,
            "session_id": session.get('_id', 'No session ID')
        })
    return jsonify({"error": "Debug mode disabled"}), 403

# --- Registration endpoint ---
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    name = data.get('name', 'User')
    if not email or not password:
        return jsonify({'message': 'Email and password required'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'Email already registered'}), 409
    user = User(
        email=email,
        name=name,
        password_hash=generate_password_hash(password),
        auth_provider='local'
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'Registration successful'}), 201

# --- Login endpoint ---
@app.route('/api/login', methods=['POST'])
def login_email():
    data = request.json
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    user = User.query.filter_by(email=email, auth_provider='local').first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'message': 'Invalid credentials'}), 401
    session.permanent = True
    session['user'] = {
        "email": user.email,
        "name": user.name,
        "sub": user.id,
        "picture": None
    }
    return jsonify({'message': 'Login successful'})

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created successfully")
        except Exception as e:
            print(f"Database initialization error: {e}")

    print("Starting Flask app on http://localhost:9095")
    print("CORS enabled for: http://localhost:3001")
    print(f"Allowed domain: {ALLOWED_DOMAIN}")

    app.run(host="0.0.0.0", port=9095, debug=True)
