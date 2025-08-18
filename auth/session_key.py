from flask import Flask, redirect, url_for, session, jsonify, render_template_string
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
import os
from datetime import timedelta
# Removed: from flask_sqlalchemy import SQLAlchemy

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- Flask Configuration ---
app.secret_key = os.getenv("SESSION_SECRET_KEY")
# Optional: Set session to be permanent and last for, e.g., 31 days
app.permanent_session_lifetime = timedelta(days=31)

# Removed: SQLAlchemy Configuration and User Model
# app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# Removed: User Database Model
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     google_id = db.Column(db.String(120), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     name = db.Column(db.String(100), nullable=False)
#     picture = db.Column(db.String(255))
#
#     def __repr__(self):
#         return f'<User {self.email}>'

# --- Authlib OAuth Configuration ---
oauth = OAuth(app)

google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
    userinfo_endpoint="https://openidconnect.googleapis.com/v1/userinfo"
)

# Allowed domain for access (e.g., "nmit.ac.in")
ALLOWED_DOMAIN = os.getenv("ALLOWED_DOMAIN")

# --- Routes ---

@app.route('/')
def home():
    """
    Home page that displays login link or user info if logged in.
    """
    user_in_session = session.get('user')
    
    if user_in_session:
        # Display basic user info directly from the session
        return render_template_string("""
            <h1>Welcome, {{ user.get('name') }} ({{ user.get('email') }})</h1>
            <img src='{{ user.get('picture') }}' alt='User Picture' width='100'><br>
            <p><a href='/logout'>Logout</a></p>
            <p>Your API data is at: <a href="/api/user/me" target="_blank">/api/user/me</a></p>
        """, user=user_in_session)
    
    return render_template_string("""
        <h1>Welcome!</h1>
        <p><a href="/login">Login with Google</a></p>
    """)

@app.route('/login')
def login():
    """
    Initiates the Google OAuth login process.
    """
    # Use server IP in redirect URI for Google OAuth
    redirect_uri = "http://172.17.35.82:5000/callback"  # Update port if needed
    return google.authorize_redirect(redirect_uri)

@app.route('/callback')
def callback():
    """
    Handles the callback from Google after authentication.
    Authenticates user and checks domain.
    No database operations here as DB is removed.
    """
    try:
        token = google.authorize_access_token()
    except Exception as e:
        print(f"Authorization error: {e}")
        return "<h3>Authorization failed. Please try again.</h3>", 400

    userinfo = google.userinfo() # Fetch user info from Google

    email = userinfo.get('email')
    if not email:
        return "<h3>Error: Could not retrieve email from Google.</h3>", 400

    domain = email.split('@')[-1]

    if domain != ALLOWED_DOMAIN:
        session.clear() # Clear session for unauthorized domain
        return f"<h3>Access denied for domain: {domain}. Only {ALLOWED_DOMAIN} is allowed.</h3>", 403

    # Database operations removed:
    # user = User.query.filter_by(email=email).first()
    # if not user:
    #     user = User(...)
    #     db.session.add(user)
    #     db.session.commit()
    # else:
    #     user.name = userinfo.get('name')
    #     user.picture = userinfo.get('picture')
    #     db.session.commit()
    
    # Store Google's raw user info in session
    session.permanent = True # Mark session as permanent if desired
    session['user'] = userinfo
    print(f"User logged in (session-only): {email}")
    
    return redirect('/')

@app.route('/api/user/me')
def get_current_user_api():
    """
    API endpoint to get the currently logged-in user's information as JSON.
    Data is fetched directly from the Flask session (no DB).
    """
    user_in_session = session.get('user')
    if not user_in_session: # 'email' check is redundant if user_in_session is None
        return jsonify({"message": "Unauthorized"}), 401 # HTTP 401 for unauthenticated

    # Since there's no DB, we return the info directly from the session
    return jsonify({
        "email": user_in_session.get('email'),
        "name": user_in_session.get('name'),
        "picture": user_in_session.get('picture'),
        "google_id": user_in_session.get('sub') # Google's unique ID
        # Only data available in the session can be returned
    })

@app.route('/logout')
def logout():
    """
    Logs out the user by clearing the Flask session.
    """
    session.clear()
    print("User logged out.")
    return redirect('/')

# --- Application Entry Point ---
if __name__ == '__main__':
    # Run Flask app on server IP so it's accessible from other devices
    app.run(debug=True, host="0.0.0.0", port=5000)