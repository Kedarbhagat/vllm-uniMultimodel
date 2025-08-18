# db_helper.py (updated to match schema exactly)

import psycopg2
import uuid
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
import json

# --- Database Configuration ---
DB_CONFIG = {
    "dbname": "LLMNET",
    "user": "postgres",
    "password": "LLMNET",
    "host": "localhost",
    "port": 5432
}

def get_connection():
    """Get a new database connection"""
    return psycopg2.connect(**DB_CONFIG)

# --- User Operations ---
def get_user_by_email(email: str) -> str:
    """Fetch or create a user by email. Always return user_id (UUID)."""
    conn = get_connection()
    cursor = conn.cursor()
    try: 
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        row = cursor.fetchone()
        if row:
            return row[0]  # just the UUID

        # create new user if not found
        user_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO users (id, email, name) VALUES (%s, %s, %s)",
            (user_id, email, email.split('@')[0])  # default name from email
        )
        conn.commit()
        return user_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

# --- Thread Operations ---
def create_thread(user_id: str, title: str = "New Chat", metadata: dict = {}) -> str:
    """Create a new conversation thread"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        thread_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata)
        
        cursor.execute("""
            INSERT INTO threads (id, user_id, title, metadata)
            VALUES (%s, %s, %s, %s)
        """, (thread_id, user_id, title, metadata_json))
        conn.commit()
        return thread_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def get_threads_by_user(user_id: str) -> List[Tuple[str, str]]:
    """Get all threads for a user"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, title,created_at 
            FROM threads 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """, (user_id,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def update_thread_title(thread_id: str, new_title: str) -> None:
    """Update the title of a thread."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE threads SET title = %s WHERE id = %s",
            (new_title, thread_id)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

# --- Message Operations ---
def add_message(
    thread_id: str,
    role: str,
    content: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None
) -> None:
    """Add a new message to a thread"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        tool_args_json = json.dumps(tool_args) if tool_args else None
        
        cursor.execute("""
            INSERT INTO messages (thread_id, role, content, tool_name, tool_args)
            VALUES (%s, %s, %s, %s, %s)
        """, (thread_id, role, content, tool_name, tool_args_json))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def get_messages_by_thread(thread_id: str) -> List[Dict[str, Any]]:
    """Get all messages for a thread"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 
                role, 
                content, 
                tool_name, 
                tool_args, 
                created_at 
            FROM messages 
            WHERE thread_id = %s 
            ORDER BY created_at
        """, (thread_id,))
        
        rows = cursor.fetchall()
        return [
            {
                "role": row[0],
                "content": row[1],
                "tool_name": row[2],
                "tool_args": json.loads(row[3]) if row[3] else None,
                "created_at": row[4].strftime("%Y-%m-%d %H:%M:%S")
            }
            for row in rows
        ]
    except Exception as e:
        raise e
    finally:
        cursor.close()
        conn.close()