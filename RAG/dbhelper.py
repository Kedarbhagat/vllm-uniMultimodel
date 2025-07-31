# db_helper.py

import psycopg2
import uuid
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

# --- Database Configuration ---
DB_CONFIG = {
    "dbname": "LLMNET",
    "user": "postgres",
    "password": "LLMNET",  # ⚠️ Change in production
    "host": "localhost",
    "port": 5432
}

# --- Connection Helper ---
def get_connection():
    return psycopg2.connect(**DB_CONFIG)

# --- User Management ---
def get_or_create_user(email: str, name: str = "Anonymous") -> str:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        result = cursor.fetchone()
        if result:
            return result[0]

        user_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO users (id, email, name) VALUES (%s, %s, %s)",
            (user_id, email, name)
        )
        conn.commit()
        return user_id
    finally:
        cursor.close()
        conn.close()

# --- Thread Management ---
import json 

def create_thread(user_id: str, title: str = "New Chat", metadata: dict = {}) -> str:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        thread_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata)  # ✅ Convert dict to JSON string
        cursor.execute(
            "INSERT INTO threads (id, user_id, title, metadata) VALUES (%s, %s, %s, %s)",
            (thread_id, user_id, title, metadata_json)
        )
        conn.commit()
        return thread_id
    finally:
        cursor.close()
        conn.close()

def get_threads_by_user(user_id: str) -> List[Tuple[str, str]]:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT id, title FROM threads WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

# --- Message Management ---
def add_message(
    thread_id: str,
    role: str,
    content: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None
):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO messages (thread_id, role, content, tool_name, tool_args)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (thread_id, role, content, tool_name, tool_args)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def get_messages_by_thread(thread_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT role, content, tool_name, tool_args, created_at FROM messages WHERE thread_id = %s ORDER BY created_at",
            (thread_id,)
        )
        rows = cursor.fetchall()
        return [
            {
                "role": row[0],
                "content": row[1],
                "tool_name": row[2],
                "tool_args": row[3],
                "created_at": row[4].strftime("%Y-%m-%d %H:%M:%S")
            }
            for row in rows
        ]
    finally:
        cursor.close()
        conn.close()
