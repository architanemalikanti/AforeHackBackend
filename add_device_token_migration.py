"""
Migration script to add device_token column to users table
Run this with: python add_device_token_migration.py
"""

import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def run_migration():
    """Add device_token column to users table"""
    try:
        # Parse the DATABASE_URL
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Check if column already exists
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='users' AND column_name='device_token';
        """)

        if cursor.fetchone():
            print("✅ device_token column already exists")
        else:
            # Add the column
            cursor.execute("""
                ALTER TABLE users
                ADD COLUMN device_token VARCHAR(255);
            """)
            conn.commit()
            print("✅ Successfully added device_token column to users table")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error running migration: {e}")
        raise

if __name__ == "__main__":
    run_migration()
