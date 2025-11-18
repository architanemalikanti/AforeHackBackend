"""
Migration script to change current_era to eras array in users table
Run this with: python add_current_era_migration.py
"""

import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def run_migration():
    """Convert current_era to eras array in users table"""
    try:
        # Parse the DATABASE_URL
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Check if eras column already exists
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='users' AND column_name='eras';
        """)

        if cursor.fetchone():
            print("✅ eras column already exists")
        else:
            # Check if old current_era column exists
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='users' AND column_name='current_era';
            """)

            has_old_column = cursor.fetchone() is not None

            if has_old_column:
                # Migrate data from current_era to eras array, then drop old column
                cursor.execute("""
                    ALTER TABLE users
                    ADD COLUMN eras TEXT[];
                """)

                # Copy existing current_era values into eras array
                cursor.execute("""
                    UPDATE users
                    SET eras = ARRAY[current_era]
                    WHERE current_era IS NOT NULL;
                """)

                # Drop old column
                cursor.execute("""
                    ALTER TABLE users
                    DROP COLUMN current_era;
                """)

                print("✅ Migrated current_era to eras array and dropped old column")
            else:
                # Just add the new eras column
                cursor.execute("""
                    ALTER TABLE users
                    ADD COLUMN eras TEXT[];
                """)
                print("✅ Successfully added eras array column to users table")

            conn.commit()

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error running migration: {e}")
        raise

if __name__ == "__main__":
    run_migration()
