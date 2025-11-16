#!/usr/bin/env python3
"""
Script to create database tables from SQLAlchemy models.
Run this once to initialize your database schema.
"""
from database.db import engine, Base
from database.models import User, Design, Follow, FollowRequest

if __name__ == "__main__":
    # Drop all tables (to reset schema with new columns)
    Base.metadata.drop_all(bind=engine)
    print("🗑️  Dropped all existing tables")

    # Create all tables with updated schema
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully!")

    # Print table names
    print("\n📋 Created tables:")
    for table in Base.metadata.sorted_tables:
        print(f"   - {table.name}")

