from .db import Base
from datetime import date, datetime
from sqlalchemy import Column, String, Boolean, Date, DateTime
from sqlalchemy.dialects.postgresql import JSONB, UUID
import uuid

class User(Base):
    __tablename__ = 'users'

    # Match actual database schema
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    name = Column(String(100), nullable=False)  # First name

    # Onboarding fields
    session_id = Column(String(255), unique=True, nullable=True)
    password = Column(String(255), nullable=True)
    birthday = Column(Date, nullable=True)
    gender = Column(String(50), nullable=True)
    sexuality = Column(String(50), nullable=True)
    ethnicity = Column(String(100), nullable=True)
    pronouns = Column(String(50), nullable=True)
    university = Column(String(200), nullable=True)
    college_major = Column(String(200), nullable=True)
    occupation = Column(String(200), nullable=True)
    conversations = Column(JSONB, default=list)  # Array of conversation dicts
    prompt = Column(String, nullable=True)  # Store the dynamic prompt state for user

