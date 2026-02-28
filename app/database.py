'''What It Does

Connects FastAPI to PostgreSQL (Supabase).

Why We Need It

Our project needs to:

Store users

Store processed emails

Store importance score

Store summary

Track deleted emails

Without DB → everything disappears when server restarts.'''


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit = False,
    autoflush = False,
    bind = engine
)

Basw = declarative_base()