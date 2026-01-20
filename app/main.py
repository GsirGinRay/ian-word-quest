import os
import shutil
import uuid
import pandas as pd
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Database Setup (SQLite)
# In Zeabur, you should mount a volume to /app/app/data to keep this persistent
DB_URL = f"sqlite:///{os.path.join(DATA_DIR, 'ian_quest.db')}"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI()

# --- DATABASE MODELS ---

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    avatar = Column(String, default="👦")
    xp = Column(Integer, default=0)
    level = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

class LevelPack(Base):
    __tablename__ = "level_packs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True) # e.g. "Chapter 1: Animals"
    difficulty = Column(String, default="Normal") # Easy, Normal, Hard
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    words = relationship("Word", back_populates="pack", cascade="all, delete-orphan")

class Word(Base):
    __tablename__ = "words"
    id = Column(Integer, primary_key=True, index=True)
    pack_id = Column(Integer, ForeignKey("level_packs.id"))
    word = Column(String, index=True)
    meaning = Column(String)
    sentence = Column(Text)
    pack = relationship("LevelPack", back_populates="words")

# Create Tables
Base.metadata.create_all(bind=engine)

# --- DEPENDENCIES ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Models (for API) ---
class UserCreate(BaseModel):
    name: str
    avatar: str = "👦"

class UserOut(BaseModel):
    id: int
    name: str
    avatar: str
    xp: int
    level: int
    class Config:
        orm_mode = True

class LevelOut(BaseModel):
    id: int
    title: str
    difficulty: str
    word_count: int
    class Config:
        orm_mode = True

# --- API ROUTES ---

@app.get("/")
async def get_index():
    with open(os.path.join(BASE_DIR, "game.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# 1. USER MANAGEMENT
@app.get("/api/users", response_model=List[UserOut])
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.post("/api/users", response_model=UserOut)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.name == user.name).first()
    if db_user:
        return db_user # Return existing if name matches
    new_user = User(name=user.name, avatar=user.avatar)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/api/users/{user_id}/progress")
def update_progress(user_id: int, xp_gained: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.xp += xp_gained
    # Simple level formula: 100 XP per level
    user.level = (user.xp // 100) + 1
    db.commit()
    return {"xp": user.xp, "level": user.level}

# 2. LEVEL MANAGEMENT (ADMIN)
@app.get("/api/levels", response_model=List[LevelOut])
def get_levels(db: Session = Depends(get_db)):
    packs = db.query(LevelPack).filter(LevelPack.is_active == True).order_by(LevelPack.created_at.desc()).all()
    results = []
    for p in packs:
        results.append({
            "id": p.id,
            "title": p.title,
            "difficulty": p.difficulty,
            "word_count": len(p.words)
        })
    return results

@app.post("/api/admin/upload")
async def upload_level(
    file: UploadFile = File(...), 
    title: str = Form(...),
    difficulty: str = Form("Normal"),
    db: Session = Depends(get_db)
):
    # Save temp file
    temp_path = os.path.join(DATA_DIR, f"temp_{uuid.uuid4()}.xlsx")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Parse Excel
        df = pd.read_excel(temp_path)
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        word_col = next((c for c in df.columns if 'word' in c or '單字' in c), None)
        mean_col = next((c for c in df.columns if 'mean' in c or '中文' in c or 'def' in c), None)
        sent_col = next((c for c in df.columns if 'sent' in c or '例句' in c or 'ex' in c), None)

        if not word_col:
            raise HTTPException(status_code=400, detail="Cannot find 'word' column in Excel")

        # Create Pack
        new_pack = LevelPack(title=title, difficulty=difficulty)
        db.add(new_pack)
        db.commit()
        db.refresh(new_pack)

        # Create Words
        count = 0
        for _, row in df.iterrows():
            if pd.isna(row[word_col]): continue
            
            w_str = str(row[word_col]).strip()
            if not w_str: continue

            w = Word(
                pack_id=new_pack.id,
                word=w_str,
                meaning=str(row[mean_col]).strip() if mean_col and not pd.isna(row[mean_col]) else "???",
                sentence=str(row[sent_col]).strip() if sent_col and not pd.isna(row[sent_col]) else f"This is a sentence about {w_str}."
            )
            db.add(w)
            count += 1
        
        db.commit()
        return {"status": "success", "pack_id": new_pack.id, "words_added": count}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.delete("/api/admin/levels/{level_id}")
def delete_level(level_id: int, db: Session = Depends(get_db)):
    pack = db.query(LevelPack).filter(LevelPack.id == level_id).first()
    if pack:
        db.delete(pack)
        db.commit()
    return {"status": "deleted"}

# 3. GAMEPLAY
@app.get("/api/game/start/{level_id}")
def start_game(level_id: int, count: int = 5, db: Session = Depends(get_db)):
    pack = db.query(LevelPack).filter(LevelPack.id == level_id).first()
    if not pack:
        raise HTTPException(status_code=404, detail="Level not found")
    
    words = pack.words
    import random
    selected = random.sample(words, min(len(words), count))
    
    return [
        {"id": w.id, "word": w.word, "meaning": w.meaning, "sentence": w.sentence}
        for w in selected
    ]

# Init default data if empty
@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    if db.query(LevelPack).count() == 0:
        # Create a default pack from existing file if available
        default_file = os.path.join(DATA_DIR, "Ian's English Words-4.xlsx")
        if os.path.exists(default_file):
            print("Loading default Excel file...")
            # Reuse logic? For now simple mock
            pass
    db.close()
