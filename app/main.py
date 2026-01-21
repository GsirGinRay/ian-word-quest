import os
import random
import uuid
import shutil
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

print("Starting V12 - Level Unlock System...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"Files in DATA_DIR: {os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else 'DIR NOT FOUND'}")

DB_PATH = os.path.join(DATA_DIR, "ian_quest.db")

# Database Setup
try:
    DB_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except:
    print("DB Fallback to Memory")
    DB_URL = "sqlite:///:memory:"
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

app = FastAPI()

# Models
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
    title = Column(String, index=True)
    difficulty = Column(String, default="Normal")
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

class UserLevelProgress(Base):
    __tablename__ = "user_level_progress"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    level_id = Column(Integer, ForeignKey("level_packs.id"), index=True)
    completed = Column(Boolean, default=False)
    best_score = Column(Integer, default=0)
    completed_at = Column(DateTime, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pre-load default levels from JSON on startup
def init_default_levels():
    db = SessionLocal()
    try:
        # Check if we already have levels
        existing_count = db.query(LevelPack).count()
        print(f"Existing levels in DB: {existing_count}")

        # If we have more than 1 level, assume it's properly initialized
        if existing_count > 1:
            print(f"Already have {existing_count} levels, skipping init")
            return

        # If we only have 1 level (Sample Level), delete it and reload from JSON
        if existing_count == 1:
            print("Only 1 level found (probably Sample Level), will reload from JSON")
            db.query(Word).delete()
            db.query(LevelPack).delete()
            db.commit()

        # Try to load from JSON file (extracted from 英檢.apkg)
        json_path = os.path.join(DATA_DIR, "words_data.json")
        print(f"Looking for JSON at: {json_path}")
        print(f"JSON file exists: {os.path.exists(json_path)}")

        if not os.path.exists(json_path):
            print(f"JSON file not found, creating sample data")
            create_sample_levels(db)
            return

        try:
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                words_data = json.load(f)

            print(f"Loaded {len(words_data)} words from JSON")

            # Split into chunks of 50 words per level
            chunk_size = 50
            for i in range(0, len(words_data), chunk_size):
                chunk = words_data[i:i+chunk_size]
                level_num = (i // chunk_size) + 1
                difficulty = "Easy" if level_num <= 14 else ("Normal" if level_num <= 28 else "Hard")

                pack = LevelPack(title=f"Level {level_num}", difficulty=difficulty)
                db.add(pack)
                db.commit()
                db.refresh(pack)

                for w in chunk:
                    word = Word(
                        pack_id=pack.id,
                        word=w.get("word", ""),
                        meaning=w.get("meaning", "???"),
                        sentence=w.get("sentence", "")
                    )
                    db.add(word)

                db.commit()

            total_levels = (len(words_data) + chunk_size - 1) // chunk_size
            print(f"SUCCESS: Created {total_levels} levels with {len(words_data)} words")

        except Exception as e:
            print(f"ERROR loading JSON: {e}")
            import traceback
            traceback.print_exc()
            create_sample_levels(db)

    finally:
        db.close()

def create_sample_levels(db):
    # Create sample levels with basic words
    sample_words = [
        {"word": "apple", "meaning": "蘋果"},
        {"word": "book", "meaning": "書"},
        {"word": "cat", "meaning": "貓"},
        {"word": "dog", "meaning": "狗"},
        {"word": "egg", "meaning": "蛋"},
        {"word": "fish", "meaning": "魚"},
        {"word": "good", "meaning": "好的"},
        {"word": "happy", "meaning": "快樂的"},
        {"word": "ice", "meaning": "冰"},
        {"word": "jump", "meaning": "跳"},
    ]

    pack = LevelPack(title="Sample Level", difficulty="Easy")
    db.add(pack)
    db.commit()
    db.refresh(pack)

    for w in sample_words:
        word = Word(pack_id=pack.id, word=w["word"], meaning=w["meaning"], sentence="")
        db.add(word)
    db.commit()
    print("Created sample level with 10 words")

# Initialize on startup
init_default_levels()

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
        from_attributes = True

class LevelOut(BaseModel):
    id: int
    title: str
    difficulty: str
    word_count: int
    class Config:
        from_attributes = True

# Routes
@app.get("/")
async def get_index():
    # Read HTML from file to avoid SyntaxError in Python
    html_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse(content="<h1>Error: index.html not found</h1>")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/users", response_model=List[UserOut])
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.post("/api/users", response_model=UserOut)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.name == user.name).first()
    if db_user: return db_user
    new_user = User(name=user.name, avatar=user.avatar)
    db.add(new_user); db.commit(); db.refresh(new_user)
    return new_user

@app.post("/api/users/{user_id}/progress")
def update_progress(user_id: int, xp_gained: int, level_id: int = None, score: int = 0, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user: raise HTTPException(status_code=404, detail="User not found")
    user.xp += xp_gained
    user.level = (user.xp // 100) + 1

    # Mark level as completed if score >= 60% (3 out of 5 correct)
    level_completed = False
    if level_id and score >= 60:
        progress = db.query(UserLevelProgress).filter(
            UserLevelProgress.user_id == user_id,
            UserLevelProgress.level_id == level_id
        ).first()

        if not progress:
            progress = UserLevelProgress(
                user_id=user_id,
                level_id=level_id,
                completed=True,
                best_score=score,
                completed_at=datetime.utcnow()
            )
            db.add(progress)
            level_completed = True
        elif not progress.completed:
            progress.completed = True
            progress.best_score = max(progress.best_score, score)
            progress.completed_at = datetime.utcnow()
            level_completed = True
        elif score > progress.best_score:
            progress.best_score = score

    db.commit()
    return {"xp": user.xp, "level": user.level, "level_completed": level_completed}

@app.get("/api/levels")
def get_levels(user_id: int = None, db: Session = Depends(get_db)):
    packs = db.query(LevelPack).filter(LevelPack.is_active == True).order_by(LevelPack.id.asc()).all()

    # Get user's completed levels
    completed_levels = set()
    if user_id:
        progress = db.query(UserLevelProgress).filter(
            UserLevelProgress.user_id == user_id,
            UserLevelProgress.completed == True
        ).all()
        completed_levels = {p.level_id for p in progress}

    results = []
    for i, p in enumerate(packs):
        # Level is unlocked if: it's the first level OR the previous level is completed
        is_first = (i == 0)
        prev_completed = (i > 0 and packs[i-1].id in completed_levels)
        is_unlocked = is_first or prev_completed or p.id in completed_levels

        results.append({
            "id": p.id,
            "title": p.title,
            "difficulty": p.difficulty,
            "word_count": len(p.words),
            "unlocked": is_unlocked,
            "completed": p.id in completed_levels
        })
    return results

@app.post("/api/admin/upload")
async def upload_level(file: UploadFile = File(...), title: str = Form(...), difficulty: str = Form("Normal"), db: Session = Depends(get_db)):
    try: import pandas as pd
    except ImportError: raise HTTPException(status_code=500, detail="Pandas not installed")
    temp_path = os.path.join(DATA_DIR, f"temp_{uuid.uuid4()}.xlsx")
    with open(temp_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    try:
        df = pd.read_excel(temp_path)
        df.columns = [str(c).lower().strip() for c in df.columns]
        word_col = next((c for c in df.columns if 'word' in c or '單字' in c), None)
        mean_col = next((c for c in df.columns if 'mean' in c or '中文' in c or 'def' in c), None)
        sent_col = next((c for c in df.columns if 'sent' in c or '例句' in c or 'ex' in c), None)
        if not word_col: raise HTTPException(status_code=400, detail="Cannot find 'word' column in Excel")
        new_pack = LevelPack(title=title, difficulty=difficulty)
        db.add(new_pack); db.commit(); db.refresh(new_pack)
        count = 0
        for _, row in df.iterrows():
            if pd.isna(row[word_col]): continue
            w_str = str(row[word_col]).strip()
            if not w_str: continue
            w = Word(pack_id=new_pack.id, word=w_str, meaning=str(row[mean_col]).strip() if mean_col and not pd.isna(row[mean_col]) else "???", sentence=str(row[sent_col]).strip() if sent_col and not pd.isna(row[sent_col]) else f"Sentence about {w_str}.")
            db.add(w); count += 1
        db.commit()
        return {"status": "success", "pack_id": new_pack.id, "words_added": count}
    except Exception as e: db.rollback(); raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.delete("/api/admin/levels/{level_id}")
def delete_level(level_id: int, db: Session = Depends(get_db)):
    pack = db.query(LevelPack).filter(LevelPack.id == level_id).first()
    if pack: db.delete(pack); db.commit()
    return {"status": "deleted"}

@app.get("/api/game/start/{level_id}")
def start_game(level_id: int, count: int = 5, db: Session = Depends(get_db)):
    pack = db.query(LevelPack).filter(LevelPack.id == level_id).first()
    if not pack: raise HTTPException(status_code=404, detail="Level not found")
    words = pack.words
    selected = random.sample(words, min(len(words), count))
    return [{"id": w.id, "word": w.word, "meaning": w.meaning, "sentence": w.sentence} for w in selected]

@app.get("/api/words/all")
def get_all_words(db: Session = Depends(get_db)):
    """Get all words for generating wrong choices"""
    words = db.query(Word).all()
    return [{"id": w.id, "word": w.word, "meaning": w.meaning} for w in words]
