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

print("Starting V10 Monster Battle Edition...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Delete old database to force re-import with correct encoding
DB_PATH = os.path.join(DATA_DIR, "ian_quest.db")
if os.path.exists(DB_PATH):
    try:
        os.remove(DB_PATH)
        print("Removed old database for fresh start")
    except:
        pass

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

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pre-load default levels from Excel on startup
def init_default_levels():
    db = SessionLocal()
    try:
        # Check if we already have levels
        existing = db.query(LevelPack).first()
        if existing:
            print(f"Levels already exist, skipping init")
            return

        # Try to load from Excel file
        excel_path = os.path.join(DATA_DIR, "Ian's English Words-4.xlsx")
        if not os.path.exists(excel_path):
            print(f"Excel file not found at {excel_path}, creating sample data")
            create_sample_levels(db)
            return

        try:
            import pandas as pd
            df = pd.read_excel(excel_path)

            # Get columns (handle encoding issues)
            cols = list(df.columns)
            word_col = cols[0] if len(cols) > 0 else None
            mean_col = cols[1] if len(cols) > 1 else None

            if not word_col:
                print("Cannot find word column")
                create_sample_levels(db)
                return

            # Split into chunks of 20 words per level
            words_data = []
            for _, row in df.iterrows():
                word = str(row[word_col]).strip() if pd.notna(row[word_col]) else ""
                meaning = str(row[mean_col]).strip() if mean_col and pd.notna(row[mean_col]) else "???"
                if word and word != "nan":
                    words_data.append({"word": word, "meaning": meaning})

            # Create level packs (chunks of 20)
            chunk_size = 20
            for i in range(0, len(words_data), chunk_size):
                chunk = words_data[i:i+chunk_size]
                level_num = (i // chunk_size) + 1
                difficulty = "Easy" if level_num <= 10 else ("Normal" if level_num <= 20 else "Hard")

                pack = LevelPack(title=f"Level {level_num}", difficulty=difficulty)
                db.add(pack)
                db.commit()
                db.refresh(pack)

                for w in chunk:
                    word = Word(pack_id=pack.id, word=w["word"], meaning=w["meaning"], sentence="")
                    db.add(word)

                db.commit()

            print(f"Loaded {len(words_data)} words into {(len(words_data) + chunk_size - 1) // chunk_size} levels")

        except ImportError:
            print("Pandas not available, creating sample data")
            create_sample_levels(db)
        except Exception as e:
            print(f"Error loading Excel: {e}")
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
def update_progress(user_id: int, xp_gained: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user: raise HTTPException(status_code=404, detail="User not found")
    user.xp += xp_gained
    user.level = (user.xp // 100) + 1
    db.commit()
    return {"xp": user.xp, "level": user.level}

@app.get("/api/levels", response_model=List[LevelOut])
def get_levels(db: Session = Depends(get_db)):
    packs = db.query(LevelPack).filter(LevelPack.is_active == True).order_by(LevelPack.created_at.desc()).all()
    results = []
    for p in packs:
        results.append({"id": p.id, "title": p.title, "difficulty": p.difficulty, "word_count": len(p.words)})
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
