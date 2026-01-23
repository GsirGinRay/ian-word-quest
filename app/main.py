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

print("Starting V13 - Multi-Question Types & Sound System...")

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

class WordProgress(Base):
    """Track user's mastery of individual words"""
    __tablename__ = "word_progress"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    word_id = Column(Integer, ForeignKey("words.id"), index=True)
    correct_count = Column(Integer, default=0)
    wrong_count = Column(Integer, default=0)
    last_seen = Column(DateTime, nullable=True)
    mastery_level = Column(Integer, default=0)  # 0-5 stars

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pre-load default levels from JSON on startup
def download_words_json():
    """Download words_data.json from GitHub if not present locally"""
    json_path = os.path.join(DATA_DIR, "words_data.json")
    if os.path.exists(json_path):
        return json_path

    print("words_data.json not found locally, downloading from GitHub...")
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/GsirGinRay/ian-word-quest/main/app/data/words_data.json"
        urllib.request.urlretrieve(url, json_path)
        print(f"Downloaded words_data.json successfully")
        return json_path
    except Exception as e:
        print(f"Failed to download: {e}")
        return None

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

        # Try to load from JSON file, download if not present
        json_path = download_words_json()
        print(f"Looking for JSON at: {json_path}")
        print(f"JSON file exists: {os.path.exists(json_path) if json_path else False}")

        if not json_path or not os.path.exists(json_path):
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

@app.post("/api/users/{user_id}/avatar")
def update_avatar(user_id: int, avatar: str, db: Session = Depends(get_db)):
    """Update user's avatar emoji"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.avatar = avatar
    db.commit()
    return {"status": "ok", "avatar": avatar}

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

    # Get mastery data for all levels if user is provided
    level_mastery = {}
    if user_id:
        # Get all word IDs by pack
        for pack in packs:
            word_ids = [w.id for w in pack.words]
            if not word_ids:
                level_mastery[pack.id] = {"mastered": 0, "total": 0}
                continue

            progress_list = db.query(WordProgress).filter(
                WordProgress.user_id == user_id,
                WordProgress.word_id.in_(word_ids)
            ).all()

            mastered = sum(1 for p in progress_list if (p.correct_count or 0) >= 3)
            level_mastery[pack.id] = {"mastered": mastered, "total": len(word_ids)}

    results = []
    for i, p in enumerate(packs):
        mastery_info = level_mastery.get(p.id, {"mastered": 0, "total": len(p.words)})
        mastered = mastery_info["mastered"]
        total = mastery_info["total"]
        mastery_percent = int(mastered / total * 100) if total > 0 else 0

        # Level is "completed" if user has mastered >= 80% of words
        is_completed = mastery_percent >= 80

        # Level is unlocked if: it's the first level OR the previous level is completed
        is_first = (i == 0)
        prev_completed = (i > 0 and level_mastery.get(packs[i-1].id, {}).get("mastered", 0) >=
                         int(level_mastery.get(packs[i-1].id, {}).get("total", 1) * 0.8))
        is_unlocked = is_first or prev_completed or is_completed

        results.append({
            "id": p.id,
            "title": p.title,
            "difficulty": p.difficulty,
            "word_count": total,
            "unlocked": is_unlocked,
            "completed": is_completed,
            "mastered": mastered,
            "mastery_percent": mastery_percent
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
def start_game(level_id: int, user_id: int = None, count: int = 5, db: Session = Depends(get_db)):
    """Start game with smart word selection based on user's mastery"""
    pack = db.query(LevelPack).filter(LevelPack.id == level_id).first()
    if not pack: raise HTTPException(status_code=404, detail="Level not found")

    all_words = pack.words
    if not user_id:
        # No user - just random selection
        selected = random.sample(all_words, min(len(all_words), count))
        return [{"id": w.id, "word": w.word, "meaning": w.meaning, "sentence": w.sentence} for w in selected]

    # Get user's progress for all words in this level
    word_ids = [w.id for w in all_words]
    progress_list = db.query(WordProgress).filter(
        WordProgress.user_id == user_id,
        WordProgress.word_id.in_(word_ids)
    ).all()
    progress_map = {p.word_id: p for p in progress_list}

    # Categorize words
    weak_words = []      # Got wrong, need practice
    unseen_words = []    # Never seen
    learning_words = []  # Seen but not mastered (< 3 correct)
    mastered_words = []  # Mastered (>= 3 correct)

    for w in all_words:
        p = progress_map.get(w.id)
        if not p:
            unseen_words.append(w)
        elif (p.wrong_count or 0) > (p.correct_count or 0):
            weak_words.append(w)
        elif (p.correct_count or 0) >= 3:
            mastered_words.append(w)
        else:
            learning_words.append(w)

    # Smart selection priority:
    # 1. Weak words (got wrong) - 40%
    # 2. Unseen words (new) - 30%
    # 3. Learning words (in progress) - 20%
    # 4. Mastered words (review) - 10%
    selected = []

    # Add weak words first (up to 40%)
    random.shuffle(weak_words)
    selected.extend(weak_words[:max(1, int(count * 0.4))])

    # Add unseen words (up to 30%)
    random.shuffle(unseen_words)
    remaining = count - len(selected)
    selected.extend(unseen_words[:max(1, min(remaining, int(count * 0.3)))])

    # Add learning words (up to 20%)
    random.shuffle(learning_words)
    remaining = count - len(selected)
    selected.extend(learning_words[:min(remaining, int(count * 0.2))])

    # Fill remaining with mastered words or any available
    remaining = count - len(selected)
    if remaining > 0:
        random.shuffle(mastered_words)
        selected.extend(mastered_words[:remaining])

    # If still not enough, add from any category
    remaining = count - len(selected)
    if remaining > 0:
        all_remaining = [w for w in all_words if w not in selected]
        random.shuffle(all_remaining)
        selected.extend(all_remaining[:remaining])

    # Shuffle final selection
    random.shuffle(selected)

    return [{"id": w.id, "word": w.word, "meaning": w.meaning, "sentence": w.sentence} for w in selected]

@app.get("/api/words/all")
def get_all_words(db: Session = Depends(get_db)):
    """Get all words for generating wrong choices"""
    words = db.query(Word).all()
    return [{"id": w.id, "word": w.word, "meaning": w.meaning} for w in words]

@app.post("/api/words/answer")
def record_word_answer(user_id: int, word_id: int, correct: bool, db: Session = Depends(get_db)):
    """Record user's answer for a word to track mastery"""
    progress = db.query(WordProgress).filter(
        WordProgress.user_id == user_id,
        WordProgress.word_id == word_id
    ).first()

    if not progress:
        progress = WordProgress(
            user_id=user_id,
            word_id=word_id,
            correct_count=0,
            wrong_count=0,
            mastery_level=0
        )
        db.add(progress)
        db.flush()  # Ensure defaults are set

    if correct:
        progress.correct_count = (progress.correct_count or 0) + 1
    else:
        progress.wrong_count = (progress.wrong_count or 0) + 1

    progress.last_seen = datetime.utcnow()

    # Calculate mastery level (0-5 stars)
    correct = progress.correct_count or 0
    wrong = progress.wrong_count or 0
    total = correct + wrong
    if total >= 3:
        accuracy = correct / total
        progress.mastery_level = min(5, int(accuracy * 6))

    db.commit()
    return {"mastery_level": progress.mastery_level or 0, "correct": correct, "wrong": wrong}

@app.get("/api/words/weak")
def get_weak_words(user_id: int, limit: int = 10, db: Session = Depends(get_db)):
    """Get words the user frequently gets wrong (weak words for review)"""
    # Get words with wrong_count > correct_count or low mastery
    weak_progress = db.query(WordProgress).filter(
        WordProgress.user_id == user_id,
        WordProgress.wrong_count > 0
    ).order_by(
        (WordProgress.wrong_count - WordProgress.correct_count).desc()
    ).limit(limit).all()

    word_ids = [wp.word_id for wp in weak_progress]
    if not word_ids:
        return []

    words = db.query(Word).filter(Word.id.in_(word_ids)).all()
    return [{"id": w.id, "word": w.word, "meaning": w.meaning, "sentence": w.sentence} for w in words]

@app.get("/api/words/progress")
def get_word_progress(user_id: int, word_ids: str, db: Session = Depends(get_db)):
    """Get mastery progress for specific words"""
    ids = [int(x) for x in word_ids.split(",") if x.strip()]
    progress = db.query(WordProgress).filter(
        WordProgress.user_id == user_id,
        WordProgress.word_id.in_(ids)
    ).all()
    return {str(p.word_id): {"mastery": p.mastery_level, "correct": p.correct_count, "wrong": p.wrong_count} for p in progress}

@app.get("/api/levels/{level_id}/mastery")
def get_level_mastery(level_id: int, user_id: int, db: Session = Depends(get_db)):
    """Get user's mastery stats for a specific level"""
    pack = db.query(LevelPack).filter(LevelPack.id == level_id).first()
    if not pack:
        raise HTTPException(status_code=404, detail="Level not found")

    word_ids = [w.id for w in pack.words]
    total_words = len(word_ids)

    if total_words == 0:
        return {"total": 0, "mastered": 0, "learning": 0, "unseen": 0, "weak": 0, "mastery_percent": 0}

    progress_list = db.query(WordProgress).filter(
        WordProgress.user_id == user_id,
        WordProgress.word_id.in_(word_ids)
    ).all()

    mastered = 0  # correct >= 3
    learning = 0  # seen but correct < 3
    weak = 0      # wrong > correct
    seen_ids = set()

    for p in progress_list:
        seen_ids.add(p.word_id)
        correct = p.correct_count or 0
        wrong = p.wrong_count or 0
        if wrong > correct:
            weak += 1
        elif correct >= 3:
            mastered += 1
        else:
            learning += 1

    unseen = total_words - len(seen_ids)

    return {
        "total": total_words,
        "mastered": mastered,
        "learning": learning,
        "unseen": unseen,
        "weak": weak,
        "mastery_percent": int(mastered / total_words * 100) if total_words > 0 else 0
    }
