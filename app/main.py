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

# ========== READING COMPREHENSION MODELS ==========
class ReadingPassage(Base):
    """Reading passages for comprehension tests"""
    __tablename__ = "reading_passages"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)  # The passage text
    lexile_level = Column(Integer, default=350)  # Lexile measure (350-700)
    difficulty = Column(String, default="beginner")  # beginner, intermediate, advanced
    vocabulary = Column(Text)  # JSON list of key vocabulary words
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    questions = relationship("ReadingQuestion", back_populates="passage", cascade="all, delete-orphan")

class ReadingQuestion(Base):
    """Questions for reading passages"""
    __tablename__ = "reading_questions"
    id = Column(Integer, primary_key=True, index=True)
    passage_id = Column(Integer, ForeignKey("reading_passages.id"), index=True)
    question_type = Column(String, default="comprehension")  # comprehension, vocabulary, inference, main_idea
    question = Column(Text)
    option_a = Column(String)
    option_b = Column(String)
    option_c = Column(String)
    option_d = Column(String, nullable=True)  # Optional 4th choice
    correct_answer = Column(String)  # 'A', 'B', 'C', or 'D'
    explanation = Column(Text, nullable=True)  # Why this answer is correct
    passage = relationship("ReadingPassage", back_populates="questions")

class UserReadingProgress(Base):
    """Track user's reading comprehension progress"""
    __tablename__ = "user_reading_progress"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    passage_id = Column(Integer, ForeignKey("reading_passages.id"), index=True)
    score = Column(Integer, default=0)  # Percentage correct
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime, nullable=True)
    time_spent = Column(Integer, default=0)  # Seconds spent reading

class UserReadingLevel(Base):
    """Track user's overall reading level"""
    __tablename__ = "user_reading_level"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, index=True)
    current_lexile = Column(Integer, default=350)  # Current estimated Lexile
    passages_completed = Column(Integer, default=0)
    total_correct = Column(Integer, default=0)
    total_questions = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow)

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

@app.delete("/api/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Delete a user and all their progress"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete user's word progress
    db.query(WordProgress).filter(WordProgress.user_id == user_id).delete()
    # Delete user's level progress
    db.query(UserLevelProgress).filter(UserLevelProgress.user_id == user_id).delete()
    # Delete user
    db.delete(user)
    db.commit()
    return {"status": "ok", "message": f"User {user.name} deleted"}

@app.post("/api/users/{user_id}/progress")
def update_progress(user_id: int, xp_gained: int, level_id: int = None, score: int = 0, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user: raise HTTPException(status_code=404, detail="User not found")
    user.xp += xp_gained
    # NEW LEVEL FORMULA: sqrt(XP/25) + 1
    # Level 1: 0 XP, Level 2: 25 XP, Level 3: 100 XP, Level 5: 400 XP, Level 10: 2025 XP
    import math
    user.level = int(math.sqrt(user.xp / 25)) + 1

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

            # Mastery = 2 correct answers (was 3, now faster)
            mastered = sum(1 for p in progress_list if (p.correct_count or 0) >= 2)
            level_mastery[pack.id] = {"mastered": mastered, "total": len(word_ids)}

    results = []
    for i, p in enumerate(packs):
        mastery_info = level_mastery.get(p.id, {"mastered": 0, "total": len(p.words)})
        mastered = mastery_info["mastered"]
        total = mastery_info["total"]
        mastery_percent = int(mastered / total * 100) if total > 0 else 0

        # Level is "completed" if user has mastered >= 40% of words
        # With 50 words, need 20 mastered = achievable in ~8 perfect battles
        is_completed = mastery_percent >= 40

        # Level is unlocked if: it's the first level OR the previous level is completed (40%)
        is_first = (i == 0)
        prev_completed = (i > 0 and level_mastery.get(packs[i-1].id, {}).get("mastered", 0) >=
                         int(level_mastery.get(packs[i-1].id, {}).get("total", 1) * 0.4))
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

    # Categorize words (mastery = 2 correct answers)
    weak_words = []      # Got wrong more than correct, need practice
    unseen_words = []    # Never seen - TOP PRIORITY for learning
    learning_words = []  # Seen but not mastered (1 correct)
    mastered_words = []  # Mastered (>= 2 correct)

    for w in all_words:
        p = progress_map.get(w.id)
        if not p:
            unseen_words.append(w)
        elif (p.wrong_count or 0) > (p.correct_count or 0):
            weak_words.append(w)
        elif (p.correct_count or 0) >= 2:  # Mastery = 2 correct
            mastered_words.append(w)
        else:
            learning_words.append(w)

    # NEW PRIORITY: Focus on learning NEW words first!
    # 1. Unseen words (NEW!) - 50% priority - this is where learning happens
    # 2. Learning words (almost mastered) - 30% - reinforce
    # 3. Weak words (struggling) - 15% - extra practice
    # 4. Mastered words (review) - 5% - minimal review
    selected = []
    selected_ids = set()

    def add_words(word_list, max_count):
        """Add words without duplicates"""
        random.shuffle(word_list)
        added = 0
        for w in word_list:
            if w.id not in selected_ids and added < max_count:
                selected.append(w)
                selected_ids.add(w.id)
                added += 1

    # Priority 1: NEW unseen words (50% = 2-3 words out of 5)
    add_words(unseen_words, max(2, int(count * 0.5)))

    # Priority 2: Learning words - almost there! (30%)
    add_words(learning_words, max(1, int(count * 0.3)))

    # Priority 3: Weak words - need practice (15%)
    add_words(weak_words, max(1, int(count * 0.15)))

    # Priority 4: Mastered - minimal review (5%)
    if len(selected) < count:
        add_words(mastered_words, 1)

    # Fill remaining if needed (from any non-mastered first)
    remaining = count - len(selected)
    if remaining > 0:
        all_unmastered = [w for w in all_words if w.id not in selected_ids and w not in mastered_words]
        add_words(all_unmastered, remaining)

    # Last resort: add mastered words
    remaining = count - len(selected)
    if remaining > 0:
        leftover = [w for w in all_words if w.id not in selected_ids]
        add_words(leftover, remaining)

    # Shuffle final selection
    random.shuffle(selected)

    return [{"id": w.id, "word": w.word, "meaning": w.meaning, "sentence": w.sentence} for w in selected]

@app.get("/api/words/all")
def get_all_words(db: Session = Depends(get_db)):
    """Get all words for generating wrong choices"""
    words = db.query(Word).all()
    return [{"id": w.id, "word": w.word, "meaning": w.meaning} for w in words]

@app.get("/api/levels/{level_id}/words")
def get_level_words(level_id: int, user_id: int = None, db: Session = Depends(get_db)):
    """Get all words in a level for browsing/preview"""
    pack = db.query(LevelPack).filter(LevelPack.id == level_id).first()
    if not pack:
        raise HTTPException(status_code=404, detail="Level not found")

    words = pack.words

    # Get user progress if provided
    progress_map = {}
    if user_id:
        word_ids = [w.id for w in words]
        progress_list = db.query(WordProgress).filter(
            WordProgress.user_id == user_id,
            WordProgress.word_id.in_(word_ids)
        ).all()
        progress_map = {p.word_id: p for p in progress_list}

    result = []
    for w in words:
        p = progress_map.get(w.id)
        result.append({
            "id": w.id,
            "word": w.word,
            "meaning": w.meaning,
            "sentence": w.sentence or "",
            "correct_count": p.correct_count if p else 0,
            "wrong_count": p.wrong_count if p else 0,
            "mastered": (p.correct_count or 0) >= 2 if p else False
        })

    return {
        "level_id": level_id,
        "level_title": pack.title,
        "total_words": len(result),
        "words": result
    }

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

# ========== READING COMPREHENSION API ==========

@app.get("/api/reading/passages")
def get_passages(user_id: int = None, db: Session = Depends(get_db)):
    """Get all reading passages with user progress"""
    passages = db.query(ReadingPassage).filter(ReadingPassage.is_active == True).order_by(ReadingPassage.lexile_level.asc()).all()

    # Get user progress if provided
    progress_map = {}
    user_level = None
    if user_id:
        progress_list = db.query(UserReadingProgress).filter(UserReadingProgress.user_id == user_id).all()
        progress_map = {p.passage_id: p for p in progress_list}
        user_level = db.query(UserReadingLevel).filter(UserReadingLevel.user_id == user_id).first()

    result = []
    for p in passages:
        prog = progress_map.get(p.id)
        result.append({
            "id": p.id,
            "title": p.title,
            "lexile_level": p.lexile_level,
            "difficulty": p.difficulty,
            "question_count": len(p.questions),
            "completed": prog.completed if prog else False,
            "score": prog.score if prog else 0,
            "unlocked": True  # For now, all passages are unlocked
        })

    return {
        "passages": result,
        "user_lexile": user_level.current_lexile if user_level else 350,
        "passages_completed": user_level.passages_completed if user_level else 0
    }

@app.get("/api/reading/passages/{passage_id}")
def get_passage_detail(passage_id: int, db: Session = Depends(get_db)):
    """Get a single passage with its questions"""
    passage = db.query(ReadingPassage).filter(ReadingPassage.id == passage_id).first()
    if not passage:
        raise HTTPException(status_code=404, detail="Passage not found")

    import json
    vocab = []
    try:
        vocab = json.loads(passage.vocabulary) if passage.vocabulary else []
    except:
        vocab = []

    questions = []
    for q in passage.questions:
        questions.append({
            "id": q.id,
            "type": q.question_type,
            "question": q.question,
            "options": {
                "A": q.option_a,
                "B": q.option_b,
                "C": q.option_c,
                "D": q.option_d
            } if q.option_d else {
                "A": q.option_a,
                "B": q.option_b,
                "C": q.option_c
            },
            "explanation": q.explanation
        })

    return {
        "id": passage.id,
        "title": passage.title,
        "content": passage.content,
        "lexile_level": passage.lexile_level,
        "difficulty": passage.difficulty,
        "vocabulary": vocab,
        "questions": questions
    }

@app.post("/api/reading/passages/{passage_id}/submit")
def submit_reading_answers(passage_id: int, user_id: int, answers: dict, db: Session = Depends(get_db)):
    """Submit answers for a reading passage and update progress"""
    passage = db.query(ReadingPassage).filter(ReadingPassage.id == passage_id).first()
    if not passage:
        raise HTTPException(status_code=404, detail="Passage not found")

    # Calculate score
    correct = 0
    total = len(passage.questions)
    results = []

    for q in passage.questions:
        user_answer = answers.get(str(q.id), "")
        is_correct = user_answer.upper() == q.correct_answer.upper()
        if is_correct:
            correct += 1
        results.append({
            "question_id": q.id,
            "correct": is_correct,
            "correct_answer": q.correct_answer,
            "user_answer": user_answer,
            "explanation": q.explanation
        })

    score = int(correct / total * 100) if total > 0 else 0

    # Update user progress for this passage
    progress = db.query(UserReadingProgress).filter(
        UserReadingProgress.user_id == user_id,
        UserReadingProgress.passage_id == passage_id
    ).first()

    if not progress:
        progress = UserReadingProgress(user_id=user_id, passage_id=passage_id)
        db.add(progress)

    progress.score = max(progress.score, score)  # Keep best score
    progress.completed = True
    progress.completed_at = datetime.utcnow()
    db.commit()

    # Update user's overall reading level
    user_level = db.query(UserReadingLevel).filter(UserReadingLevel.user_id == user_id).first()
    if not user_level:
        user_level = UserReadingLevel(user_id=user_id)
        db.add(user_level)

    user_level.passages_completed += 1
    user_level.total_correct += correct
    user_level.total_questions += total

    # Adjust Lexile based on performance
    if score >= 80 and user_level.current_lexile < passage.lexile_level + 50:
        user_level.current_lexile = min(700, user_level.current_lexile + 25)
    elif score < 50 and user_level.current_lexile > passage.lexile_level - 50:
        user_level.current_lexile = max(300, user_level.current_lexile - 15)

    user_level.last_updated = datetime.utcnow()
    db.commit()

    # Calculate XP
    xp_earned = correct * 15 + (20 if score == 100 else 0)

    return {
        "score": score,
        "correct": correct,
        "total": total,
        "results": results,
        "xp_earned": xp_earned,
        "new_lexile": user_level.current_lexile
    }

@app.get("/api/reading/user/{user_id}/level")
def get_user_reading_level(user_id: int, db: Session = Depends(get_db)):
    """Get user's current reading level and stats"""
    user_level = db.query(UserReadingLevel).filter(UserReadingLevel.user_id == user_id).first()
    if not user_level:
        return {"current_lexile": 350, "passages_completed": 0, "accuracy": 0}

    accuracy = int(user_level.total_correct / user_level.total_questions * 100) if user_level.total_questions > 0 else 0
    return {
        "current_lexile": user_level.current_lexile,
        "passages_completed": user_level.passages_completed,
        "total_correct": user_level.total_correct,
        "total_questions": user_level.total_questions,
        "accuracy": accuracy
    }

# ========== INITIALIZE SAMPLE READING PASSAGES ==========
def init_sample_passages():
    """Create sample reading passages if none exist"""
    db = SessionLocal()
    try:
        existing = db.query(ReadingPassage).count()
        if existing > 0:
            print(f"Already have {existing} reading passages, skipping init")
            return

        import json

        # Sample passages at different Lexile levels
        sample_passages = [
            # Level 1: Lexile 350L - Very simple
            {
                "title": "The Red Ball",
                "content": """Tom has a ball. The ball is red. Tom likes his ball very much.

Tom plays with the ball every day. He throws it up. He catches it.

One day, the ball goes over the fence. Tom is sad. His friend Amy helps him. Amy gets the ball back.

Tom says, "Thank you, Amy!" Now Tom and Amy play together.""",
                "lexile_level": 350,
                "difficulty": "beginner",
                "vocabulary": json.dumps([
                    {"word": "ball", "meaning": "球"},
                    {"word": "throws", "meaning": "丟、投"},
                    {"word": "catches", "meaning": "接住"},
                    {"word": "fence", "meaning": "籬笆"},
                    {"word": "together", "meaning": "一起"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "What color is Tom's ball?", "a": "Red", "b": "Blue", "c": "Green", "correct": "A", "explain": "The story says 'The ball is red.'"},
                    {"type": "comprehension", "q": "What does Tom do with the ball every day?", "a": "He sleeps with it", "b": "He plays with it", "c": "He eats it", "correct": "B", "explain": "The story says 'Tom plays with the ball every day.'"},
                    {"type": "comprehension", "q": "Where does the ball go?", "a": "Under the bed", "b": "Into the water", "c": "Over the fence", "correct": "C", "explain": "The story says 'the ball goes over the fence.'"},
                    {"type": "comprehension", "q": "Who helps Tom get the ball?", "a": "His mom", "b": "Amy", "c": "His dad", "correct": "B", "explain": "The story says 'His friend Amy helps him.'"}
                ]
            },
            # Level 2: Lexile 400L
            {
                "title": "My Pet Dog",
                "content": """I have a pet dog named Max. Max is a golden retriever. He has soft, yellow fur and big brown eyes.

Every morning, I take Max for a walk in the park. He loves to run and chase birds. Sometimes he jumps into the pond to swim!

Max is also very smart. He can do many tricks. He can sit, shake hands, and roll over. When I say "fetch," he runs to get his ball.

At night, Max sleeps next to my bed. He keeps me safe. Max is my best friend.""",
                "lexile_level": 400,
                "difficulty": "beginner",
                "vocabulary": json.dumps([
                    {"word": "retriever", "meaning": "獵犬（品種）"},
                    {"word": "fur", "meaning": "毛皮"},
                    {"word": "chase", "meaning": "追逐"},
                    {"word": "pond", "meaning": "池塘"},
                    {"word": "tricks", "meaning": "把戲"},
                    {"word": "fetch", "meaning": "去撿回來"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "What kind of dog is Max?", "a": "A poodle", "b": "A golden retriever", "c": "A bulldog", "correct": "B", "explain": "The story says 'Max is a golden retriever.'"},
                    {"type": "comprehension", "q": "What does Max like to chase in the park?", "a": "Cats", "b": "Cars", "c": "Birds", "correct": "C", "explain": "The story says 'He loves to run and chase birds.'"},
                    {"type": "vocabulary", "q": "What does 'fetch' mean in this story?", "a": "To sleep", "b": "To get something and bring it back", "c": "To eat food", "correct": "B", "explain": "'Fetch' means to go get something and bring it back."},
                    {"type": "inference", "q": "Why does the writer say Max keeps them safe?", "a": "Max is a guard dog", "b": "Max sleeps next to the bed at night", "c": "Max is very big", "correct": "B", "explain": "Dogs sleeping nearby can alert their owners to danger."}
                ]
            },
            # Level 3: Lexile 450L
            {
                "title": "The School Library",
                "content": """The school library is my favorite place. It is a large room with many bookshelves. There are thousands of books about different topics.

Mrs. Chen is the librarian. She helps students find books and answers our questions. She always has good book recommendations.

I visit the library twice a week. I like to sit by the window and read. The chairs there are very comfortable. Sometimes I do my homework there because it is quiet.

Last week, I discovered a series about space exploration. The books describe astronauts traveling to Mars. I have already read three books in the series. I cannot wait to read the rest!""",
                "lexile_level": 450,
                "difficulty": "intermediate",
                "vocabulary": json.dumps([
                    {"word": "library", "meaning": "圖書館"},
                    {"word": "bookshelves", "meaning": "書架"},
                    {"word": "librarian", "meaning": "圖書館員"},
                    {"word": "recommendations", "meaning": "推薦"},
                    {"word": "discovered", "meaning": "發現"},
                    {"word": "exploration", "meaning": "探索"},
                    {"word": "astronauts", "meaning": "太空人"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "How often does the writer visit the library?", "a": "Every day", "b": "Once a week", "c": "Twice a week", "correct": "C", "explain": "The story says 'I visit the library twice a week.'"},
                    {"type": "comprehension", "q": "Who is Mrs. Chen?", "a": "A teacher", "b": "The librarian", "c": "A student", "correct": "B", "explain": "The story says 'Mrs. Chen is the librarian.'"},
                    {"type": "vocabulary", "q": "What does 'recommendations' mean?", "a": "Rules to follow", "b": "Suggestions about what is good", "c": "Homework assignments", "correct": "B", "explain": "Recommendations are suggestions about things that might be good or useful."},
                    {"type": "inference", "q": "Why does the writer do homework in the library?", "a": "Because it is quiet there", "b": "Because Mrs. Chen tells them to", "c": "Because there are no chairs at home", "correct": "A", "explain": "The story says 'I do my homework there because it is quiet.'"},
                    {"type": "main_idea", "q": "What is this story mainly about?", "a": "How to become an astronaut", "b": "The writer's love for the school library", "c": "Mrs. Chen's job", "correct": "B", "explain": "The whole story is about why the writer loves the school library."}
                ]
            },
            # Level 4: Lexile 500L
            {
                "title": "Weather Around the World",
                "content": """Weather is different in every part of the world. Some places are hot all year long, while others have cold winters with lots of snow.

Near the equator, countries like Brazil and Indonesia have tropical weather. It is warm and humid there, and it rains frequently. Tropical rainforests grow in these regions because of all the rain.

In contrast, countries like Canada and Russia have very cold winters. Temperatures can drop below minus thirty degrees Celsius! People who live there must wear heavy coats and boots to stay warm.

Some places have four distinct seasons: spring, summer, autumn, and winter. Each season brings different weather. In spring, flowers bloom. Summer is hot and sunny. Autumn brings colorful leaves. Winter is cold and sometimes snowy.

Scientists who study weather are called meteorologists. They use special tools to predict what the weather will be like tomorrow or next week.""",
                "lexile_level": 500,
                "difficulty": "intermediate",
                "vocabulary": json.dumps([
                    {"word": "equator", "meaning": "赤道"},
                    {"word": "tropical", "meaning": "熱帶的"},
                    {"word": "humid", "meaning": "潮濕的"},
                    {"word": "frequently", "meaning": "頻繁地"},
                    {"word": "contrast", "meaning": "對比"},
                    {"word": "distinct", "meaning": "明顯不同的"},
                    {"word": "meteorologists", "meaning": "氣象學家"},
                    {"word": "predict", "meaning": "預測"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "What kind of weather do countries near the equator have?", "a": "Cold and snowy", "b": "Tropical - warm and humid", "c": "Dry and windy", "correct": "B", "explain": "The story says 'countries like Brazil and Indonesia have tropical weather. It is warm and humid there.'"},
                    {"type": "comprehension", "q": "How cold can winters get in Canada and Russia?", "a": "Below minus 30 degrees Celsius", "b": "Below zero degrees Celsius", "c": "Below 10 degrees Celsius", "correct": "A", "explain": "The story says 'Temperatures can drop below minus thirty degrees Celsius!'"},
                    {"type": "vocabulary", "q": "What does 'predict' mean?", "a": "To remember the past", "b": "To say what will happen in the future", "c": "To write a story", "correct": "B", "explain": "Predict means to say what you think will happen before it happens."},
                    {"type": "comprehension", "q": "What do scientists who study weather call themselves?", "a": "Biologists", "b": "Meteorologists", "c": "Geologists", "correct": "B", "explain": "The story says 'Scientists who study weather are called meteorologists.'"},
                    {"type": "main_idea", "q": "What is the main idea of this passage?", "a": "How to become a meteorologist", "b": "Weather is different in different parts of the world", "c": "Why rainforests are important", "correct": "B", "explain": "The passage describes how weather varies in different regions of the world."}
                ]
            },
            # Level 5: Lexile 550L
            {
                "title": "The Amazing Honey Bee",
                "content": """Honey bees are remarkable insects that play an important role in our world. They live together in large groups called colonies, and each colony can have up to 60,000 bees!

Every colony has three types of bees: the queen, workers, and drones. The queen is the only bee that lays eggs. Worker bees are all female, and they do most of the work. They build the honeycomb, gather food, and protect the hive. Drones are male bees whose main job is to mate with the queen.

Bees are essential for pollination. When bees visit flowers to collect nectar, pollen sticks to their fuzzy bodies. As they fly from flower to flower, they spread this pollen. This process helps plants produce fruits and seeds. In fact, about one-third of the food we eat depends on pollination by bees!

Unfortunately, bee populations are declining around the world. Pesticides, habitat loss, and climate change are all threats to bees. Scientists and farmers are working together to find ways to protect these valuable insects.

You can help bees too! Planting flowers in your garden gives bees more places to find food. Avoiding pesticides in your yard also keeps bees safe.""",
                "lexile_level": 550,
                "difficulty": "intermediate",
                "vocabulary": json.dumps([
                    {"word": "remarkable", "meaning": "非凡的"},
                    {"word": "colonies", "meaning": "群落"},
                    {"word": "honeycomb", "meaning": "蜂巢"},
                    {"word": "essential", "meaning": "必要的"},
                    {"word": "pollination", "meaning": "授粉"},
                    {"word": "nectar", "meaning": "花蜜"},
                    {"word": "declining", "meaning": "下降中"},
                    {"word": "pesticides", "meaning": "殺蟲劑"},
                    {"word": "habitat", "meaning": "棲息地"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "How many bees can live in one colony?", "a": "Up to 6,000", "b": "Up to 60,000", "c": "Up to 600", "correct": "B", "explain": "The story says 'each colony can have up to 60,000 bees!'"},
                    {"type": "comprehension", "q": "What is the queen bee's main job?", "a": "To gather food", "b": "To protect the hive", "c": "To lay eggs", "correct": "C", "explain": "The story says 'The queen is the only bee that lays eggs.'"},
                    {"type": "vocabulary", "q": "What does 'essential' mean in this passage?", "a": "Not important", "b": "Very necessary", "c": "Dangerous", "correct": "B", "explain": "Essential means very important or necessary."},
                    {"type": "inference", "q": "Why are bees important for our food?", "a": "They make honey for us to eat", "b": "They help plants produce fruits through pollination", "c": "They eat harmful insects", "correct": "B", "explain": "The passage explains that pollination by bees helps plants produce fruits and seeds."},
                    {"type": "main_idea", "q": "What would be the best title for the last two paragraphs?", "a": "How Bees Make Honey", "b": "Threats to Bees and How to Help", "c": "The Life of a Worker Bee", "correct": "B", "explain": "The last two paragraphs discuss problems facing bees and what we can do to help."}
                ]
            }
        ]

        for p_data in sample_passages:
            passage = ReadingPassage(
                title=p_data["title"],
                content=p_data["content"],
                lexile_level=p_data["lexile_level"],
                difficulty=p_data["difficulty"],
                vocabulary=p_data["vocabulary"]
            )
            db.add(passage)
            db.commit()
            db.refresh(passage)

            # Add questions
            for q_data in p_data["questions"]:
                question = ReadingQuestion(
                    passage_id=passage.id,
                    question_type=q_data["type"],
                    question=q_data["q"],
                    option_a=q_data["a"],
                    option_b=q_data["b"],
                    option_c=q_data["c"],
                    option_d=q_data.get("d"),
                    correct_answer=q_data["correct"],
                    explanation=q_data["explain"]
                )
                db.add(question)

            db.commit()

        print(f"Created {len(sample_passages)} sample reading passages")

    finally:
        db.close()

# Initialize sample passages
init_sample_passages()
