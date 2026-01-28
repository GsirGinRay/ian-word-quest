import os
import random
import uuid
import shutil
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, DateTime, Text, text, inspect
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
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

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

class World(Base):
    """Game Worlds (Realms) for the Adventure Mode"""
    __tablename__ = "worlds"
    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String, unique=True, index=True)  # academy, wilds, etc.
    name = Column(String)
    description = Column(Text)
    theme_color = Column(String, default="#4CAF50")
    min_lexile = Column(Integer, default=300)
    max_lexile = Column(Integer, default=900)
    order = Column(Integer, default=1)
    passages = relationship("ReadingPassage", back_populates="world", order_by="ReadingPassage.id")


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
    image_url = Column(String, nullable=True)  # Path to cover image
    
    # Adventure Mode Fields
    world_id = Column(Integer, ForeignKey("worlds.id"), nullable=True)
    chapter = Column(Integer, default=1)   # 1-4 per world
    episode = Column(Integer, default=1)   # 1-5 per chapter
    boss_name = Column(String, nullable=True)  # Name of the enemy/challenge
    boss_image = Column(String, nullable=True) # Image of the enemy
    boss_hp = Column(Integer, default=100)     # Health points for battle
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    world = relationship("World", back_populates="passages")
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

def check_and_migrate_db():
    """Check for missing columns and migrate if necessary"""
    try:
        inspector = inspect(engine)
        
        # 1. Check ReadingPassage columns
        if inspector.has_table("reading_passages"):
            columns = [c["name"] for c in inspector.get_columns("reading_passages")]
            with engine.connect() as conn:
                if "image_url" not in columns:
                    print("MIGRATION: Adding image_url to reading_passages")
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN image_url VARCHAR"))
                
                # New 2.0 columns
                if "world_id" not in columns:
                    print("MIGRATION: Adding world_id to reading_passages")
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN world_id INTEGER"))
                
                if "chapter" not in columns:
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN chapter INTEGER DEFAULT 1"))
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN episode INTEGER DEFAULT 1"))
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN boss_name VARCHAR"))
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN boss_image VARCHAR"))
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN boss_hp INTEGER DEFAULT 100"))
                    print("MIGRATION: Added Adventure Mode columns")
                
                conn.commit()
                
        # 2. Check if Worlds table exists (handled by create_all, but we might need to populate it)
        
    except Exception as e:
        print(f"Migration check failed: {e}")

# Run migration check on startup
check_and_migrate_db()

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

@app.get("/api/worlds")
def get_worlds(db: Session = Depends(get_db)):
    """Get list of game worlds"""
    worlds = db.query(World).order_by(World.order).all()
    return [{
        "id": w.id,
        "slug": w.slug,
        "name": w.name, 
        "description": w.description,
        "theme_color": w.theme_color,
        "min_lexile": w.min_lexile,
        "max_lexile": w.max_lexile
    } for w in worlds]

@app.get("/api/reading/passages")
def get_passages(user_id: int = None, world_id: int = None, db: Session = Depends(get_db)):
    """Get all reading passages with user progress, optionally filtered by world"""
    query = db.query(ReadingPassage).filter(ReadingPassage.is_active == True)
    
    if world_id:
        query = query.filter(ReadingPassage.world_id == world_id)
        
    passages = query.order_by(ReadingPassage.world_id.asc(), ReadingPassage.chapter.asc(), ReadingPassage.episode.asc()).all()

    # Get user progress if provided
    progress_map = {}
    user_level = None
    if user_id:
        progress_list = db.query(UserReadingProgress).filter(UserReadingProgress.user_id == user_id).all()
        progress_map = {p.passage_id: p for p in progress_list}
        user_level = db.query(UserReadingLevel).filter(UserReadingLevel.user_id == user_id).first()

    result = []
    
    # Pre-fetch world info if we are listing all
    worlds = {w.id: w for w in db.query(World).all()}

    for p in passages:
        u_prog = progress_map.get(p.id)
        
        # Get World Name
        world_name = "Unknown"
        if p.world_id and p.world_id in worlds:
            world_name = worlds[p.world_id].name
            
        result.append({
            "id": p.id,
            "title": p.title,
            "lexile_level": p.lexile_level,
            "difficulty": p.difficulty,
            "image_url": p.image_url,
            "completed": u_prog.completed if u_prog else False,
            "score": u_prog.score if u_prog else 0,
            
            # Adventure Mode Fields
            "world_id": p.world_id,
            "world_name": world_name,
            "chapter": p.chapter,
            "episode": p.episode,
            "boss_name": p.boss_name,
            "boss_image": p.boss_image,
            "boss_hp": p.boss_hp
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
        "image_url": passage.image_url,
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

@app.post("/api/admin/reset-passages")
def reset_reading_passages(db: Session = Depends(get_db)):
    """Delete all reading passages and reinitialize with new stories"""
    # Delete all questions first (foreign key constraint)
    db.query(ReadingQuestion).delete()
    # Delete all passages
    db.query(ReadingPassage).delete()
    # Delete user progress
    db.query(UserReadingProgress).delete()
    db.commit()

    # Reinitialize with new stories
    init_sample_passages()

    count = db.query(ReadingPassage).count()
    return {"message": f"Reset complete! Created {count} new passages."}

# ========== INITIALIZE SAMPLE READING PASSAGES ==========
def init_sample_passages():
    """Create sample reading passages if none exist"""
    db = SessionLocal()
    try:
        existing = db.query(ReadingPassage).count()
        # if existing > 0:
        #    print(f"Already have {existing} reading passages, skipping init")
        #    return

        import json
        import glob
        import os
        
        sample_passages = []

        # Load from JSON files in data/stories
        BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Assuming BASE_DIR is needed here
        stories_dir = os.path.join(BASE_DIR, "data", "stories")
        if not os.path.exists(stories_dir):
            os.makedirs(stories_dir)
            
        json_files = glob.glob(os.path.join(stories_dir, "*.json"))
        print(f"Found {len(json_files)} story files in {stories_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        sample_passages.extend(data)
                        print(f"Loaded {len(data)} stories from {os.path.basename(json_file)}")
                    elif isinstance(data, dict):
                        sample_passages.append(data)
                        print(f"Loaded 1 story from {os.path.basename(json_file)}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        # If no external files, fall back to default hardcoded stories (OR merge them)
        # For now, let's keep the hardcoded ones as a fallback/starter set if nothing else exists
        # But we want to map them to worlds if possible.
        
        # Hardcoded set (Legacy + Starters)
        legacy_passages = [
            # ===== DETECTIVE AMY SERIES =====
            # Episode 1: Lexile 350L
            {
                "title": "🔍 Detective Amy #1: The Missing Cookies",
                "world": "academy", "chapter": 1, "episode": 1,
                "image": "/static/passages/amy1.jpg",
                "content": """Amy loves to solve mysteries. She is only 10 years old, but everyone calls her "Detective Amy."

One day, her mom baked chocolate cookies. She put them on the kitchen table. But when Amy came home from school, the cookies were GONE!

"Who took my cookies?" Mom asked.

Amy looked around. She saw brown crumbs on the floor. The crumbs made a trail. Amy followed the trail... to the dog's bed!

There was Max, the family dog. He had chocolate on his nose!

"I found the cookie thief!" Amy laughed. "It's Max!"

Mom said, "Good job, Detective Amy. But Max can't eat chocolate. It's bad for dogs!"

Amy learned something new: chocolate is dangerous for dogs. She gave Max a dog treat instead.""",
                "lexile_level": 350,
                "difficulty": "beginner",
                "vocabulary": json.dumps([
                    {"word": "detective", "meaning": "偵探"},
                    {"word": "mystery", "meaning": "謎團"},
                    {"word": "crumbs", "meaning": "碎屑"},
                    {"word": "trail", "meaning": "痕跡、軌跡"},
                    {"word": "thief", "meaning": "小偷"},
                    {"word": "dangerous", "meaning": "危險的"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "What does everyone call Amy?", "a": "Cookie Girl", "b": "Detective Amy", "c": "Super Amy", "correct": "B", "explain": "The story says 'everyone calls her Detective Amy.'"},
                    {"type": "comprehension", "q": "What was missing?", "a": "The dog", "b": "Amy's homework", "c": "Chocolate cookies", "correct": "C", "explain": "Mom's chocolate cookies were gone from the table."},
                    {"type": "inference", "q": "How did Amy know Max took the cookies?", "a": "Max told her", "b": "She followed the crumbs to Max's bed", "c": "She saw Max eat them", "correct": "B", "explain": "Amy followed the trail of crumbs to Max's bed and saw chocolate on his nose."},
                    {"type": "main_idea", "q": "What lesson did Amy learn?", "a": "Dogs are bad pets", "b": "Cookies are yummy", "c": "Chocolate is dangerous for dogs", "correct": "C", "explain": "Mom explained that chocolate is bad for dogs."}
                ]
            },
            # Episode 2: Lexile 400L
            {
                "title": "🔍 Detective Amy #2: The Disappearing Lunch",
                 "world": "academy", "chapter": 1, "episode": 2,
                "image": "/static/passages/amy2.jpg",
                "content": """Something strange was happening at school. Every day, someone's lunch disappeared!

On Monday, Ben's sandwich was gone. On Tuesday, Sara's apple vanished. On Wednesday, even the teacher's salad disappeared!

"We need Detective Amy!" the students said.

Amy began her investigation. She asked questions. "When did your lunch disappear?" Everyone said the same thing: "During morning recess."

Amy had an idea. On Thursday, she stayed inside during recess. She hid behind the bookshelf and watched.

Soon, the classroom door opened slowly. A small figure came in. It was... a squirrel!

The furry thief grabbed a banana from Tom's lunchbox. It ran out the window!

Amy told the teacher. They closed the window. No more lunches disappeared after that!

"Another mystery solved!" Amy said proudly.""",
                "lexile_level": 400,
                "difficulty": "beginner",
                "vocabulary": json.dumps([
                    {"word": "disappeared", "meaning": "消失了"},
                    {"word": "vanished", "meaning": "不見了"},
                    {"word": "investigation", "meaning": "調查"},
                    {"word": "recess", "meaning": "下課時間"},
                    {"word": "squirrel", "meaning": "松鼠"},
                    {"word": "figure", "meaning": "身影"},
                    {"word": "furry", "meaning": "毛茸茸的"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "When did the lunches disappear?", "a": "After school", "b": "During morning recess", "c": "At night", "correct": "B", "explain": "Everyone said their lunch disappeared during morning recess."},
                    {"type": "comprehension", "q": "Where did Amy hide?", "a": "Under the desk", "b": "Behind the bookshelf", "c": "In the closet", "correct": "B", "explain": "Amy hid behind the bookshelf to watch."},
                    {"type": "comprehension", "q": "Who was the real thief?", "a": "A student", "b": "The teacher", "c": "A squirrel", "correct": "C", "explain": "A squirrel was coming through the window to steal food."},
                    {"type": "inference", "q": "How did they stop the thief?", "a": "They caught the squirrel", "b": "They closed the window", "c": "They called the police", "correct": "B", "explain": "They closed the window so the squirrel couldn't get in anymore."}
                ]
            },
            # Episode 3: Lexile 450L (Academy World)
            {
                "title": "🔍 Detective Amy #3: The Secret Note",
                "world": "academy", "chapter": 1, "episode": 3,
                "image": "/static/passages/amy3.jpg",
                "content": """One sunny morning, Amy found a crumpled piece of paper inside her desk. It was a secret note!

It said: "Meet me at the big oak tree after schoo. Don't be late!"

The note was unsigned. Who could it be?

Amy looked at the handwriting. It was messy. The letters were big and slanted. She looked around the classroom.

Ben was writing in his notebook. His letters were small and neat. It wasn't Ben.
Sara was drawing. She used purple ink. The note was written in black pencil. It wasn't Sara.

Then Amy saw Tom. Tom was looking at her. He quickly looked away. He had a black pencil in his hand.

Amy smiled. She knew who wrote the note.

After school, she went to the big oak tree. Tom was there, holding two ice creams.

"I wanted to share these with you," Tom said shyly. "You are a great detective!"

Amy laughed. "And you need to work on your handwriting!" she teased. They ate their ice creams happily.""",
                "lexile_level": 450,
                "difficulty": "beginner",
                "vocabulary": json.dumps([
                    {"word": "crumpled", "meaning": "皺巴巴的"},
                    {"word": "unsigned", "meaning": "未署名的"},
                    {"word": "handwriting", "meaning": "筆跡"},
                    {"word": "messy", "meaning": "凌亂的"},
                    {"word": "slanted", "meaning": "傾斜的"},
                    {"word": "shyly", "meaning": "害羞地"},
                    {"word": "teased", "meaning": "戲弄、取笑"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "Where did Amy find the note?", "a": "On the floor", "b": "Inside her desk", "c": "In her bag", "correct": "B", "explain": "Amy found the crumpled paper inside her desk."},
                    {"type": "inference", "q": "Why did Amy think it was Tom?", "a": "He told her", "b": "He was holding a black pencil and looking at her", "c": "He waved at her", "correct": "B", "explain": "Amy saw Tom looking at her and holding a black pencil, which matched the note."},
                    {"type": "vocabulary", "q": "What does 'messy' mean?", "a": "Very clean", "b": "Not neat or organized", "c": "Colorful", "correct": "B", "explain": "Messy means untidy or not neat. Tom's handwriting was messy."},
                    {"type": "comprehension", "q": "What was the surprise?", "a": "A puppy", "b": "Ice creams", "c": "A new book", "correct": "B", "explain": "Tom was waiting with two ice creams to share."}
                ]
            },
            # ===== WILDS SERIES (New) =====
            {
                "title": "🌲 Wilds: The Forest Guardian",
                "world": "wilds", "chapter": 1, "episode": 1,
                "image": "https://images.unsplash.com/photo-1448375240586-dfd8d395ea6c?q=80&w=800",
                "content": """Deep in the Whispering Woods, there lived a bear named Koda. Koda was not an ordinary bear. He was the Guardian of the Forest.

One day, Koda heard a loud noise. CRACK! BOOM!

He ran towards the sound. He saw a group of men with big machines. They were cutting down the ancient trees!

"Stop!" Koda roared. But the men could not understand bear language. They only heard a scary growl. They were frightened and ran away.

But Koda knew they would come back. He needed a plan.

He gathered all the animals: the squirrels, the owls, and the foxes. "We must protect our home," Koda said.

The next day, the men returned. But this time, the forest was ready.
The squirrels dropped nuts on their heads. The owls hooted loudly to scare them. The foxes stole their keys.

The men were confused and annoyed. "This forest is haunted!" they yelled. They packed up their machines and left, never to return.

The animals cheered. The Whispering Woods were safe once again, thanks to their brave Guardian.""",
                "lexile_level": 500,
                "difficulty": "intermediate",
                "vocabulary": json.dumps([
                    {"word": "guardian", "meaning": "守護者"},
                    {"word": "ordinary", "meaning": "普通的"},
                    {"word": "ancient", "meaning": "古老的"},
                    {"word": "frightened", "meaning": "受驚嚇的"},
                    {"word": "gathered", "meaning": "聚集"},
                    {"word": "annoyed", "meaning": "惱怒的"},
                    {"word": "haunted", "meaning": "鬧鬼的"}
                ]),
                "questions": [
                    {"type": "main_idea", "q": "What is the main idea of the story?", "a": "Bears like honey", "b": "Animals working together to protect their home", "c": "Men building a house", "correct": "B", "explain": "The story is about Koda and the animals protecting the forest from being cut down."},
                    {"type": "vocabulary", "q": "What is a 'guardian'?", "a": "Someone who destroys things", "b": "Someone who protects things", "c": "A type of tree", "correct": "B", "explain": "A guardian is a protector. Koda protected the forest."},
                    {"type": "detail", "q": "What did the squirrels do?", "a": "Stole keys", "b": "Dropped nuts on the men's heads", "c": "Hooted loudly", "correct": "B", "explain": "The squirrels dropped nuts from the trees."},
                    {"type": "inference", "q": "Why did the men think the forest was haunted?", "a": "They saw a ghost", "b": "Strange things were happening to them", "c": "It was dark", "correct": "B", "explain": "The animals' tricks (falling nuts, stolen keys) made the men think spirits were attacking them."}
                ]
            },
            # ===== KINGDOM SERIES (New) =====
            {
                "title": "⚔️ Kingdom: The Brave Knight",
                "world": "kingdom", "chapter": 1, "episode": 1,
                "image": "https://images.unsplash.com/photo-1627768846176-59178ad80f48?q=80&w=800",
                "content": """Sir Leo was a young knight. He wanted to be a hero, but he was afraid of dragons.

"A knight must be brave!" his father told him. "You must face your fears."

One day, the King's golden crown was stolen. A giant dragon had taken it to the Black Mountain.

"Who will retrieve my crown?" the King asked. All the other knights were busy fighting wars. Only Leo was left.

"I... I will go," Leo stammered. His knees were shaking.

Leo climbed the steep mountain. At the top, he found a cave. Inside, a red dragon was sleeping on a pile of gold. The crown was right on its nose!

Leo crept closer. Suddenly, the dragon opened one eye. "ROAR! Who dares to wake me?"

Leo was terrified, but he stood tall. "I am Sir Leo. I came for the King's crown."

The dragon looked at him. "You are very small," the dragon said. "But you are polite. Most knights just attack me."

"I don't want to fight," Leo said. "I just want the crown."

"Take it," the dragon yawned. "It's uncomfortable anyway."

Leo grabbed the crown and ran back to the castle. He realized that being brave doesn't mean having no fear. It means doing what is right, even when you are scared.""",
                "lexile_level": 600,
                "difficulty": "intermediate",
                "vocabulary": json.dumps([
                    {"word": "knight", "meaning": "騎士"},
                    {"word": "retrieve", "meaning": "找回、取回"},
                    {"word": "stammered", "meaning": "結巴地說"},
                    {"word": "steep", "meaning": "陡峭的"},
                    {"word": "terrified", "meaning": "非常害怕的"},
                    {"word": "polite", "meaning": "有禮貌的"},
                    {"word": "uncomfortable", "meaning": "不舒服的"}
                ]),
                "questions": [
                    {"type": "inference", "q": "Why did the dragon give back the crown?", "a": "It was scared of Leo", "b": "Leo was polite and didn't attack", "c": "The crown was too heavy", "correct": "B", "explain": "The dragon appreciated that Leo was polite and didn't try to fight him."},
                    {"type": "vocabulary", "q": "What does 'retrieve' mean?", "a": "To throw away", "b": "To bring back", "c": "To break", "correct": "B", "explain": "The King wanted someone to bring back (retrieve) his stolen crown."},
                    {"type": "main_idea", "q": "What lesson did Leo learn?", "a": "Dragons are friendly", "b": "Bravery is acting in spite of fear", "c": "Gold is heavy", "correct": "B", "explain": "Leo learned that being brave means doing the right thing even when you are scared."},
                    {"type": "detail", "q": "Where was the crown?", "a": "On the dragon's nose", "b": "Under the dragon's tail", "c": "In a chest", "correct": "A", "explain": "The story says the crown was right on the sleeping dragon's nose."}
                ]
            },
            # ===== TIME TRAVELERS SERIES (continued) =====
            # Episode 2: Lexile 500L
            {
                "title": "⏰ Time Travelers #2: Escape from Ancient Egypt",
                "image": "/static/passages/time2.jpg",
                "content": """Tom pressed a button on the magic watch. "Let's go to ancient Egypt!" he said.

WHOOOOSH!

When they opened their eyes, they were standing in a desert. The sun was burning hot. In front of them stood the Great Pyramid - and it was being built RIGHT NOW!

Thousands of workers were pulling huge stone blocks. Men shouted orders. Lily grabbed Tom's arm. "This is amazing!" she whispered.

Suddenly, a guard spotted them. "STOP! Who are you?" he yelled. "Spies! Catch them!"

Tom and Lily started running. The guards chased them between the giant stone blocks. Tom's heart was pounding!

"Press the watch!" Lily screamed.

Tom looked at the watch, but his hands were shaking. He couldn't find the right button!

A guard reached out to grab Lily's arm. Just then, Tom found the button and pressed it hard.

WHOOOOSH!

They landed back in the attic, safe and sound. But Lily was holding something in her hand - a small golden scarab beetle!

"We brought back treasure from ancient Egypt!" she gasped.

Grandma walked in. She saw the beetle and smiled mysteriously. "Ah, I see you found my old watch. And it still works perfectly..."

What did Grandma mean? Did SHE use the watch before?

TO BE CONTINUED...""",
                "lexile_level": 500,
                "difficulty": "intermediate",
                "vocabulary": json.dumps([
                    {"word": "ancient", "meaning": "古代的"},
                    {"word": "pyramid", "meaning": "金字塔"},
                    {"word": "desert", "meaning": "沙漠"},
                    {"word": "guard", "meaning": "守衛"},
                    {"word": "spotted", "meaning": "發現了"},
                    {"word": "chased", "meaning": "追趕"},
                    {"word": "pounding", "meaning": "怦怦跳"},
                    {"word": "scarab", "meaning": "聖甲蟲"},
                    {"word": "mysteriously", "meaning": "神秘地"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "Where did Tom and Lily travel to?", "a": "Ancient China", "b": "Ancient Egypt", "c": "Ancient Rome", "correct": "B", "explain": "Tom said 'Let's go to ancient Egypt!' and they saw the Great Pyramid."},
                    {"type": "comprehension", "q": "What did the guards think Tom and Lily were?", "a": "Tourists", "b": "Workers", "c": "Spies", "correct": "C", "explain": "The guard yelled 'Spies! Catch them!'"},
                    {"type": "inference", "q": "Why couldn't Tom press the watch button quickly?", "a": "The button was broken", "b": "His hands were shaking from fear", "c": "The watch ran out of power", "correct": "B", "explain": "The story says 'his hands were shaking' because they were being chased."},
                    {"type": "comprehension", "q": "What did Lily bring back from Egypt?", "a": "A stone block", "b": "A golden scarab beetle", "c": "A pyramid model", "correct": "B", "explain": "Lily was holding 'a small golden scarab beetle.'"},
                    {"type": "inference", "q": "What secret does Grandma seem to have?", "a": "She built the watch", "b": "She has used the watch for time travel before", "c": "She works at a museum", "correct": "B", "explain": "Grandma smiled mysteriously and said the watch 'still works perfectly,' suggesting she used it before."}
                ]
            },
            # ===== DETECTIVE AMY SERIES (continued) =====
            # Episode 3: Lexile 550L
            {
                "title": "🔍 Detective Amy #3: The Ghost in the Library",
                "image": "/static/passages/amy3.jpg",
                "content": """Strange things were happening in the school library. Books fell off shelves by themselves. Lights flickered on and off. And sometimes, students heard whispering when nobody was there!

"It's a ghost!" everyone said. Some students were too scared to enter the library.

But Detective Amy didn't believe in ghosts. "There must be a logical explanation," she said.

Amy decided to investigate. She stayed late after school and hid between the bookshelves. The library grew dark and quiet. Amy waited nervously.

Suddenly - WHOOSH! A book flew off the shelf! Amy jumped, but she didn't scream. Instead, she looked carefully.

She noticed something: there was a small hole in the wall behind the bookshelf. Cold air was blowing through the hole! When the wind blew hard, it pushed the books off the shelves.

Amy followed the hole and found that it connected to the old air conditioning system. The broken machine was making the whispering sound!

The next day, Amy explained everything to the principal. The maintenance team fixed the hole and repaired the air conditioner.

"No more ghost!" the students cheered.

Amy smiled. "Remember: every mystery has a solution. You just have to look for clues!"

But that night, Amy received a mysterious letter with no return address. It said: "Great detective work. I'm watching you. - A Friend"

Who was watching Amy? A new mystery was about to begin...

TO BE CONTINUED...""",
                "lexile_level": 550,
                "difficulty": "intermediate",
                "vocabulary": json.dumps([
                    {"word": "flickered", "meaning": "閃爍"},
                    {"word": "whispering", "meaning": "低語聲"},
                    {"word": "logical", "meaning": "合理的"},
                    {"word": "explanation", "meaning": "解釋"},
                    {"word": "investigate", "meaning": "調查"},
                    {"word": "nervously", "meaning": "緊張地"},
                    {"word": "maintenance", "meaning": "維修"},
                    {"word": "principal", "meaning": "校長"},
                    {"word": "mysterious", "meaning": "神秘的"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "What strange things happened in the library?", "a": "Books disappeared completely", "b": "Books fell, lights flickered, and there was whispering", "c": "The librarian acted weird", "correct": "B", "explain": "Books fell off shelves, lights flickered on and off, and students heard whispering."},
                    {"type": "comprehension", "q": "Why did Amy stay late in the library?", "a": "To read books", "b": "To do homework", "c": "To investigate the mystery", "correct": "C", "explain": "Amy decided to investigate and hid between the bookshelves."},
                    {"type": "inference", "q": "What was really causing the 'ghost' sounds?", "a": "A real ghost", "b": "Students playing tricks", "c": "A broken air conditioning system", "correct": "C", "explain": "The hole connected to the old air conditioning system, and the broken machine was making the whispering sound."},
                    {"type": "vocabulary", "q": "What does 'logical' mean?", "a": "Scary and supernatural", "b": "Making sense and reasonable", "c": "Funny and silly", "correct": "B", "explain": "Logical means something that makes sense and can be explained with reasoning."},
                    {"type": "inference", "q": "Who might have sent the mysterious letter at the end?", "a": "The principal", "b": "Someone who knows about Amy's detective skills", "c": "The ghost", "correct": "B", "explain": "The letter praised Amy's 'detective work,' so someone has been following her mysteries."},
                    {"type": "main_idea", "q": "What lesson does this story teach?", "a": "Ghosts are real", "b": "Libraries are dangerous", "c": "Every mystery has a logical solution if you look for clues", "correct": "C", "explain": "Amy said 'every mystery has a solution. You just have to look for clues!'"}
                ]
            },
            # ===== TIME TRAVELERS SERIES (continued) =====
            # Episode 3: Lexile 600L
            {
                "title": "⏰ Time Travelers #3: Knights and Dragons",
                "image": "/static/passages/time3.jpg",
                "content": """Grandma sat down with Tom and Lily. She held the magic watch carefully.

"I used this watch many times when I was young," Grandma said. "I've seen dinosaurs, met ancient kings, and even visited the future. But one trip was the most dangerous of all."

Tom and Lily listened with wide eyes.

"I traveled to medieval Europe," Grandma continued. "The year was 1352. I accidentally appeared inside a castle during a battle!"

"What happened?" Lily asked excitedly.

Grandma smiled. "Why don't I show you? Let's go together this time."

She pressed the buttons expertly. WHOOOOSH!

Suddenly, they stood in a stone courtyard. Knights in shining armor rushed past them. Swords clashed and arrows flew through the air!

"We need to hide!" Grandma shouted. She led them behind a wooden cart.

A young knight fell near them. He was injured and couldn't get up. Without thinking, Tom ran out and helped drag the knight to safety.

"You saved my life!" the knight said gratefully. "I am Prince Edward. When I become king, I will remember your bravery."

Grandma pressed the watch. WHOOOOSH! They returned home.

"Grandma," Tom asked, "did Prince Edward really become king?"

Grandma pulled out an old history book. "Look at page 47."

Tom found the page. It showed a painting of King Edward III - and the king was wearing the EXACT same face as the young knight they saved!

"Our family has been part of history," Grandma winked. "More than you'll ever know."

TO BE CONTINUED...""",
                "lexile_level": 600,
                "difficulty": "intermediate",
                "vocabulary": json.dumps([
                    {"word": "medieval", "meaning": "中世紀的"},
                    {"word": "accidentally", "meaning": "意外地"},
                    {"word": "expertly", "meaning": "熟練地"},
                    {"word": "courtyard", "meaning": "庭院"},
                    {"word": "armor", "meaning": "盔甲"},
                    {"word": "clashed", "meaning": "碰撞"},
                    {"word": "injured", "meaning": "受傷的"},
                    {"word": "gratefully", "meaning": "感激地"},
                    {"word": "bravery", "meaning": "勇敢"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "What year did Grandma travel to in her dangerous trip?", "a": "1250", "b": "1352", "c": "1452", "correct": "B", "explain": "Grandma said 'The year was 1352.'"},
                    {"type": "comprehension", "q": "Where did they appear when they traveled?", "a": "In a forest", "b": "In a castle during a battle", "c": "On a ship", "correct": "B", "explain": "They appeared inside a castle during a battle."},
                    {"type": "inference", "q": "Why did Tom run out to help the knight?", "a": "He wanted a reward", "b": "Grandma told him to", "c": "He acted bravely without thinking", "correct": "C", "explain": "The story says 'Without thinking, Tom ran out and helped.'"},
                    {"type": "comprehension", "q": "Who was the young knight they saved?", "a": "A farmer", "b": "Prince Edward, who later became king", "c": "A merchant", "correct": "B", "explain": "The knight said 'I am Prince Edward' and the history book showed he became King Edward III."},
                    {"type": "inference", "q": "What does Grandma mean when she says 'Our family has been part of history'?", "a": "Their family writes history books", "b": "Their family has influenced important historical events through time travel", "c": "Their family is very old", "correct": "B", "explain": "Tom saved Prince Edward who became a king, showing how their time travels affected history."},
                    {"type": "vocabulary", "q": "What does 'medieval' mean?", "a": "Related to the Middle Ages in Europe", "b": "Related to modern times", "c": "Related to the future", "correct": "A", "explain": "Medieval refers to the Middle Ages, roughly 500-1500 CE in European history."}
                ]
            },
            # ===== DETECTIVE AMY SERIES (continued) =====
            # Episode 4: Lexile 650L
            {
                "title": "🔍 Detective Amy #4: The Secret Code",
                "image": "/static/passages/amy4.jpg",
                "content": """Amy couldn't stop thinking about the mysterious letter. Someone was watching her, but who?

Then more strange things began happening. Amy found coded messages in her locker. The first one said: "PHHW DW WKH ROG WUHH."

Amy recognized it immediately - a Caesar cipher! She shifted each letter back by 3. M-E-E-T A-T T-H-E O-L-D T-R-E-E.

"Meet at the old tree," Amy whispered. The old oak tree was behind the school playground.

Should she go? It could be dangerous. But Amy's curiosity was stronger than her fear.

After school, she walked to the old tree. Nobody was there. But she found another note pinned to the bark: "LOOK UP."

Amy looked up and gasped. In the branches sat three kids from her class - Ben, Sara, and Tom! They were all wearing matching pins that said "Junior Detectives."

"Surprise!" they laughed.

Ben explained, "We've been watching your cases. You're amazing at solving mysteries! We created this club to learn from you."

Amy felt touched. "But why all the secrets and codes?"

"We wanted to prove we could be good detectives too," Sara said. "Did we pass the test?"

Amy grinned. "You used proper encryption, left clever clues, and never revealed your identities. You definitely passed!"

From that day on, the Junior Detectives Club met every week. Amy taught them how to find clues, and together they solved even bigger mysteries.

But Amy kept wondering - those first mysterious letters... were they really from Ben's group? Or was someone else still watching?

TO BE CONTINUED...""",
                "lexile_level": 650,
                "difficulty": "advanced",
                "vocabulary": json.dumps([
                    {"word": "coded", "meaning": "編碼的"},
                    {"word": "cipher", "meaning": "密碼"},
                    {"word": "shifted", "meaning": "移動"},
                    {"word": "curiosity", "meaning": "好奇心"},
                    {"word": "pinned", "meaning": "釘住"},
                    {"word": "matching", "meaning": "相配的"},
                    {"word": "touched", "meaning": "感動的"},
                    {"word": "encryption", "meaning": "加密"},
                    {"word": "revealed", "meaning": "揭露"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "What kind of code was used in the message?", "a": "Binary code", "b": "Caesar cipher", "c": "Morse code", "correct": "B", "explain": "Amy recognized it as a Caesar cipher and shifted each letter back by 3."},
                    {"type": "comprehension", "q": "What did the decoded message say?", "a": "Meet at the old tree", "b": "Go to the library", "c": "Come to school early", "correct": "A", "explain": "When Amy decoded it, it said 'Meet at the old tree.'"},
                    {"type": "inference", "q": "Why did Amy decide to go even though it might be dangerous?", "a": "Her parents made her go", "b": "Her curiosity was stronger than her fear", "c": "She didn't think it was dangerous", "correct": "B", "explain": "The story says 'Amy's curiosity was stronger than her fear.'"},
                    {"type": "comprehension", "q": "What was the surprise waiting for Amy?", "a": "A birthday party", "b": "Three classmates who wanted to form a detective club", "c": "A treasure chest", "correct": "B", "explain": "Ben, Sara, and Tom were there wearing 'Junior Detectives' pins and wanted to learn from Amy."},
                    {"type": "inference", "q": "Why does the story end with Amy wondering if someone else is still watching?", "a": "She is paranoid", "b": "There may be another mystery connected to the original letters", "c": "She forgot who sent them", "correct": "B", "explain": "Amy questions whether the FIRST mysterious letters were really from Ben's group, suggesting another mystery."},
                    {"type": "vocabulary", "q": "What does 'encryption' mean?", "a": "Writing in a foreign language", "b": "Converting information into a secret code", "c": "Writing very small", "correct": "B", "explain": "Encryption means converting information into code so only certain people can read it."}
                ]
            },
            # ===== TIME TRAVELERS SERIES (continued) =====
            # Episode 4: Lexile 700L
            {
                "title": "⏰ Time Travelers #4: The Future City",
                 "world": "neon-city", "chapter": 1, "episode": 1,
                "image": "/static/passages/time4.jpg",
                "content": """After their adventure with the knights, Tom had an idea.

"Grandma, can the watch travel to the future too?"

Grandma hesitated. "Yes, but traveling to the future is more dangerous. The future isn't fixed - it changes based on our decisions today. What we see might not come true."

Lily's eyes sparkled. "Can we try? Just once?"

Grandma sighed but smiled. "Very well. But we must be careful."

She set the watch to the year 2150. WHOOOOSH!

They appeared in a city that took their breath away. Flying cars zoomed through transparent tubes in the sky. Buildings stretched up like silver needles, covered with gardens on every floor. Robots walked alongside humans, chatting like old friends.

"This is incredible!" Tom exclaimed.

A friendly robot approached them. "Welcome, time travelers! I am Assistant Unit 7. You are the seventeenth group from the past to visit this month."

"People time travel often?" Lily asked, surprised.

"Oh yes! After Dr. Chen invented affordable time machines in 2089, temporal tourism became popular. Many people visit their ancestors or see historical events."

Tom frowned. "Wait - if everyone time travels, doesn't that mess up history?"

The robot beeped thoughtfully. "Excellent question! The Temporal Protection Agency monitors all trips. They prevent anyone from making changes that would harm the timeline."

Suddenly, an alarm blared. Red lights flashed everywhere.

"WARNING," a voice announced. "Unauthorized temporal anomaly detected. All time travelers must return to their origin points immediately."

Grandma grabbed the children's hands. "That's our cue to leave!"

She pressed the watch frantically. WHOOOOSH!

They landed back in the attic. But something was different. The room looked older. Dust covered everything.

"Grandma," Tom whispered, "how long were we gone?"

Outside the window, they saw their house - but it looked abandoned. What had happened while they were away?

TO BE CONTINUED...""",
                "lexile_level": 700,
                "difficulty": "advanced",
                "vocabulary": json.dumps([
                    {"word": "hesitated", "meaning": "猶豫"},
                    {"word": "transparent", "meaning": "透明的"},
                    {"word": "incredible", "meaning": "難以置信的"},
                    {"word": "ancestors", "meaning": "祖先"},
                    {"word": "temporal", "meaning": "時間的"},
                    {"word": "unauthorized", "meaning": "未經授權的"},
                    {"word": "anomaly", "meaning": "異常"},
                    {"word": "frantically", "meaning": "瘋狂地"},
                    {"word": "abandoned", "meaning": "被遺棄的"}
                ]),
                "questions": [
                    {"type": "comprehension", "q": "What year did they travel to?", "a": "2050", "b": "2150", "c": "2250", "correct": "B", "explain": "Grandma set the watch to the year 2150."},
                    {"type": "comprehension", "q": "According to the robot, who invented affordable time machines?", "a": "Grandma", "b": "A scientist named Newton", "c": "Dr. Chen in 2089", "correct": "C", "explain": "The robot said 'Dr. Chen invented affordable time machines in 2089.'"},
                    {"type": "inference", "q": "Why did Grandma say traveling to the future is dangerous?", "a": "There are monsters in the future", "b": "The future isn't fixed and changes based on present decisions", "c": "The watch might break", "correct": "B", "explain": "Grandma explained 'The future isn't fixed - it changes based on our decisions today.'"},
                    {"type": "comprehension", "q": "What does the Temporal Protection Agency do?", "a": "Build time machines", "b": "Monitor time trips to prevent harmful changes to history", "c": "Give tours of the future", "correct": "B", "explain": "The Agency 'monitors all trips' and 'prevents anyone from making changes that would harm the timeline.'"},
                    {"type": "inference", "q": "What problem might they face at the end of the story?", "a": "They lost the watch", "b": "Time passed differently while they were gone, and something changed at home", "c": "They traveled to the wrong year", "correct": "B", "explain": "The attic looked older, dusty, and the house looked abandoned - suggesting time passed strangely while they were away."},
                    {"type": "vocabulary", "q": "What does 'anomaly' mean?", "a": "Something normal", "b": "Something unusual or unexpected", "c": "A type of robot", "correct": "B", "explain": "Anomaly means something that deviates from what is normal or expected."}
                ]
            }
        ]
        
        # Combine loaded and legacy
        sample_passages.extend(legacy_passages)

        print("Checking/Updating sample reading passages...")
        created_count = 0
        updated_count = 0

        for p_data in sample_passages:
            # Check if passage already exists
            existing_passage = db.query(ReadingPassage).filter(ReadingPassage.title == p_data["title"]).first()
            
            # Resolve world_slug to world_id if present
            world_id = None
            if "world" in p_data:
                world = db.query(World).filter(World.slug == p_data["world"]).first()
                if world:
                    world_id = world.id

            if existing_passage:
                # Update existing passage
                # logic to update columns...
                existing_passage.image_url = p_data.get("image")
                existing_passage.content = p_data["content"]
                existing_passage.vocabulary = p_data["vocabulary"]
                
                # Update 2.0 fields
                if world_id: existing_passage.world_id = world_id
                if "chapter" in p_data: existing_passage.chapter = p_data["chapter"]
                if "episode" in p_data: existing_passage.episode = p_data["episode"]
                if "boss" in p_data:
                    existing_passage.boss_name = p_data["boss"].get("name")
                    existing_passage.boss_hp = p_data["boss"].get("hp", 100)
                    existing_passage.boss_image = p_data["boss"].get("image")
                
                db.commit()
                updated_count += 1
            else:
                # Create new passage
                passage = ReadingPassage(
                    title=p_data["title"],
                    content=p_data["content"],
                    lexile_level=p_data["lexile_level"],
                    difficulty=p_data["difficulty"],
                    image_url=p_data.get("image"),
                    vocabulary=p_data["vocabulary"],
                    world_id=world_id,
                    chapter=p_data.get("chapter", 1),
                    episode=p_data.get("episode", 1),
                    boss_name=p_data.get("boss", {}).get("name"),
                    boss_image=p_data.get("boss", {}).get("image"),
                    boss_hp=p_data.get("boss", {}).get("hp", 100)
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
                created_count += 1

        print(f"Finished initialization: Created {created_count}, Updated {updated_count} passages")

    finally:
        db.close()

# Initialize sample passages
def init_worlds():
    """Initialize the 5 game worlds"""
    db = SessionLocal()
    try:
        if db.query(World).count() > 0:
            return

        worlds_data = [
            {
                "slug": "academy",
                "name": "The Academy",
                "description": "Where it all begins. Solve school mysteries and find the secret lab.",
                "theme_color": "#4CAF50", # Green
                "min_lexile": 350,
                "max_lexile": 450,
                "order": 1
            },
            {
                "slug": "wilds",
                "name": "The Wilds",
                "description": "A magical forest full of talking animals and ancient ruins.",
                "theme_color": "#FF9800", # Orange
                "min_lexile": 450,
                "max_lexile": 550,
                "order": 2
            },
            {
                "slug": "kingdom",
                "name": "Kingdom's Edge",
                "description": "Travel to the medieval past. Knights, dragons, and dark magic.",
                "theme_color": "#9C27B0", # Purple
                "min_lexile": 550,
                "max_lexile": 650,
                "order": 3
            },
            {
                "slug": "neon-city",
                "name": "Neon City",
                "description": "A cyberpunk future. Hackers, robots, and high-tech crimes.",
                "theme_color": "#00BCD4", # Cyan
                "min_lexile": 650,
                "max_lexile": 750,
                "order": 4
            },
            {
                "slug": "ancients",
                "name": "The Ancients",
                "description": "The realm of myths. Face the trials of Greek gods.",
                "theme_color": "#FFC107", # Gold
                "min_lexile": 750,
                "max_lexile": 900,
                "order": 5
            }
        ]

        for w_data in worlds_data:
            world = World(**w_data)
            db.add(world)
        
        db.commit()
        print("Initialized 5 Game Worlds")
    
    except Exception as e:
        print(f"Error initializing worlds: {e}")
    finally:
        db.close()

def run_migrations():
    """Ensure all database tables and columns exist"""
    print("Checking database schema...")
    try:
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)
        
        inspector = inspect(engine)
        with engine.connect() as conn:
            # 1. Check reading_passages columns
            if inspector.has_table("reading_passages"):
                columns = [c["name"] for c in inspector.get_columns("reading_passages")]
                
                # Image URL
                if "image_url" not in columns:
                    print("MIGRATION: Adding image_url to reading_passages")
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN image_url VARCHAR"))
                
                # Adventure Mode fields
                if "world_id" not in columns:
                    print("MIGRATION: Adding world_id to reading_passages")
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN world_id INTEGER"))
                
                if "chapter" not in columns:
                    print("MIGRATION: Adding chapter/episode/boss columns")
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN chapter INTEGER DEFAULT 1"))
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN episode INTEGER DEFAULT 1"))
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN boss_name VARCHAR"))
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN boss_image VARCHAR"))
                    conn.execute(text("ALTER TABLE reading_passages ADD COLUMN boss_hp INTEGER DEFAULT 100"))
                
                conn.commit()
        print("Migration check complete.")
    except Exception as e:
        print(f"Migration Error: {e}")

# Run migrations/checks
run_migrations()

# Initialize worlds first
init_worlds()

# Initialize sample passages
try:
    init_sample_passages()
except Exception as e:
    print(f"Error initializing passages: {e}")
