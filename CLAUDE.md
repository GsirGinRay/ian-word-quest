# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ian's Word Quest is an English vocabulary learning game built as a single-page FastAPI application. Players complete "missions" by constructing sentences from scrambled words, earning XP to level up.

## Tech Stack

- **Backend**: FastAPI (Python 3.9+) with SQLAlchemy ORM
- **Database**: SQLite (file-based in `app/data/ian_quest.db`, falls back to in-memory)
- **Frontend**: Vanilla JavaScript with inline HTML served from FastAPI endpoints
- **Deployment**: Docker container on port 8080 (Zeabur hosting)

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (default port 8000)
uvicorn app.main:app --reload

# Run on production port
uvicorn app.main:app --host 0.0.0.0 --port 8080

# Build Docker image
docker build -t ian-word-quest .

# Run Docker container
docker run -p 8080:8080 ian-word-quest
```

## Architecture

### Backend (`app/main.py`)

Single-file FastAPI application containing:
- **Database Models**: `User`, `LevelPack`, `Word` (SQLAlchemy)
- **Pydantic Schemas**: `UserCreate`, `UserOut`, `LevelOut`
- **Auto-init**: On startup, loads words from `app/data/Ian's English Words-4.xlsx` and creates 30 levels (20 words each)
- **API Endpoints**:
  - `GET /` - Serves index.html
  - `GET/POST /api/users` - User management
  - `POST /api/users/{id}/progress` - Update XP/level
  - `GET /api/levels` - List active level packs
  - `POST /api/admin/upload` - Upload Excel files to create missions
  - `DELETE /api/admin/levels/{id}` - Delete level pack
  - `GET /api/game/start/{level_id}` - Start game session with random words

### Frontend (`app/index.html`)

Single HTML file with embedded CSS and JavaScript (Chinese UI). Views controlled by `switchView()`:
1. **Login View** - User selection/creation
2. **Dashboard View** - Level grid selection
3. **Game View** - Multiple-choice quiz (show English word, pick Chinese meaning)
4. **Admin View** - Upload Excel, manage levels

### Game Mechanics

- Multiple-choice format: Display English word, 4 Chinese options (1 correct, 3 random wrong)
- 5 questions per round, 20 XP per correct answer
- Progress bar shows completion
- Results screen with emoji feedback based on score

### Data Flow

1. On first startup, Excel file auto-loads into 30 level packs
2. Player selects user → selects level → plays 5-question quiz
3. XP updates after each round, level = XP / 100 + 1

## Key Patterns

- Uses `addEventListener` instead of inline onclick (CSP compatibility)
- Database auto-creates tables and pre-loads data on startup
- In-memory DB fallback if file DB fails
- Tablet-optimized: large touch targets, responsive grid layout
