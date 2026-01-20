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
- **API Endpoints**:
  - `GET /` - Serves index.html
  - `GET/POST /api/users` - User management
  - `POST /api/users/{id}/progress` - Update XP/level
  - `GET /api/levels` - List active level packs
  - `POST /api/admin/upload` - Upload Excel files to create missions
  - `DELETE /api/admin/levels/{id}` - Delete level pack
  - `GET /api/game/start/{level_id}` - Start game session with random words

### Frontend (`app/index.html`)

Single HTML file with embedded CSS and JavaScript. Four view sections controlled by `switchView()`:
1. **Login View** - User selection/creation
2. **Dashboard View** - Mission/level selection
3. **Game View** - Word scramble gameplay
4. **Admin View** - Parent zone for uploading Excel missions

### Data Flow

1. Excel files (columns: word/meaning/sentence) are uploaded via admin panel
2. Parsed with pandas into `LevelPack` + `Word` records
3. Game fetches random subset of words from selected pack
4. Player arranges scrambled sentence words; correct answer grants XP

### Excel Format for Missions

Columns auto-detected by keywords:
- Word column: contains "word" or "單字"
- Meaning column: contains "mean", "中文", or "def"
- Sentence column: contains "sent", "例句", or "ex"

## Key Patterns

- HTML templates are served as files from `app/` directory (not embedded strings)
- Database auto-creates tables on startup via `Base.metadata.create_all()`
- In-memory DB fallback if file DB fails to initialize
- Frontend uses vanilla JS with inline onclick handlers for simplicity
