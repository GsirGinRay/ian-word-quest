import os
import random
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Path to the specific Excel file
EXCEL_PATH = "/app/data/Ian's English Words-4.xlsx" # For Docker environment
LOCAL_PATH = "../Ian's English Words-4.xlsx"        # For local testing

def load_words_from_excel():
    path = None
    if os.path.exists("Ian's English Words-4.xlsx"):
        path = "Ian's English Words-4.xlsx"
    elif os.path.exists(LOCAL_PATH):
        path = LOCAL_PATH
    
    # Defaults if file fails
    default_words = [
        {"word": "abandon", "meaning": "放棄", "sentence": "He decided to abandon the sinking ship."},
        {"word": "ability", "meaning": "能力", "sentence": "She has the ability to solve complex problems."},
        {"word": "abroad", "meaning": "在國外", "sentence": "They plan to study abroad next year."},
        {"word": "absence", "meaning": "缺席", "sentence": "His absence was noticed by the teacher."},
        {"word": "absolute", "meaning": "絕對的", "sentence": "I have absolute confidence in you."}
    ]

    if not path:
        print("Excel file not found, using defaults.")
        return default_words

    try:
        # Try to read the excel file
        df = pd.read_excel(path)
        
        # Normalize headers to lowercase to find columns easier
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # Intelligent column mapping (guessing common names)
        word_col = next((c for c in df.columns if 'word' in c or '單字' in c), None)
        mean_col = next((c for c in df.columns if 'mean' in c or '中文' in c or 'def' in c), None)
        sent_col = next((c for c in df.columns if 'sent' in c or '例句' in c or 'ex' in c), None)

        if not word_col:
            return default_words

        words = []
        for _, row in df.iterrows():
            if pd.isna(row[word_col]): continue
            
            w = str(row[word_col]).strip()
            m = str(row[mean_col]).strip() if mean_col and not pd.isna(row[mean_col]) else "???"
            s = str(row[sent_col]).strip() if sent_col and not pd.isna(row[sent_col]) else f"This is a sentence about {w}."
            
            words.append({
                "word": w,
                "meaning": m,
                "sentence": s
            })
            
        return words if words else default_words
        
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return default_words

@app.get("/")
async def get_index():
    with open("app/game.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/words")
async def get_words(count: int = 5):
    all_words = load_words_from_excel()
    # Return a random sample
    if len(all_words) <= count:
        return all_words
    return random.sample(all_words, count)

@app.get("/api/health")
async def health():
    return {"status": "ok"}
