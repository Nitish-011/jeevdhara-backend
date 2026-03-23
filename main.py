"""
Jeevdhara Backend API
FastAPI microservice for biodiversity classification and gamification.
"""

import os
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from groq import Groq

load_dotenv()

VISION_ENGINE = "gemini-2.5-flash"
LEADERBOARD_ENGINE = "llama3-70b-8192"

# Initialize Firebase Admin
firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS")
if firebase_creds_json:
    try:
        cert_dict = json.loads(firebase_creds_json)
        cred = credentials.Certificate(cert_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("INFO: Firebase initialized.")
    except Exception as e:
        print(f"CRITICAL: Firebase init failed. {e}")
        db = None
else:
    print("WARNING: FIREBASE_CREDENTIALS missing.")
    db = None

# Initialize AI Clients
api_keys_str = os.getenv("GOOGLE_API_KEYS", "")
API_KEYS = [key.strip() for key in api_keys_str.split(",") if key.strip()]

if not API_KEYS:
    raise ValueError("CRITICAL: No Google API keys found.")

current_key_index = 0

def get_gemini_client() -> genai.GenerativeModel:
    """Returns a newly configured Gemini client using the active API key."""
    genai.configure(api_key=API_KEYS[current_key_index])
    return genai.GenerativeModel(
        model_name=VISION_ENGINE,
        generation_config={"response_mime_type": "application/json"}
    )

gemini_model = get_gemini_client()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


app = FastAPI(title="Jeevdhara API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Database Operations

def get_user(username: str) -> dict:
    if not db: return {"uploads": 0, "points": 0, "badges": []}
    doc = db.collection('users').document(username).get()
    return doc.to_dict() if doc.exists else {"uploads": 0, "points": 0, "badges": []}

def save_user(username: str, data: dict):
    if db: db.collection('users').document(username).set(data)

def is_hash_seen(file_hash: str) -> bool:
    if not db: return False
    doc = db.collection('processed_hashes').document(file_hash).get()
    return doc.exists

def save_hash(file_hash: str):
    if db: 
        db.collection('processed_hashes').document(file_hash).set({
            "processed_at": firestore.SERVER_TIMESTAMP
        })

def get_top_users(limit: int = 10) -> List[tuple]:
    if not db: return []
    users_ref = db.collection('users').order_by('points', direction=firestore.Query.DESCENDING).limit(limit)
    return [(doc.id, doc.to_dict()) for doc in users_ref.stream()]


# Core Logic

def get_image_hash(file_contents: bytes) -> str:
    return hashlib.md5(file_contents).hexdigest()

def smart_unwrap(data: Any) -> list:
    """Extracts target arrays from deeply nested JSON structures."""
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        list_values = [v for v in data.values() if isinstance(v, list)]
        if len(list_values) > 0 and len(list_values) == len(data.keys()):
            extracted = []
            for city_key, val_list in data.items():
                for item in val_list:
                    if isinstance(item, dict): item['_mapped_city'] = city_key 
                    extracted.append(item)
            return extracted
        for val in data.values():
            if isinstance(val, list): return val
            elif isinstance(val, dict):
                for sub_val in val.values():
                    if isinstance(sub_val, list): return sub_val
        return [data]
    return []

def read_json_safe(filepath: str) -> list:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return smart_unwrap(json.load(f))
    except Exception as e:
        print(f"Skipping bad JSON: {filepath} - {e}")
        return []

def load_all_databases() -> dict:
    """Compiles geographic data into memory on startup."""
    database = {"parks": [], "depts": [], "ngos": [], "lands": [], "fauna": [], "flora": []}
    base_dir = "data"
    
    if not os.path.exists(base_dir): return database

    for state_folder in os.listdir(base_dir):
        state_path = os.path.join(base_dir, state_folder)
        if not os.path.isdir(state_path): continue

        database["parks"].extend(read_json_safe(os.path.join(state_path, "forest.json")))
        database["ngos"].extend(read_json_safe(os.path.join(state_path, "ngo.json")))
        database["lands"].extend(read_json_safe(os.path.join(state_path, "land.json")))
        
        for d_file in ["forest_department.json", "forest_division.json"]:
            path = os.path.join(state_path, d_file)
            if os.path.exists(path):
                database["depts"].extend(read_json_safe(path))
                break 

        for endg_folder in ["endangered_species", "endenger_species"]:
            endg_path = os.path.join(state_path, endg_folder)
            if os.path.exists(endg_path):
                database["fauna"].extend(read_json_safe(os.path.join(endg_path, "fauna.json")))
                database["flora"].extend(read_json_safe(os.path.join(endg_path, "flora.json")))
                
    return database

global_db = load_all_databases()

def get_rich_context(city_name: str) -> dict:
    c_lower = city_name.lower()
    
    def match(obj, keys):
        if not isinstance(obj, dict): return False
        return any(c_lower in str(obj.get(k, '')).lower() for k in keys)

    return {
        "assigned_park": next((p for p in global_db["parks"] if match(p, ['district', 'forestName'])), {}),
        "assigned_department": next((d for d in global_db["depts"] if match(d, ['address', 'headquarters', 'district', 'name'])), {}),
        "nearby_ngos": [n for n in global_db["ngos"] if match(n, ['workingDistricts', 'headquarters'])][:2],
        "nearby_degraded_lands": [l for l in global_db["lands"] if match(l, ['district', 'nearbyForest'])][:2],
        "local_endangered_fauna": [f for f in global_db["fauna"] if match(f, ['_mapped_city', 'statesFound', 'habitat'])][:3],
        "local_endangered_flora": [f for f in global_db["flora"] if match(f, ['_mapped_city', 'statesFound', 'habitat'])][:3]
    }

def search_species_in_db(sci_name: str, common_name: str) -> Optional[dict]:
    sl = sci_name.lower() if sci_name else ""
    cl = common_name.lower() if common_name else ""
    
    for f in global_db["fauna"] + global_db["flora"]:
        if (sl and sl in str(f.get('scientificName', '')).lower()) or (cl and cl in str(f.get('commonName', '')).lower()):
            return f
    return None

def update_user_rank(username: str, species_data: dict) -> dict:
    user = get_user(username)
    user["uploads"] += 1
    
    category = str(species_data.get('ecological_category', 'Unknown')).lower()
    if "native" in category: points_earned = 50  
    elif "invasive" in category: points_earned = 20  
    else: points_earned = 5   
        
    try: rarity = int(species_data.get('rarity_score', 0))
    except (ValueError, TypeError): rarity = 0

    if rarity >= 8: points_earned += 50 
        
    user["points"] += points_earned
    
    if user["uploads"] >= 1 and "Rookie Scout" not in user["badges"]: user["badges"].append("Rookie Scout")
    if user["points"] >= 100 and "Biodiversity Guardian" not in user["badges"]: user["badges"].append("Biodiversity Guardian")
    if user["points"] >= 300 and "Apex Ranger" not in user["badges"]: user["badges"].append("Apex Ranger")
    
    save_user(username, user)
    return user

def generate_with_fallback(prompt: str, image_data: bytes, mime_type: str) -> str:
    """Wraps Gemini generation to handle quota exhaustion across multiple keys."""
    global current_key_index, gemini_model
    max_retries = len(API_KEYS)
    
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content([prompt, {'mime_type': mime_type, 'data': image_data}])
            return response.text
        except ResourceExhausted:
            print(f"INFO: Key {current_key_index + 1} exhausted. Rotating.")
            current_key_index = (current_key_index + 1) % len(API_KEYS)
            gemini_model = get_gemini_client()
            time.sleep(1) 
        except Exception as e:
            raise e
            
    raise Exception("All configured API keys have exceeded their quotas.")


leaderboard_cache = {
    "message": "Welcome to the Jeevdhara Leaderboard! Keep exploring!",
    "top_user": None,
    "top_score": -1,
    "last_updated": 0.0
}

# Routes

@app.get("/leaderboard")
async def show_leaderboard():
    global leaderboard_cache
    
    top_contributors = get_top_users(limit=10)
    top_user = top_contributors[0][0] if top_contributors else "No one"
    top_score = top_contributors[0][1].get('points', 0) if top_contributors else 0
    current_time = time.time()
    
    needs_update = (
        top_user != leaderboard_cache["top_user"] or 
        top_score != leaderboard_cache["top_score"] or 
        (current_time - leaderboard_cache["last_updated"]) > 600
    )
    
    if needs_update and top_score > 0:
        gpt_prompt = f"Act as an Indian Chief Forest Ranger. Write a short, motivating 1-sentence quote for a leaderboard where '{top_user}' is winning with {top_score} conservation points."
        try:
            completion = groq_client.chat.completions.create(
                model=LEADERBOARD_ENGINE,
                messages=[{"role": "user", "content": gpt_prompt}]
            )
            leaderboard_cache["message"] = completion.choices[0].message.content.strip()
            leaderboard_cache["top_user"] = top_user
            leaderboard_cache["top_score"] = top_score
            leaderboard_cache["last_updated"] = current_time
        except Exception as e:
            print(f"Failed to refresh leaderboard message: {e}")

    return {
        "engine": f"{LEADERBOARD_ENGINE} (Cached)",
        "timestamp": datetime.now().isoformat(),
        "chief_ranger_message": leaderboard_cache["message"],
        "top_contributors": top_contributors
    }

@app.post("/analyze-biodiversity")
@limiter.limit("15/minute")
async def analyze_input(
    request: Request,
    file: UploadFile = File(None),          
    nearest_city: str = Form(None),         
    state_name: str = Form(None),            
    username: str = Form("Anonymous"),
    language: str = Form("english")
):
    try:
        if not file and not nearest_city:
            raise HTTPException(status_code=400, detail="Missing image or location parameters.")

        if nearest_city and not file:
            return {
                "status": "success",
                "mode": "city_exploration",
                "message": f"Retrieved data for {nearest_city}.",
                "location_context": get_rich_context(nearest_city)
            }

        contents = await file.read()
        
        safe_mime_type = file.content_type or "image/jpeg"
        if safe_mime_type == "application/octet-stream":
            safe_mime_type = "image/png" if file.filename and file.filename.lower().endswith('.png') else "image/jpeg"

        file_hash = get_image_hash(contents)
        is_duplicate = is_hash_seen(file_hash)
        ctx = get_rich_context(nearest_city) if nearest_city else None
        
        if ctx:
            ngo_names = [n.get("ngoName", "") for n in ctx["nearby_ngos"]]
            land_names = [l.get("landName", "") for l in ctx["nearby_degraded_lands"]]
            context_string = (
                f"--- LOCAL KNOWLEDGE BASE FOR {nearest_city.upper()} ---\n"
                f"Sanctuary: {ctx['assigned_park'].get('forestName', 'N/A')}\n"
                f"Degraded Lands: {', '.join(land_names) if land_names else 'None listed'}\n"
                f"Local NGOs: {', '.join(ngo_names) if ngo_names else 'None listed'}\n"
            )
            geography_rule = f"You operate in {state_name or 'India'}. Reference local laws."
        else:
            context_string = "No location provided. Determine species and general Indian habitat."
            geography_rule = "Provide analysis based on general Indian wildlife guidelines."

        gemini_prompt = (
            f"Act as the Chief Conservator of Forests.\nAnalyze this image.\n\n{context_string}\n\n"
            f"STEP 1: Identify species.\n"
            f"STEP 2: Detect if this is a downloaded internet/stock photo.\n"
            f"STEP 3: Generate a highly accurate conservation report in {language}.\n\n"
            f"RULES:\n"
            f"1. GEOGRAPHY: {geography_rule}\n"
            f"2. FAKE DETECTION: If downloaded/stock photo, set 'is_stock_photo' to true and explain in 'stock_photo_reason'.\n"
            f"3. ECOLOGY: Classify as 'Native', 'Invasive', 'Ornamental', or 'Domestic'.\n"
            f"4. DOMESTIC/ORNAMENTAL: Set threat_level to 'None', rarity_score to 0, requires_forest_guard_dispatch to false.\n\n"
            f"Return ONLY valid JSON with EXACT keys: common name, species, scientific_name, "
            f"is_stock_photo (bool), stock_photo_reason, ecological_category, legal_status, "
            f"suitability_for_reforestation (bool), immediate_action_steps (array), "
            f"threat_level (High/Medium/Low/None), rarity_score (int 0-10), requires_forest_guard_dispatch (bool)."
        )
        
        raw_response = generate_with_fallback(gemini_prompt, contents, safe_mime_type)
        final_analysis = json.loads(raw_response)

        db_match = search_species_in_db(final_analysis.get("scientific_name"), final_analysis.get("species"))
        if db_match:
            final_analysis["database_record"] = db_match

        user_stats = get_user(username)
        awarded_points = False
        message = "Points successfully recorded."

        if final_analysis.get("is_stock_photo", False):
            message = "No points awarded: Image detected as stock/downloaded."
        elif is_duplicate:
            message = "No points awarded: Image has already been analyzed."
        else:
            save_hash(file_hash)
            user_stats = update_user_rank(username, final_analysis)
            awarded_points = True

        return {
            "status": "success",
            "metadata": {
                "vision_engine": VISION_ENGINE,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "points_awarded": awarded_points
            },
            "location_context": ctx if ctx else "No location provided.",
            "biodiversity_analysis": final_analysis,
            "gamification": {
                "user": username,
                "points": user_stats["points"],
                "rank": user_stats["badges"][-1] if user_stats["badges"] else "None",
                "message": message
            }
        }

    except Exception as e:
        print(f"ERROR: {str(e)}") 
        raise HTTPException(status_code=500, detail="Internal processing error.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)