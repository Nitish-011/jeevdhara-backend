# 🌿 Jeevdhara API (Backend)

An AI-powered biodiversity classification and gamification microservice built with FastAPI and Google Gemini. This backend supports the Jeevdhara Flutter app by processing ecological data, identifying species, and managing a conservation leaderboard.

## 🚀 Features
* **Multimodal AI Classification:** Uses Gemini 2.5 Flash to identify flora/fauna and assess ecological impact (Native vs. Invasive).
* **Anti-Cheat Detection:** Hashes images to prevent duplicate point farming and uses AI to flag downloaded stock photos.
* **Gamification Engine:** Awards points and badges based on species rarity and ecological category.
* **Serverless Database:** Stores user ranks and image hashes securely in Firebase Firestore.
* **Key Rotation:** Built-in round-robin fallback for Google API keys to manage Free Tier quotas.

## 🛠️ Tech Stack
* **Framework:** FastAPI (Python)
* **AI Models:** Google Gemini 2.5 Flash (Vision), Groq Llama 3 (Leaderboard Agent)
* **Database:** Firebase Admin SDK (Firestore)
* **Hosting:** Ready for Render/Railway deployment

## ⚙️ Local Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Add a `.env` file with your `GOOGLE_API_KEYS`, `GROQ_API_KEY`, and `FIREBASE_CREDENTIALS`.
4. Run the server: `uvicorn main:app --reload`