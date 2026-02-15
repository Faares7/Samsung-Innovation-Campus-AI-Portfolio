# How to Use the Online Retail Prediction System

### Step 1 — Install dependencies
pip install -r requirements.txt

### Step 2 — Run the API
uvicorn app:app --reload --host 0.0.0.0 --port 8000

### Step 3 — Open the frontend
Go to http://127.0.0.1:8000/static/index.html

### Step 4 — Enter the values
Fill in all form fields.

### Step 5 — Click “Predict”
The model prediction will appear instantly.

### Troubleshooting
- "Failed to fetch": API is not running.
- "CORS error": Add CORSMiddleware to FastAPI.
- "Model not found": Ensure model.pkl is inside `src/api/`.
