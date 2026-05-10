import os
import json
import uuid
import tempfile
import secrets
import httpx
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import openai
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Doc Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Epic FHIR config ──────────────────────────────────────────────────────────
EPIC_BASE        = "https://fhir.epic.com/interconnect-fhir-oauth"
EPIC_FHIR_R4     = f"{EPIC_BASE}/api/FHIR/R4"
EPIC_AUTH_URL    = f"{EPIC_BASE}/oauth2/authorize"
EPIC_TOKEN_URL   = f"{EPIC_BASE}/oauth2/token"
REDIRECT_URI     = "http://localhost:8000/callback"
EPIC_CLIENT_ID   = os.getenv("EPIC_CLIENT_ID", "")

EPIC_SCOPES = (
    "openid fhirUser "
    "patient/Appointment.read patient/Appointment.write "
    "patient/Slot.read patient/Schedule.read patient/Patient.read "
    "launch/patient"
)

# In-memory session store  {session_id: {access_token, patient_id, expires_at}}
sessions: dict[str, dict] = {}
# OAuth state → pending session id
oauth_states: dict[str, str] = {}

# ── Specialty → SNOMED service-type code mapping ─────────────────────────────
SPECIALTY_CODES = {
    "cardiology":          "394579002",
    "primary care":        "394814009",
    "general practice":    "394814009",
    "dermatology":         "394582007",
    "orthopedics":         "394801008",
    "neurology":           "394591006",
    "endocrinology":       "394583002",
    "gastroenterology":    "394584008",
    "gynecology":          "394586005",
    "ophthalmology":       "394594003",
    "psychiatry":          "394587001",
    "pulmonology":         "394572006",
    "rheumatology":        "394810000",
    "urology":             "394612005",
}

def specialty_code(specialty: str) -> str | None:
    return SPECIALTY_CODES.get(specialty.lower().strip())

# ── Claude prompt ─────────────────────────────────────────────────────────────
CLAUDE_PROMPT = """You are a helpful medical assistant. A patient just had a doctor's appointment and the visit was recorded.
Your job is to help them understand what happened and what they need to do next.

Here is the transcript of the appointment:
<transcript>
{transcript}
</transcript>

Please analyze this and return a JSON object with the following structure:
{{
  "summary": "A clear, plain-language summary of the visit in 3-5 sentences. Avoid medical jargon. Write as if explaining to a friend.",
  "diagnosis_or_reason": "What the visit was about or any diagnosis mentioned, in simple terms.",
  "medications": [
    {{
      "name": "medication name",
      "dosage": "dosage if mentioned",
      "instructions": "how and when to take it",
      "needs_pickup": true
    }}
  ],
  "todos": [
    "Specific action the patient needs to take"
  ],
  "follow_up_appointments": [
    {{
      "reason": "why the follow-up is needed",
      "specialty": "type of doctor or department",
      "timeframe": "when (e.g., in 2 weeks, in 3 months)"
    }}
  ],
  "important_notes": [
    "Any warnings, things to watch for, or lifestyle changes mentioned"
  ],
  "questions_for_next_visit": [
    "Any unresolved questions the patient might want to ask next time"
  ]
}}

Only include sections that are relevant based on the transcript. If no medications were prescribed, return an empty array.
Return only valid JSON, no extra text.
"""


# ── Models ────────────────────────────────────────────────────────────────────
class TranscriptRequest(BaseModel):
    transcript: str

class BookRequest(BaseModel):
    session_id: str
    slot_id: str
    specialty: str
    reason: str = ""


# ── Transcription & Analysis ──────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            result = openai_client.audio.transcriptions.create(
                model="whisper-1", file=f, language="en"
            )
        return {"transcript": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        os.unlink(tmp_path)


@app.post("/analyze")
async def analyze_transcript(request: TranscriptRequest):
    if not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty")
    try:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": CLAUDE_PROMPT.format(transcript=request.transcript)}]
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse Claude response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


# ── Epic OAuth2 (SMART on FHIR) ───────────────────────────────────────────────
@app.get("/auth/login")
def epic_login(specialty: str = ""):
    """Return the Epic OAuth2 authorization URL for the frontend to open."""
    if not EPIC_CLIENT_ID:
        raise HTTPException(status_code=500, detail="EPIC_CLIENT_ID not set in .env")

    state = secrets.token_urlsafe(16)
    session_id = str(uuid.uuid4())
    oauth_states[state] = {"session_id": session_id, "specialty": specialty}

    params = {
        "response_type": "code",
        "client_id": EPIC_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": EPIC_SCOPES,
        "state": state,
        "aud": EPIC_FHIR_R4,
    }
    from urllib.parse import urlencode
    url = f"{EPIC_AUTH_URL}?{urlencode(params)}"
    return {"auth_url": url, "session_id": session_id}


@app.get("/callback")
async def epic_callback(code: str = Query(None), state: str = Query(None), error: str = Query(None)):
    """Handle Epic OAuth2 callback, exchange code for access token."""
    if error:
        return HTMLResponse(_callback_html("error", f"Epic auth error: {error}"))

    if not code or not state or state not in oauth_states:
        return HTMLResponse(_callback_html("error", "Invalid OAuth state"))

    pending = oauth_states.pop(state)
    session_id = pending["session_id"]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            EPIC_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": EPIC_CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if resp.status_code != 200:
        return HTMLResponse(_callback_html("error", f"Token exchange failed: {resp.text}"))

    token_data = resp.json()
    patient_id = token_data.get("patient", "")

    sessions[session_id] = {
        "access_token": token_data["access_token"],
        "patient_id": patient_id,
        "specialty": pending.get("specialty", ""),
    }

    return HTMLResponse(_callback_html("success", session_id))


def _callback_html(status: str, payload: str) -> str:
    """Tiny page that posts result back to the opener window."""
    return f"""<!DOCTYPE html><html><body>
<script>
  window.opener.postMessage({{"epic_auth": "{status}", "payload": "{payload}"}}, "*");
  window.close();
</script>
<p>{"Connected! This window will close." if status == "success" else "Error: " + payload}</p>
</body></html>"""


# ── Epic FHIR — Slot Search ───────────────────────────────────────────────────
@app.get("/fhir/slots")
async def get_slots(
    session_id: str,
    specialty: str,
    date_from: str = Query(default=None, description="YYYY-MM-DD"),
    date_to:   str = Query(default=None, description="YYYY-MM-DD"),
):
    session = _require_session(session_id)
    token = session["access_token"]
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"}

    # Default: next 14 days
    start = date_from or datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    end   = date_to   or datetime.now(timezone.utc).replace(
        year=datetime.now().year + (1 if datetime.now().month == 12 else 0)
    ).strftime("%Y-%m-%dT23:59:59Z")

    code = specialty_code(specialty)
    params = {"status": "free", "start": f"ge{start}", "start": f"le{end}"}
    if code:
        params["service-type"] = code

    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{EPIC_FHIR_R4}/Slot", params=params, headers=headers)

    if resp.status_code == 401:
        raise HTTPException(status_code=401, detail="Session expired — please reconnect MyChart")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"FHIR slot search failed: {resp.text}")

    bundle = resp.json()
    slots = []
    for entry in bundle.get("entry", []):
        r = entry.get("resource", {})
        slots.append({
            "id": r.get("id"),
            "start": r.get("start"),
            "end": r.get("end"),
            "status": r.get("status"),
            "schedule_ref": (r.get("schedule") or {}).get("reference", ""),
        })

    return {"slots": slots, "total": len(slots)}


# ── Epic FHIR — Book Appointment ──────────────────────────────────────────────
@app.post("/fhir/appointment")
async def book_appointment(req: BookRequest):
    session = _require_session(req.session_id)
    token      = session["access_token"]
    patient_id = session["patient_id"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/fhir+json",
        "Accept": "application/fhir+json",
    }

    code = specialty_code(req.specialty)
    appointment = {
        "resourceType": "Appointment",
        "status": "booked",
        "serviceType": [{"coding": [{"system": "http://snomed.info/sct", "code": code}]}] if code else [],
        "description": req.reason or f"Follow-up: {req.specialty}",
        "slot": [{"reference": f"Slot/{req.slot_id}"}],
        "participant": [
            {"actor": {"reference": f"Patient/{patient_id}"}, "status": "accepted"}
        ],
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{EPIC_FHIR_R4}/Appointment",
            json=appointment,
            headers=headers,
        )

    if resp.status_code == 401:
        raise HTTPException(status_code=401, detail="Session expired — please reconnect MyChart")
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=f"Booking failed: {resp.text}")

    booked = resp.json()
    return {
        "appointment_id": booked.get("id"),
        "status": booked.get("status"),
        "start": booked.get("start"),
        "end": booked.get("end"),
    }


# ── Epic FHIR — List Patient Appointments ────────────────────────────────────
@app.get("/fhir/appointments")
async def list_appointments(session_id: str):
    session = _require_session(session_id)
    token      = session["access_token"]
    patient_id = session["patient_id"]
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"}

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{EPIC_FHIR_R4}/Appointment",
            params={"patient": patient_id, "status": "booked,arrived,fulfilled"},
            headers=headers,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"FHIR error: {resp.text}")

    bundle = resp.json()
    appts = []
    for entry in bundle.get("entry", []):
        r = entry.get("resource", {})
        appts.append({
            "id": r.get("id"),
            "status": r.get("status"),
            "start": r.get("start"),
            "end": r.get("end"),
            "description": r.get("description", ""),
        })
    return {"appointments": appts}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _require_session(session_id: str) -> dict:
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated with MyChart. Please connect first.")
    return session


@app.get("/health")
def health():
    return {"status": "ok"}
