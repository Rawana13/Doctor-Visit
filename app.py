import os
import json
import uuid
import secrets
from datetime import datetime, timezone
from urllib.parse import urlencode

import httpx
import gradio as gr
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from groq import Groq
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────
groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Epic FHIR config ──────────────────────────────────────────────────────────
EPIC_BASE      = "https://fhir.epic.com/interconnect-fhir-oauth"
EPIC_FHIR_R4   = f"{EPIC_BASE}/api/FHIR/R4"
EPIC_AUTH_URL  = f"{EPIC_BASE}/oauth2/authorize"
EPIC_TOKEN_URL = f"{EPIC_BASE}/oauth2/token"
REDIRECT_URI   = "http://localhost:7860/callback"
EPIC_CLIENT_ID = os.getenv("EPIC_CLIENT_ID", "")

EPIC_SCOPES = (
    "openid fhirUser "
    "patient/Appointment.read patient/Appointment.write "
    "patient/Slot.read patient/Schedule.read patient/Patient.read "
    "launch/patient"
)

SPECIALTY_CODES = {
    "cardiology": "394579002", "primary care": "394814009",
    "dermatology": "394582007", "orthopedics": "394801008",
    "neurology": "394591006",  "endocrinology": "394583002",
    "gastroenterology": "394584008", "gynecology": "394586005",
    "ophthalmology": "394594003", "psychiatry": "394587001",
    "pulmonology": "394572006", "rheumatology": "394810000",
    "urology": "394612005",
}

sessions: dict     = {}   # session_id → {access_token, patient_id}
oauth_states: dict = {}   # state → {session_id, specialty}

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
  "todos": ["Specific action the patient needs to take"],
  "follow_up_appointments": [
    {{
      "reason": "why the follow-up is needed",
      "specialty": "type of doctor or department",
      "timeframe": "when (e.g., in 2 weeks, in 3 months)"
    }}
  ],
  "important_notes": ["Any warnings, things to watch for, or lifestyle changes mentioned"]
}}

Only include sections relevant to the transcript. Return only valid JSON, no extra text."""


# ── Core logic ────────────────────────────────────────────────────────────────
def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        result = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
        )
    return result.text


def analyze_transcript(transcript: str) -> dict:
    message = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": CLAUDE_PROMPT.format(transcript=transcript)}]
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ── HTML formatters ───────────────────────────────────────────────────────────
def render_summary(data: dict) -> str:
    diagnosis = data.get("diagnosis_or_reason", "")
    summary   = data.get("summary", "No summary available.")
    badge = (f'<span style="display:inline-block;background:#dbeafe;color:#1d4ed8;'
             f'padding:4px 12px;border-radius:20px;font-size:13px;font-weight:600;margin-top:10px">'
             f'{diagnosis}</span>') if diagnosis else ""
    return f"""
<div style="background:#f8fafc;border-left:4px solid #4f46e5;padding:18px 20px;
            border-radius:8px;font-size:15px;line-height:1.7;color:#1e293b">
  {summary}
</div>
{badge}"""


def render_todos(data: dict) -> str:
    todos = list(data.get("todos", []))
    for med in data.get("medications", []):
        if med.get("needs_pickup"):
            todos.insert(0, f"Pick up prescription: {med['name']}")
    for appt in data.get("follow_up_appointments", []):
        todos.append(f"Schedule follow-up: {appt.get('specialty','appointment')} ({appt.get('timeframe','')})")

    if not todos:
        return "<p style='color:#94a3b8;font-style:italic'>No action items found.</p>"

    items = ""
    for todo in todos:
        items += f"""
<div style="display:flex;align-items:flex-start;gap:10px;padding:12px 14px;
            background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:8px">
  <span style="font-size:16px;margin-top:1px">☐</span>
  <span style="font-size:14px;color:#1e293b">{todo}</span>
</div>"""
    return items


def render_medications(data: dict) -> str:
    meds = data.get("medications", [])
    if not meds:
        return "<p style='color:#94a3b8;font-style:italic'>No medications prescribed.</p>"

    html = ""
    for med in meds:
        pickup = ('<span style="background:#c2410c;color:white;font-size:11px;font-weight:700;'
                  'padding:2px 8px;border-radius:20px;margin-left:8px">Needs pickup</span>'
                  if med.get("needs_pickup") else "")
        html += f"""
<div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;
            padding:14px 16px;margin-bottom:10px">
  <div style="font-weight:700;font-size:15px;color:#9a3412">{med.get('name','')}{pickup}</div>
  {'<div style="font-size:13px;color:#7c2d12;margin-top:4px"><b>Dosage:</b> ' + med.get('dosage','') + '</div>' if med.get('dosage') else ''}
  {'<div style="font-size:13px;color:#7c2d12;margin-top:2px"><b>Instructions:</b> ' + med.get('instructions','') + '</div>' if med.get('instructions') else ''}
</div>"""
    return html


def render_appointments(data: dict, session_id: str) -> str:
    appts = data.get("follow_up_appointments", [])
    if not appts:
        return "<p style='color:#94a3b8;font-style:italic'>No follow-up appointments needed.</p>"

    html = ""
    for appt in appts:
        specialty = appt.get("specialty", "Follow-up")
        reason    = appt.get("reason", "")
        timeframe = appt.get("timeframe", "")

        if session_id:
            btn = f"""<button onclick="scheduleAppt('{specialty}','{session_id}')"
              style="margin-top:10px;padding:8px 16px;background:#166534;color:white;
                     border:none;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer">
              Schedule via MyChart
            </button>"""
        else:
            state = secrets.token_urlsafe(16)
            sid   = str(uuid.uuid4())
            oauth_states[state] = {"session_id": sid, "specialty": specialty}
            params = {
                "response_type": "code", "client_id": EPIC_CLIENT_ID,
                "redirect_uri": REDIRECT_URI, "scope": EPIC_SCOPES,
                "state": state, "aud": EPIC_FHIR_R4,
            }
            auth_url = f"{EPIC_AUTH_URL}?{urlencode(params)}"
            btn = f"""<a href="{auth_url}" target="_blank"
              style="display:inline-block;margin-top:10px;padding:8px 16px;background:#166534;
                     color:white;border-radius:6px;font-size:13px;font-weight:600;
                     text-decoration:none">
              Connect MyChart &amp; Schedule
            </a>"""

        html += f"""
<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
            padding:14px 16px;margin-bottom:10px">
  <div style="font-weight:700;font-size:15px;color:#166534">{specialty}</div>
  {'<div style="font-size:13px;color:#15803d;margin-top:4px">' + reason + '</div>' if reason else ''}
  {'<div style="font-size:13px;color:#15803d;margin-top:2px"><b>When:</b> ' + timeframe + '</div>' if timeframe else ''}
  {btn}
</div>"""
    return html


def render_notes(data: dict) -> str:
    notes = data.get("important_notes", [])
    if not notes:
        return ""
    items = "".join(
        f'<li style="margin-bottom:6px">{n}</li>' for n in notes
    )
    return f"""
<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:14px 18px">
  <div style="font-weight:700;color:#92400e;margin-bottom:8px">⚠️ Important Notes</div>
  <ul style="margin:0;padding-left:18px;font-size:14px;color:#78350f;line-height:1.7">{items}</ul>
</div>"""


# ── Main processing function ──────────────────────────────────────────────────
def process_visit(audio, session_state):
    if audio is None:
        return (
            "<p style='color:#e11d48'>Please record or upload audio first.</p>",
            "", "", "", "", gr.update(visible=False), session_state
        )

    try:
        # Transcribe
        transcript = transcribe_audio(audio)
    except Exception as e:
        return (
            f"<p style='color:#e11d48'>Transcription failed: {e}</p>",
            "", "", "", "", gr.update(visible=False), session_state
        )

    try:
        # Analyze
        data = analyze_transcript(transcript)
    except Exception as e:
        return (
            f"<p style='color:#e11d48'>Analysis failed: {e}</p>",
            "", "", "", "", gr.update(visible=False), session_state
        )

    session_id = session_state.get("session_id", "")

    return (
        render_summary(data),
        render_todos(data),
        render_medications(data),
        render_appointments(data, session_id),
        render_notes(data),
        gr.update(visible=True),   # show transcript accordion
        {**session_state, "last_data": data, "transcript": transcript},
    )


def show_transcript(session_state):
    return session_state.get("transcript", "No transcript yet.")


# ── Gradio UI ─────────────────────────────────────────────────────────────────
HEADER_HTML = """
<div style="text-align:center;padding:24px 0 8px">
  <div style="font-size:48px">🩺</div>
  <h1 style="font-size:28px;font-weight:800;color:#1e293b;margin:8px 0 4px">Doc Assistant</h1>
  <p style="color:#64748b;font-size:15px;margin:0">
    Record your appointment → get a plain-language summary, action items, and scheduling
  </p>
  <div style="margin-top:10px;padding:8px 16px;background:#fef3c7;border:1px solid #fde68a;
              border-radius:8px;display:inline-block;font-size:12px;color:#92400e">
    ⚠️ This summary is AI-generated. Always confirm instructions with your healthcare provider.
  </div>
</div>"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),
    title="Doc Assistant",
    css="""
    .gradio-container { max-width: 860px !important; margin: auto; }
    .gr-button-primary { background: #4f46e5 !important; }
    footer { display: none !important; }
    """
) as demo:

    session_state = gr.State({})

    gr.HTML(HEADER_HTML)

    # ── Record / Upload ───────────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("### 🎙️ Record or Upload Your Appointment")
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Appointment Audio",
        )
        analyze_btn = gr.Button("✨ Analyze Appointment", variant="primary", size="lg")

    # ── Results ───────────────────────────────────────────────────────────────
    with gr.Column(visible=True):
        gr.Markdown("---")
        gr.Markdown("### 📋 Visit Summary")
        summary_out = gr.HTML()

        gr.Markdown("### ✅ Action Items")
        todos_out = gr.HTML()

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💊 Medications")
                meds_out = gr.HTML()
            with gr.Column():
                gr.Markdown("### 📅 Follow-up Appointments")
                appts_out = gr.HTML()

        notes_out = gr.HTML()

        with gr.Accordion("🗒️ Full Transcript", open=False, visible=False) as transcript_acc:
            transcript_out = gr.Textbox(
                label="", lines=8, interactive=False,
                placeholder="Transcript will appear here..."
            )

    # ── Wire up ───────────────────────────────────────────────────────────────
    analyze_btn.click(
        fn=process_visit,
        inputs=[audio_input, session_state],
        outputs=[
            summary_out, todos_out, meds_out, appts_out,
            notes_out, transcript_acc, session_state
        ]
    )

    transcript_acc.expand(
        fn=show_transcript,
        inputs=[session_state],
        outputs=[transcript_out]
    )


# ── Mount on FastAPI for Epic OAuth callback ──────────────────────────────────
fastapi_app = FastAPI(title="Doc Assistant")

@fastapi_app.get("/callback")
async def epic_callback(
    code: str = Query(None), state: str = Query(None), error: str = Query(None)
):
    if error:
        return HTMLResponse(_callback_html("error", f"Auth error: {error}"))
    if not code or not state or state not in oauth_states:
        return HTMLResponse(_callback_html("error", "Invalid OAuth state"))

    pending    = oauth_states.pop(state)
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
    sessions[session_id] = {
        "access_token": token_data["access_token"],
        "patient_id":   token_data.get("patient", ""),
        "specialty":    pending.get("specialty", ""),
    }
    return HTMLResponse(_callback_html("success", session_id))


def _callback_html(status: str, payload: str) -> str:
    msg = "Connected to MyChart! You can close this window." if status == "success" else f"Error: {payload}"
    return f"""<!DOCTYPE html><html><body style="font-family:sans-serif;text-align:center;padding:60px">
<div style="font-size:48px">{"✅" if status == "success" else "❌"}</div>
<h2>{msg}</h2>
<script>
  if (window.opener) {{
    window.opener.postMessage({{"epic_auth": "{status}", "payload": "{payload}"}}, "*");
    window.close();
  }}
</script>
</body></html>"""


# Mount Gradio on FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
