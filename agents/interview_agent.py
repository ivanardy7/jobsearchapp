"""
Mock Interview Agent — Simulates HR interview with AI.
Supports text-based and voice-based (OpenAI Whisper + TTS) interview.
Routes to N8N webhooks when configured, falls back to direct OpenAI.
Audio functions (transcribe/TTS) always use direct OpenAI (binary data).
"""

import io
from openai import OpenAI
import config


INTERVIEWER_PROMPT = """Kamu adalah seorang HR Interviewer profesional di perusahaan besar Indonesia. 
Kamu sedang melakukan mock interview dengan seorang kandidat.

Konteks:
- CV Kandidat sudah diberikan
- Posisi yang dilamar: {job_title} di {company_name}
- Deskripsi pekerjaan sudah diberikan

Aturan:
1. Tanyakan SATU pertanyaan interview pada satu waktu
2. Setelah kandidat menjawab, berikan feedback singkat dan lanjut pertanyaan berikutnya
3. Campurkan pertanyaan behavioral, technical, dan situational
4. Bersikap profesional tapi ramah
5. Gunakan Bahasa Indonesia (kecuali posisi mengharuskan Bahasa Inggris)
6. Setelah 5-7 pertanyaan, akhiri interview dan berikan ringkasan feedback

Format jawaban:
- Jika ini pertanyaan baru: langsung tanyakan pertanyaannya
- Jika sedang memberi feedback: berikan feedback singkat lalu pertanyaan selanjutnya
- Jika interview selesai: berikan summary dengan format:

## 📋 Ringkasan Interview
### Skor Keseluruhan: [X]/10
### Kelebihan:
- [point]
### Area Perbaikan:
- [point]
### Tips:
- [tip]"""


def start_interview(cv_text: str, job_info: dict) -> dict:
    """
    Start a mock interview session.

    Args:
        cv_text: User's CV content
        job_info: dict with job_title, company_name, job_description

    Returns dict with:
    - "response": AI's first question
    - "available": whether AI service is configured
    """
    # Try N8N first
    if config.is_n8n_configured():
        try:
            from n8n_client import start_interview_n8n
            ai_text = start_interview_n8n(cv_text, job_info)
            if ai_text and not ai_text.startswith("N8N error") and not ai_text.startswith("Tidak dapat"):
                return {"response": ai_text, "available": True}
            else:
                return {"response": f"Error: {ai_text}", "available": True}
        except Exception as e:
            return {"response": f"Error N8N: {str(e)}", "available": True}

    # Fallback to local OpenAI
    if not config.is_openai_configured():
        return {"response": None, "available": False}

    try:
        client = OpenAI(api_key=config.get_openai_api_key())

        system_prompt = INTERVIEWER_PROMPT.format(
            job_title=job_info.get("job_title", "Unknown Position"),
            company_name=job_info.get("company_name", "Unknown Company"),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""[INFO INTERVIEW]
CV Kandidat:
{cv_text[:3000]}

Deskripsi Pekerjaan:
{job_info.get('job_description', 'N/A')[:2000]}

Mulai interview sekarang. Perkenalkan diri kamu sebagai HR dan mulai dengan pertanyaan pertama.""",
            },
        ]

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=800,
        )

        return {
            "response": response.choices[0].message.content,
            "available": True,
        }
    except Exception as e:
        return {"response": f"Error: {str(e)}", "available": True}


def continue_interview(
    cv_text: str,
    job_info: dict,
    interview_history: list[dict],
    user_answer: str,
) -> dict:
    """
    Continue the mock interview with user's answer.

    Args:
        cv_text: User's CV content
        job_info: Job information dict
        interview_history: Previous Q&A messages
        user_answer: User's answer to the current question

    Returns dict with:
    - "response": AI's feedback + next question (or summary if interview is done)
    - "available": whether AI service is configured
    """
    # Try N8N first
    if config.is_n8n_configured():
        try:
            from n8n_client import continue_interview_n8n
            ai_text = continue_interview_n8n(cv_text, job_info, interview_history, user_answer)
            if ai_text and not ai_text.startswith("N8N error") and not ai_text.startswith("Tidak dapat"):
                return {"response": ai_text, "available": True}
            else:
                return {"response": f"Error: {ai_text}", "available": True}
        except Exception as e:
            return {"response": f"Error N8N: {str(e)}", "available": True}

    # Fallback to local OpenAI
    if not config.is_openai_configured():
        return {"response": None, "available": False}

    try:
        client = OpenAI(api_key=config.get_openai_api_key())

        system_prompt = INTERVIEWER_PROMPT.format(
            job_title=job_info.get("job_title", "Unknown Position"),
            company_name=job_info.get("company_name", "Unknown Company"),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"[KONTEKS]\nCV: {cv_text[:2000]}\nJob: {job_info.get('job_description', '')[:1000]}",
            },
            {
                "role": "assistant",
                "content": "Baik, saya sudah memahami profil kandidat dan posisi yang dilamar. Mari kita mulai interview.",
            },
        ]

        # Add interview history
        for msg in interview_history:
            messages.append(msg)

        # Add current answer
        messages.append({"role": "user", "content": user_answer})

        # Check if we should end the interview (after 6+ exchanges)
        user_count = sum(1 for m in interview_history if m["role"] == "user")
        if user_count >= 5:
            messages.append({
                "role": "system",
                "content": "Interview sudah cukup panjang. Berikan feedback terakhir dan RINGKASAN INTERVIEW dengan skor keseluruhan.",
            })

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1200,
        )

        return {
            "response": response.choices[0].message.content,
            "available": True,
        }
    except Exception as e:
        return {"response": f"Error: {str(e)}", "available": True}


# ─── Audio Functions (always direct OpenAI, not via N8N) ──


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio to text using OpenAI Whisper.
    Returns transcribed text.
    Always uses direct OpenAI (binary audio data is not suitable for N8N webhooks).
    """
    if not config.is_openai_configured():
        return ""

    try:
        client = OpenAI(api_key=config.get_openai_api_key())
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="id",  # Indonesian
        )
        return transcript.text
    except Exception as e:
        return f"[Transcription error: {str(e)}]"


def text_to_speech(text: str) -> bytes:
    """
    Convert text to speech using OpenAI TTS.
    Returns audio bytes (mp3).
    Always uses direct OpenAI (binary audio output is not suitable for N8N webhooks).
    """
    if not config.is_openai_configured():
        return b""

    try:
        client = OpenAI(api_key=config.get_openai_api_key())
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",  # Natural female voice, good for Indonesian
            input=text,
        )
        return response.content
    except Exception as e:
        return b""
