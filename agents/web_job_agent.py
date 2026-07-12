"""
Web Job Agent — Uses OpenAI to generate realistic job suggestions
based on user's CV analysis. Suggestions are labeled by platform source.
"""

import json
import config
from openai import OpenAI


SUGGESTION_PROMPT = """Kamu adalah AI career advisor Indonesia. Berdasarkan CV berikut, buatlah TEPAT 9 saran lowongan pekerjaan yang realistis dan relevan untuk kandidat ini di Indonesia.

Bagi menjadi 3 kelompok sumber:
- 3 lowongan berlabel sumber "LinkedIn"
- 3 lowongan berlabel sumber "JobStreet"  
- 3 lowongan berlabel sumber "Google Jobs"

Untuk setiap lowongan, berikan data berikut dalam format JSON:
- job_title: Judul posisi (realistis, sesuai industri Indonesia)
- company_name: Nama perusahaan (gunakan perusahaan nyata di Indonesia yang relevan)
- location: Lokasi kerja (kota di Indonesia)
- work_type: Tipe kerja (Full-time / Part-time / Contract / Remote / Hybrid)
- salary: Kisaran gaji (dalam Rupiah, contoh: "Rp 8.000.000 - Rp 12.000.000/bulan") atau "Negotiable"
- description: Deskripsi singkat pekerjaan (2-3 kalimat)
- source: Sumber platform ("LinkedIn" / "JobStreet" / "Google Jobs")

PENTING:
- Gunakan nama perusahaan yang NYATA dan terkenal di Indonesia
- Sesuaikan level posisi dengan pengalaman di CV
- Variasikan lokasi (tidak semua Jakarta)
- Buat deskripsi yang spesifik dan relevan dengan skill di CV

Kembalikan HANYA JSON array (tanpa markdown code block), contoh format:
[{"job_title": "...", "company_name": "...", "location": "...", "work_type": "...", "salary": "...", "description": "...", "source": "LinkedIn"}, ...]
"""


def generate_job_suggestions(cv_text: str) -> list[dict]:
    """
    Use OpenAI to generate realistic job suggestions based on CV.
    Returns a list of job dicts with source labels.
    """
    api_key = config.get_openai_api_key()
    if not api_key:
        return []

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SUGGESTION_PROMPT},
                {"role": "user", "content": f"CV Kandidat:\n\n{cv_text[:4000]}"},
            ],
            temperature=0.7,
            max_tokens=3000,
        )

        raw = response.choices[0].message.content.strip()

        # Clean up markdown code block if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

        jobs = json.loads(raw)

        if not isinstance(jobs, list):
            return []

        # Normalize and validate each job
        valid_jobs = []
        for job in jobs:
            if isinstance(job, dict) and job.get("job_title"):
                valid_jobs.append({
                    "job_title": job.get("job_title", "Unknown Position"),
                    "company_name": job.get("company_name", "Unknown Company"),
                    "location": job.get("location", "Indonesia"),
                    "work_type": job.get("work_type", "Full-time"),
                    "salary": job.get("salary", "Negotiable"),
                    "description": job.get("description", ""),
                    "source": job.get("source", "Google Jobs"),
                })

        return valid_jobs

    except json.JSONDecodeError:
        return []
    except Exception:
        return []
