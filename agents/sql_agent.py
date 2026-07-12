"""
SQL Agent — Converts natural language to SQL queries.
Uses N8N webhook or direct OpenAI to generate SQL, then executes against the database.
"""

from openai import OpenAI
import config
from database import DatabaseManager

SYSTEM_PROMPT = """Kamu adalah SQL Query Generator. Tugasmu mengubah pertanyaan user tentang data lowongan pekerjaan menjadi SQL query.

Database schema:
Table: jobs
Columns:
- id (INTEGER, primary key)
- job_title (VARCHAR) — judul pekerjaan
- company_name (VARCHAR) — nama perusahaan
- location (VARCHAR) — lokasi kerja
- work_type (VARCHAR) — tipe: 'Full time', 'Paruh waktu', 'Kontrak/Temporer', 'Kasual'
- salary_raw (VARCHAR) — gaji dalam format text asli
- salary_min (FLOAT, nullable) — gaji minimum dalam Rupiah
- salary_max (FLOAT, nullable) — gaji maximum dalam Rupiah
- job_description (TEXT) — deskripsi pekerjaan
- scrape_timestamp (VARCHAR) — timestamp scraping

Rules:
1. Hanya generate SELECT queries (READ-ONLY). JANGAN buat INSERT, UPDATE, DELETE, DROP, ALTER.
2. Jawab HANYA dengan SQL query, tanpa penjelasan lain.
3. Gunakan LIKE untuk pencarian text (case-insensitive pakai LOWER()).
4. Limit results ke 20 jika tidak disebutkan.
5. Salary dalam Rupiah (contoh: 10000000 = Rp 10.000.000).
6. Untuk pertanyaan agregat, gunakan COUNT, AVG, MIN, MAX, GROUP BY sesuai kebutuhan."""


def generate_sql_query(natural_language_query: str) -> str:
    """
    Convert natural language question to SQL query using N8N or OpenAI.
    Returns the SQL query string.
    """
    # Try N8N first
    if config.is_n8n_configured():
        try:
            from n8n_client import generate_sql_query_n8n
            sql = generate_sql_query_n8n(natural_language_query)
            if sql and not sql.startswith("N8N error") and not sql.startswith("Tidak dapat"):
                # Clean up: remove markdown code blocks if present
                if sql.startswith("```"):
                    lines = sql.split("\n")
                    sql = "\n".join(lines[1:-1])
                return sql
        except Exception as e:
            return f"-- Error N8N: {str(e)}"

    # Fallback to local OpenAI
    if not config.is_openai_configured():
        return ""

    try:
        client = OpenAI(api_key=config.get_openai_api_key())
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": natural_language_query},
            ],
            temperature=0,
            max_tokens=500,
        )
        sql = response.choices[0].message.content.strip()
        # Clean up: remove markdown code blocks if present
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1])
        return sql
    except Exception as e:
        return f"-- Error: {str(e)}"


def query_jobs_natural_language(question: str) -> dict:
    """
    Full pipeline: natural language → SQL → execute → format results.

    Returns dict with:
    - "question": original question
    - "sql_query": generated SQL
    - "results": list of result dicts
    - "ai_explanation": AI explanation of results
    """
    result = {
        "question": question,
        "sql_query": "",
        "results": [],
        "ai_explanation": None,
    }

    if not config.is_openai_configured() and not config.is_n8n_configured():
        result["ai_explanation"] = "⚠️ OpenAI API key belum diatur. Masukkan API key di file .env"
        return result

    # Step 1: Generate SQL
    sql_query = generate_sql_query(question)
    result["sql_query"] = sql_query

    if sql_query.startswith("-- Error"):
        result["ai_explanation"] = sql_query
        return result

    # Safety check: only allow SELECT
    if not sql_query.strip().upper().startswith("SELECT"):
        result["ai_explanation"] = "⚠️ Query yang di-generate bukan SELECT query. Ditolak untuk keamanan."
        return result

    # Step 2: Execute SQL (always local)
    db = DatabaseManager()
    results = db.execute_raw_sql(sql_query)
    result["results"] = results

    # Step 3: AI explanation
    if results and "error" not in results[0]:
        if config.is_n8n_configured():
            try:
                from n8n_client import explain_sql_results_n8n
                explanation = explain_sql_results_n8n(question, sql_query, results)
                result["ai_explanation"] = explanation
            except Exception:
                result["ai_explanation"] = f"Ditemukan {len(results)} hasil."
        elif config.is_openai_configured():
            try:
                client = OpenAI(api_key=config.get_openai_api_key())
                context = f"Pertanyaan: {question}\nSQL: {sql_query}\nHasil ({len(results)} rows): {str(results[:10])}"

                response = client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "Kamu adalah data analyst. Jelaskan hasil query database ini dalam Bahasa Indonesia dengan ringkas dan informatif."},
                        {"role": "user", "content": context},
                    ],
                    temperature=0.5,
                    max_tokens=800,
                )
                result["ai_explanation"] = response.choices[0].message.content
            except Exception as e:
                result["ai_explanation"] = f"Ditemukan {len(results)} hasil."

    return result
