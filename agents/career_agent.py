"""
Career Consultant Agent — AI career advisor for chat-based consultation.
Multi-turn conversation about career goals, aspirations, and skill development.
Supports N8N webhook routing or direct OpenAI calls.
"""

from openai import OpenAI
import config


SYSTEM_PROMPT = """Kamu adalah Career Consultant AI yang berpengalaman. Tugasmu adalah membantu user mendiskusikan karir dan cita-cita mereka.

Konteks: User telah meng-upload CV mereka. Berdasarkan CV tersebut, bantu mereka:
1. Memahami posisi karir mereka saat ini
2. Mengeksplorasi cita-cita dan tujuan karir
3. Mengidentifikasi skill gap dan cara mengatasinya
4. Merekomendasikan langkah-langkah konkret untuk mencapai tujuan
5. Memberikan insight tentang tren industri yang relevan

Rules:
- Jawab dalam Bahasa Indonesia (kecuali user berbicara dalam Bahasa Inggris)
- Bersikap supportive dan encouraging
- Berikan saran yang spesifik dan actionable
- Tanyakan pertanyaan follow-up untuk memahami user lebih baik
- Jangan menghakimi pilihan karir user"""


def get_career_response(
    cv_text: str,
    chat_history: list[dict],
    user_message: str,
    target_job: dict = None,
) -> dict:
    """
    Generate career consultation response.

    Args:
        cv_text: The user's CV content
        chat_history: List of {"role": "user"/"assistant", "content": "..."} messages
        user_message: Current user message
        target_job: Optional dict with targeted job info (job_title, company_name, job_description)

    Returns dict with:
    - "response": AI response text
    - "available": whether AI service is configured
    """
    # Check if we should route to N8N (only for non-database questions)
    use_n8n_path = config.is_n8n_configured()
    if use_n8n_path:
        db_keywords = ["gaji", "salary", "rata-rata", "average", "jumlah", "total", "lowongan", "database", "db", "statistik", "banyak", "berapa"]
        is_db_query = any(kw in user_message.lower() for kw in db_keywords)
        
        if not is_db_query:
            try:
                from n8n_client import career_chat_n8n
                ai_text = career_chat_n8n(cv_text, chat_history, user_message, target_job=target_job)
                if ai_text and not ai_text.startswith("N8N error") and not ai_text.startswith("Tidak dapat"):
                    return {"response": ai_text, "available": True}
            except Exception:
                # Fallback to local OpenAI on N8N failure
                pass

    # Fallback to local OpenAI
    if not config.is_openai_configured():
        return {
            "response": None,
            "available": False,
        }

    try:
        client = OpenAI(api_key=config.get_openai_api_key())

        # Build system prompt with target job context
        enhanced_prompt = SYSTEM_PROMPT
        if target_job:
            enhanced_prompt += f"""\n\nKONTEKS PENTING: User menargetkan posisi spesifik:
- Posisi: {target_job.get('job_title', 'N/A')}
- Perusahaan: {target_job.get('company_name', 'N/A')}

Fokuskan saran karir kamu untuk membantu user mencapai posisi tersebut.
Berikan insight tentang skill yang dibutuhkan, cara mempersiapkan diri, dan langkah konkret menuju posisi tersebut.
Jika user menanyakan data/statistik tentang lowongan kerja (seperti rata-rata gaji, jumlah lowongan kerja, dll.), gunakan tool `query_job_database_statistics` untuk mencari data di database Aiven SQL, lalu jelaskan hasilnya kepada user."""
        else:
            enhanced_prompt += "\n\nJika user menanyakan data/statistik tentang lowongan kerja (seperti rata-rata gaji, jumlah lowongan kerja, dll.), gunakan tool `query_job_database_statistics` untuk mencari data di database Aiven SQL, lalu jelaskan hasilnya kepada user."

        # Build tools schema
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_job_database_statistics",
                    "description": (
                        "Gunakan fungsi ini untuk mencari statistik data lowongan pekerjaan riil dari database Aiven SQL "
                        "(seperti mencari rata-rata gaji, jumlah lowongan per lokasi/perusahaan, tipe pekerjaan, "
                        "atau daftar lowongan spesifik). Input berupa pertanyaan bahasa alami dalam Bahasa Indonesia."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Pertanyaan spesifik dalam Bahasa Indonesia tentang data lowongan pekerjaan, contoh: 'Berapa rata-rata gaji sales?', 'Tampilkan 5 perusahaan dengan lowongan terbanyak', 'Berapa jumlah lowongan di Jakarta Barat?'",
                            }
                        },
                        "required": ["question"],
                    },
                },
            }
        ]

        # Build messages
        messages = [
            {"role": "system", "content": enhanced_prompt},
        ]

        # Add CV context as first user message if available
        if cv_text:
            messages.append({
                "role": "user",
                "content": f"[KONTEKS: Berikut CV saya untuk referensi]\n\n{cv_text[:4000]}",
            })
            messages.append({
                "role": "assistant",
                "content": "Terima kasih sudah berbagi CV kamu! Saya sudah membacanya. Silakan ceritakan tentang cita-cita atau tujuan karir yang ingin kamu capai, dan saya akan bantu memberikan saran berdasarkan profil kamu saat ini. 😊",
            })

        # Add conversation history
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current message
        messages.append({"role": "user", "content": user_message})

        # Call OpenAI with tools
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=1500,
        )

        response_message = response.choices[0].message
        
        # Check if the model wants to call a tool
        if response_message.tool_calls:
            messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                
                if function_name == "query_job_database_statistics":
                    import json
                    function_args = json.loads(tool_call.function.arguments)
                    question_arg = function_args.get("question")
                    
                    # Execute tool
                    from agents.sql_agent import generate_sql_query
                    from database import DatabaseManager
                    
                    sql_query = generate_sql_query(question_arg)
                    tool_result = ""
                    if not sql_query or sql_query.startswith("-- Error"):
                        tool_result = f"Gagal menerjemahkan pertanyaan menjadi query SQL: {sql_query}"
                    else:
                        try:
                            db = DatabaseManager()
                            db_results = db.execute_raw_sql(sql_query)
                            if not db_results:
                                tool_result = "Tidak ditemukan data yang sesuai di database."
                            elif isinstance(db_results, list) and len(db_results) > 0 and "error" in db_results[0]:
                                tool_result = f"Error database: {db_results[0]['error']}"
                            else:
                                tool_result = json.dumps(db_results[:15], indent=2)
                        except Exception as ex:
                            tool_result = f"Error mengakses database: {str(ex)}"
                    
                    # Append tool response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result
                    })
            
            # Second call to get the final response explaining database results
            second_response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
            )
            return {
                "response": second_response.choices[0].message.content,
                "available": True,
            }
            
        return {
            "response": response_message.content,
            "available": True,
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "available": True,
        }
