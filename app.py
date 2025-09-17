# app.py
import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict

st.set_page_config(page_title="Chat con Memoria (Groq)", layout="wide")
st.title("Chatbot con Memoria — Groq API")
st.markdown("Usa `groq_api_key` en `st.secrets` y el endpoint `/openai/v1/responses` por defecto.")

# -------------------------
# Config
# -------------------------
SYSTEM_PROMPT = "Eres un asistente conversacional útil y conciso. Responde en español."
MODEL_NAME = "llama3-8b-8192"
DEFAULT_MAX_CHARS_HISTORY = 6000
DEFAULT_GROQ_ENDPOINT = "https://api.groq.com/openai/v1/responses"  # endpoint correcto según docs. :contentReference[oaicite:1]{index=1}

# -------------------------
# session_state
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT, "time": datetime.utcnow().isoformat()}]

def add_message(role: str, content: str):
    st.session_state.history.append({"role": role, "content": content, "time": datetime.utcnow().isoformat()})

def clear_history():
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT, "time": datetime.utcnow().isoformat()}]

def history_to_messages_for_api(history: List[Dict]) -> List[Dict]:
    # Groq acepta un campo "input" o "messages" según el endpoint; aquí enviamos "messages" estilo chat.
    return [{"role": m["role"], "content": m["content"]} for m in history]

def truncate_history_by_chars(history: List[Dict], max_chars: int) -> List[Dict]:
    if max_chars <= 0:
        return history
    total = sum(len(m["content"]) for m in history)
    if total <= max_chars:
        return history
    new_hist = [h for h in history if h["role"] == "system"]
    reversed_msgs = [m for m in history if m["role"] != "system"][::-1]
    current_chars = sum(len(m["content"]) for m in new_hist)
    for m in reversed_msgs:
        if current_chars + len(m["content"]) <= max_chars:
            new_hist.append(m)
            current_chars += len(m["content"])
        else:
            remaining = max_chars - current_chars
            if remaining > 50:
                truncated = m.copy()
                truncated["content"] = m["content"][-remaining:]
                new_hist.append(truncated)
                current_chars += len(truncated["content"])
            break
    return new_hist[::-1]

# -------------------------
# Llamada a Groq con parsing específico
# -------------------------
def parse_groq_response(data: dict) -> str:
    """
    Extrae el texto de la respuesta según la estructura que usa Groq:
    data['output'][i]['content'][j]['text'] (ejemplo en la doc).
    Devuelve el primer texto encontrado o un fallback corto.
    """
    if not isinstance(data, dict):
        return str(data)

    # 1) buscar output -> content -> text
    outputs = data.get("output")
    if isinstance(outputs, list):
        for out in outputs:
            if isinstance(out, dict):
                content_list = out.get("content") or []
                if isinstance(content_list, list):
                    for c in content_list:
                        if isinstance(c, dict):
                            # caso típico: {'type':'output_text', 'text': "..."}
                            if "text" in c and isinstance(c["text"], str):
                                return c["text"]
                            # a veces el texto puede estar en 'content' o 'message' campos (robusto)
                            if "content" in c and isinstance(c["content"], str):
                                return c["content"]
    # 2) estructura estilo OpenAI (choices -> message/content o text)
    choices = data.get("choices")
    if isinstance(choices, list) and len(choices) > 0:
        ch0 = choices[0]
        if isinstance(ch0, dict):
            msg = ch0.get("message")
            if isinstance(msg, dict) and "content" in msg:
                # msg["content"] puede ser dict o str
                if isinstance(msg["content"], str):
                    return msg["content"]
                if isinstance(msg["content"], dict) and "text" in msg["content"]:
                    return msg["content"]["text"]
            if "text" in ch0:
                return ch0["text"]
    # 3) fallback: algún campo 'text' directo
    if "text" in data and isinstance(data["text"], str):
        return data["text"]

    # último recurso: devolver un JSON truncado para debugging
    try:
        return json.dumps(data)[:3000]
    except Exception:
        return str(data)[:1000]

def call_groq_api(messages: List[Dict], api_key: str, endpoint: str, timeout: int = 30) -> str:
    """
    Hace POST a Groq. Devuelve siempre string (respuesta o mensaje de error legible).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Groq admite un body con "model" y "input" o con "messages". Aquí usamos "model" + "messages".
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.2,
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        return f"[ERROR de conexión] {e}"

    if not resp.ok:
        body_text = resp.text[:1500] if resp.text else "<no body>"
        return f"[HTTP {resp.status_code}] Error llamando al endpoint.\nURL: {endpoint}\nRespuesta (parcial): {body_text}"

    try:
        data = resp.json()
    except Exception:
        return resp.text[:4000]

    return parse_groq_response(data)

# -------------------------
# Sidebar: secrets y endpoint
# -------------------------
with st.sidebar:
    st.header("Configuración")
    max_chars = st.number_input("Máx. caracteres de historial (truncado)", value=DEFAULT_MAX_CHARS_HISTORY, step=1000)
    show_raw = st.checkbox("Mostrar historial crudo", value=False)
    st.markdown("Guarda `groq_api_key` y opcionalmente `groq_endpoint` en `.streamlit/secrets.toml` o en Streamlit Cloud Secrets.")

    api_key = st.secrets.get("groq_api_key", None)
    groq_endpoint_default = st.secrets.get("groq_endpoint", "") or DEFAULT_GROQ_ENDPOINT
    endpoint_override = st.text_input("Endpoint (override)", value=groq_endpoint_default)

    if api_key:
        st.success("Clave encontrada en st.secrets (no se muestra).")
    else:
        st.warning("No se detectó `groq_api_key` en st.secrets — se usará simulador si no hay clave.")

    # Botón para probar endpoint (rápido)
    if st.button("Probar endpoint"):
        test_payload = {"model": MODEL_NAME, "input": "Prueba rápida", "max_tokens": 10}
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = requests.post(endpoint_override, headers=headers, json=test_payload, timeout=10)
            st.markdown(f"**Resultado prueba:** HTTP {resp.status_code}")
            st.code(resp.text[:2000])
        except Exception as e:
            st.error(f"Error en la prueba: {e}")

# -------------------------
# UI principal: chat
# -------------------------
chat_col, aux_col = st.columns([3, 1])

with chat_col:
    st.subheader("Conversación")
    for msg in st.session_state.history:
        if msg["role"] == "system":
            continue
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    user_input = st.chat_input("Escribe tu mensaje...")
    if user_input:
        add_message("user", user_input)
        hist_for_api = truncate_history_by_chars(st.session_state.history, max_chars)
        messages_payload = history_to_messages_for_api(hist_for_api)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Generando respuesta..._")

        if not api_key:
            assistant_text = f"(SIMULADOR) He recibido: {user_input[:200]}. Responde como asistente en español."
            placeholder.markdown(assistant_text)
            add_message("assistant", assistant_text)
        else:
            assistant_text = call_groq_api(messages_payload, api_key=api_key, endpoint=endpoint_override)
            if assistant_text.startswith("[HTTP") or assistant_text.startswith("[ERROR"):
                placeholder.markdown(f"**Error llamando a la API:**\n\n{assistant_text}")
                add_message("assistant", f"**(Error de la API)**: {assistant_text}")
            else:
                placeholder.markdown(assistant_text)
                add_message("assistant", assistant_text)

with aux_col:
    st.subheader("Acciones")
    if st.button("Limpiar memoria"):
        clear_history()
        st.experimental_rerun()

    if st.button("Exportar historial (JSON)"):
        json_hist = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
        st.download_button("Descargar historial JSON", data=json_hist, file_name="historial_chat.json", mime="application/json")

    st.markdown("---")
    st.write(f"{len(st.session_state.history)} mensajes (incluye system prompt).")
    if show_raw:
        st.json(st.session_state.history)
