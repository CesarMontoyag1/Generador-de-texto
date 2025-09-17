# app.py
import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict

st.set_page_config(page_title="Chat con Memoria (Groq)", layout="wide")
st.title("Chatbot con Memoria — Groq API (payload `input` corregido)")
st.markdown("Envía `input` (texto) al endpoint `/openai/v1/responses` de Groq. Guarda tu clave en `st.secrets['groq_api_key']`.")

# -------------------------
# Config
# -------------------------
SYSTEM_PROMPT = "Eres un asistente conversacional útil y conciso. Responde en español."
MODEL_NAME = "llama3-8b-8192"
DEFAULT_MAX_CHARS_HISTORY = 6000
DEFAULT_GROQ_ENDPOINT = "https://api.groq.com/openai/v1/responses"

# -------------------------
# session_state
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT, "time": datetime.utcnow().isoformat()}]

def add_message(role: str, content: str):
    st.session_state.history.append({"role": role, "content": content, "time": datetime.utcnow().isoformat()})

def clear_history():
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT, "time": datetime.utcnow().isoformat()}]

def truncate_history_by_chars(history: List[Dict], max_chars: int) -> List[Dict]:
    """Trunca mensajes antiguos hasta que el total de caracteres quede por debajo de max_chars."""
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

def build_input_from_history(history: List[Dict]) -> str:
    """
    Construye un único string 'input' a partir del historial.
    Formato:
      [SYSTEM]: ...
      [USER]: ...
      [ASSISTANT]: ...
    Esto simplifica el payload para el endpoint que espera 'input'.
    """
    parts = []
    for m in history:
        role = m["role"].upper()
        # Normalizar role: SYSTEM, USER, ASSISTANT
        if role not in {"SYSTEM", "USER", "ASSISTANT"}:
            role = "USER"
        content = m["content"].strip()
        parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)

# -------------------------
# Parseo de respuesta de Groq
# -------------------------
def parse_groq_response(data: dict) -> str:
    """
    Extrae el texto de la respuesta según estructura típica de Groq:
    data['output'][i]['content'][j]['text'].
    """
    if not isinstance(data, dict):
        return str(data)

    outputs = data.get("output")
    if isinstance(outputs, list):
        for out in outputs:
            if isinstance(out, dict):
                content_list = out.get("content") or []
                if isinstance(content_list, list):
                    for c in content_list:
                        if isinstance(c, dict):
                            if "text" in c and isinstance(c["text"], str):
                                return c["text"]
                            if "content" in c and isinstance(c["content"], str):
                                return c["content"]

    # fallback estilo OpenAI
    choices = data.get("choices")
    if isinstance(choices, list) and len(choices) > 0:
        ch0 = choices[0]
        if isinstance(ch0, dict):
            msg = ch0.get("message")
            if isinstance(msg, dict):
                # msg["content"] puede ser dict o str
                cont = msg.get("content")
                if isinstance(cont, str):
                    return cont
                if isinstance(cont, dict) and "text" in cont:
                    return cont["text"]
            if "text" in ch0 and isinstance(ch0["text"], str):
                return ch0["text"]

    if "text" in data and isinstance(data["text"], str):
        return data["text"]

    try:
        return json.dumps(data)[:3000]
    except Exception:
        return str(data)[:1000]

# -------------------------
# Llamada a Groq usando 'input'
# -------------------------
def call_groq_api_with_input(input_text: str, api_key: str, endpoint: str, timeout: int = 30) -> str:
    """
    Envía payload con 'input' al endpoint de Groq. Devuelve string (respuesta o mensaje de error).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "input": input_text,
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
        st.success("Clave detectada en st.secrets (no se muestra).")
    else:
        st.warning("No se detectó `groq_api_key` en st.secrets — se usará simulador si no hay clave.")

    # Botón para probar endpoint (usa 'input' en lugar de 'messages')
    if st.button("Probar endpoint"):
        test_input = "[SYSTEM]: prueba\n\n[USER]: Hola"
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = requests.post(endpoint_override, headers=headers, json={"model": MODEL_NAME, "input": test_input, "max_tokens": 10}, timeout=10)
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
        input_text = build_input_from_history(hist_for_api)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Generando respuesta..._")

        if not api_key:
            assistant_text = f"(SIMULADOR) He recibido: {user_input[:200]}. Responde como asistente en español."
            placeholder.markdown(assistant_text)
            add_message("assistant", assistant_text)
        else:
            assistant_text = call_groq_api_with_input(input_text, api_key=api_key, endpoint=endpoint_override)
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
