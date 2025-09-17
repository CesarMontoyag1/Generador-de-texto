# app.py
import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict

st.set_page_config(page_title="Chat con Memoria (Groq)", layout="wide")
st.title("Chatbot con Memoria — Groq API")
st.markdown(
    "Ejemplo de chat stateful que guarda historial en `st.session_state` y llama al modelo "
    "`llama3-8b-8192` vía la API de Groq. La clave debe almacenarse en `.streamlit/secrets.toml` "
    "o en Secrets de Streamlit Cloud como `groq_api_key`."
)

# -------------------------
# Configuración por defecto
# -------------------------
SYSTEM_PROMPT = (
    "Eres un asistente conversacional útil y conciso. Responde en español a menos que se pida otro idioma."
)
MODEL_NAME = "llama3-8b-8192"
DEFAULT_MAX_CHARS_HISTORY = 6000

# -------------------------
# Inicializar session state
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT, "time": datetime.utcnow().isoformat()}]

def add_message(role: str, content: str):
    st.session_state.history.append({"role": role, "content": content, "time": datetime.utcnow().isoformat()})

def clear_history():
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT, "time": datetime.utcnow().isoformat()}]

def history_to_messages_for_api(history: List[Dict]) -> List[Dict]:
    """Convierte el historial al formato esperado por la API (ajusta si la doc oficial lo requiere)."""
    return [{"role": m["role"], "content": m["content"]} for m in history]

def truncate_history_by_chars(history: List[Dict], max_chars: int) -> List[Dict]:
    """
    Trunca mensajes antiguos hasta que el total de caracteres esté por debajo de max_chars.
    Mantiene siempre el primer 'system' prompt.
    """
    if max_chars <= 0:
        return history
    total = sum(len(m["content"]) for m in history)
    if total <= max_chars:
        return history
    # Mantener system prompt
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
    return new_hist[::-1]  # devolver en orden cronológico

# -------------------------
# Función de llamada a Groq
# -------------------------
def call_groq_api(messages: List[Dict], api_key: str, endpoint: str = None, timeout: int = 30) -> str:
    """
    Envía los mensajes a la API de Groq y devuelve la respuesta de tipo string.
    Ajusta endpoint y parsing según la respuesta real de Groq.
    """
    if endpoint is None:
        endpoint = "https://api.groq.com/v1/generate"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.2,
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Intentar parsear varias estructuras posibles de respuesta
    # 1) {"output":[{"content": "..."}]}
    if isinstance(data, dict):
        out = data.get("output")
        if isinstance(out, list) and len(out) > 0:
            first = out[0]
            if isinstance(first, dict) and "content" in first:
                return first["content"]
            elif isinstance(first, str):
                return first
        # 2) estructura tipo OpenAI: choices -> [ { "message": {"content": ...} } ]
        choices = data.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            ch0 = choices[0]
            if isinstance(ch0, dict):
                msg = ch0.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]
                if "text" in ch0:
                    return ch0["text"]
        # 3) fallback: si existe un campo 'text'
        if "text" in data and isinstance(data["text"], str):
            return data["text"]

    # Si no encontramos el texto esperado, devolvemos un extracto para debugging
    # (no exceder longitud para evitar mostrar secretos)
    try:
        return json.dumps(data)[:4000]
    except Exception:
        return str(data)

# -------------------------
# Sidebar: configuración y secrets
# -------------------------
with st.sidebar:
    st.header("Configuración")
    max_chars = st.number_input("Máx. caracteres de historial (truncado)", value=DEFAULT_MAX_CHARS_HISTORY, step=1000)
    show_raw = st.checkbox("Mostrar historial crudo", value=False)
    st.markdown("**Secretos:** agrega `groq_api_key` y opcionalmente `groq_endpoint` en `.streamlit/secrets.toml` o en Streamlit Cloud Secrets.")

    # Obtener secret de forma segura (NO imprimirla)
    api_key = st.secrets.get("groq_api_key", None)
    groq_endpoint = st.secrets.get("groq_endpoint", None)

    if api_key:
        st.success("Clave detectada (no se muestra por seguridad).")
    else:
        st.warning("No se encontró `groq_api_key` en st.secrets. La app usará un simulador local.")

# -------------------------
# UI principal: chat
# -------------------------
chat_col, aux_col = st.columns([3, 1])

with chat_col:
    st.subheader("Conversación")
    # mostrar mensajes (evitar mostrar system prompt)
    for msg in st.session_state.history:
        role = msg["role"]
        if role == "system":
            continue
        content = msg["content"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)

    # Campo de entrada
    user_input = st.chat_input("Escribe tu mensaje...")
    if user_input:
        add_message("user", user_input)

        # Preparamos historial truncado y payload
        hist_for_api = truncate_history_by_chars(st.session_state.history, max_chars)
        messages_payload = history_to_messages_for_api(hist_for_api)

        # Mostrar placeholder mientras se genera la respuesta
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Generando respuesta..._")

        try:
            if api_key:
                assistant_text = call_groq_api(messages_payload, api_key=api_key, endpoint=groq_endpoint)
            else:
                # Fallback simulador para probar la UI sin clave
                assistant_text = f"(SIMULADOR) He recibido: {user_input[:200]}. Responde como asistente en español."
            # Reemplazar placeholder por la respuesta real
            placeholder.markdown(assistant_text)
            add_message("assistant", assistant_text)
        except Exception as e:
            placeholder.markdown(f"**Error generando respuesta:** {str(e)}")
            st.exception(e)

with aux_col:
    st.subheader("Acciones")
    if st.button("Limpiar memoria"):
        clear_history()
        st.experimental_rerun()

    if st.button("Exportar historial (JSON)"):
        json_hist = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
        st.download_button("Descargar historial JSON", data=json_hist, file_name="historial_chat.json", mime="application/json")

    st.markdown("---")
    st.markdown("Historial actual:")
    st.write(f"{len(st.session_state.history)} mensajes (incluye system prompt).")
    if show_raw:
        st.json(st.session_state.history)

# -------------------------
# Fin de app
# -------------------------
