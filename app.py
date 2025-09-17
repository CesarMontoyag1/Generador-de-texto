# app.py
import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict

st.set_page_config(page_title="Chat con Memoria (Groq) - Debug Endpoint", layout="wide")
st.title("Chatbot con Memoria — Groq API (debug endpoint)")

SYSTEM_PROMPT = (
    "Eres un asistente conversacional útil y conciso. Responde en español a menos que se pida otro idioma."
)
MODEL_NAME = "llama3-8b-8192"
DEFAULT_MAX_CHARS_HISTORY = 6000

# -------------------------
# session_state inicial
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT, "time": datetime.utcnow().isoformat()}]

def add_message(role: str, content: str):
    st.session_state.history.append({"role": role, "content": content, "time": datetime.utcnow().isoformat()})

def clear_history():
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT, "time": datetime.utcnow().isoformat()}]

def history_to_messages_for_api(history: List[Dict]) -> List[Dict]:
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
# call_groq_api mejorado (no lanza excepción a la UI)
# -------------------------
def call_groq_api(messages: List[Dict], api_key: str, endpoint: str, timeout: int = 30) -> str:
    """
    Llama a la API de Groq y SIEMPRE devuelve un string (respuesta o mensaje de error legible).
    No imprime la API key en ningún caso.
    """
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

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        # errores de conexión, DNS, timeout, etc.
        return f"[ERROR de conexión] {e}"

    # Si status != 2xx, capturamos info y devolvemos mensaje legible (sin imprimir el token)
    if not resp.ok:
        # Intentamos obtener cuerpo de respuesta (corto)
        body_text = ""
        try:
            # si viene JSON, lo mostramos de forma legible (limitado a 1500 chars)
            body = resp.text
            body_text = body[:1500]
        except Exception:
            body_text = "<no se pudo leer body>"
        return f"[HTTP {resp.status_code}] Error llamando al endpoint.\nURL: {endpoint}\nRespuesta (parcial): {body_text}"

    # Si es 200 OK, parseamos
    try:
        data = resp.json()
    except Exception:
        # si no es JSON, devolvemos raw text
        return resp.text[:4000]

    # Intentos de extracción robusta (según ejemplos)
    if isinstance(data, dict):
        out = data.get("output")
        if isinstance(out, list) and len(out) > 0:
            first = out[0]
            if isinstance(first, dict) and "content" in first:
                return first["content"]
            if isinstance(first, str):
                return first
        choices = data.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            ch0 = choices[0]
            if isinstance(ch0, dict):
                msg = ch0.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]
                if "text" in ch0:
                    return ch0["text"]
        if "text" in data and isinstance(data["text"], str):
            return data["text"]

    # fallback: devuelve pequeña porción del JSON para debugging
    return json.dumps(data)[:4000]

# -------------------------
# Sidebar: secrets, endpoint override y prueba
# -------------------------
with st.sidebar:
    st.header("Configuración / Debug")
    max_chars = st.number_input("Máx. caracteres de historial (truncado)", value=DEFAULT_MAX_CHARS_HISTORY, step=1000)
    show_raw = st.checkbox("Mostrar historial crudo", value=False)
    st.markdown("Agrega `groq_api_key` y opcionalmente `groq_endpoint` en `.streamlit/secrets.toml` o en Streamlit Cloud Secrets.")

    api_key = st.secrets.get("groq_api_key", None)
    groq_endpoint_default = st.secrets.get("groq_endpoint", "")
    # Campo para probar/override del endpoint en tiempo real
    endpoint_override = st.text_input("Endpoint (override)", value=groq_endpoint_default or "https://api.groq.com/v1/generate")
    st.markdown("Si la URL por defecto falla, prueba con la URL correcta que te indique la doc de Groq.")

    st.write("")  # espacio
    if api_key:
        st.success("Clave encontrada en st.secrets (no se muestra).")
    else:
        st.warning("No se detectó `groq_api_key` en st.secrets — se usará simulador si no hay clave.")

    # Botón para probar endpoint con un payload mínimo (útil para depurar 404)
    if st.button("Probar endpoint"):
        test_payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "Prueba ping"}], "max_tokens": 10}
        try:
            resp = requests.post(endpoint_override, headers={"Authorization": f"Bearer {api_key}" } if api_key else {}, json=test_payload, timeout=10)
            st.markdown(f"**Resultado prueba:** HTTP {resp.status_code}")
            # mostramos parte del body para entender error (limitado)
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

    user_input = st.chat_input("Escribe tu mensaje...")
    if user_input:
        add_message("user", user_input)

        hist_for_api = truncate_history_by_chars(st.session_state.history, max_chars)
        messages_payload = history_to_messages_for_api(hist_for_api)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Generando respuesta..._")

        # Si no hay api_key, usamos simulador
        if not api_key:
            assistant_text = f"(SIMULADOR) He recibido: {user_input[:200]}. Responde como asistente en español."
            placeholder.markdown(assistant_text)
            add_message("assistant", assistant_text)
        else:
            # Llamada segura que devuelve string con respuesta o mensaje de error
            assistant_text = call_groq_api(messages_payload, api_key=api_key, endpoint=endpoint_override)
            # Si la respuesta comienza por "[HTTP" u "[ERROR", es un mensaje de fallo
            if assistant_text.startswith("[HTTP") or assistant_text.startswith("[ERROR"):
                # mostramos error al usuario y en la sección de acciones
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
    st.markdown("Historial actual:")
    st.write(f"{len(st.session_state.history)} mensajes (incluye system prompt).")
    if show_raw:
        st.json(st.session_state.history)
