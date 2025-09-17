import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict

st.set_page_config(page_title="Chat con Memoria (Groq API)", layout="wide")
st.title("Chatbot con Memoria — Ejercicio 2")
st.markdown("Ejemplo de chat stateful que guarda historial en `st.session_state` y envía la conversación a un LLM (llama3-8b-8192) vía API.")

# --- Configuración inicial ---
SYSTEM_PROMPT = (
    "Eres un asistente conversacional útil y conciso. Responde en español a menos que se pida otro idioma."
)
MODEL_NAME = "llama3-8b-8192"  # solo referencia; la implementación concreta depende del endpoint de Groq
DEFAULT_MAX_CHARS_HISTORY = 6000  # ajuste para truncado simple por caracteres

# --- Helpers para session state ---
if "history" not in st.session_state:
    # history: lista de mensajes con dicts {'role': 'user'|'assistant'|'system', 'content': str, 'time': iso}
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT, "time": datetime.utcnow().isoformat()}]

def add_message(role: str, content: str):
    st.session_state.history.append({"role": role, "content": content, "time": datetime.utcnow().isoformat()})

def clear_history():
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT, "time": datetime.utcnow().isoformat()}]

def history_to_messages_for_api(history: List[Dict]) -> List[Dict]:
    """
    Convierte el historial al formato de mensajes esperado por la API.
    Ajusta según la especificación real de Groq (ej. nombres de campos).
    """
    msgs = []
    for m in history:
        msgs.append({"role": m["role"], "content": m["content"]})
    return msgs

def truncate_history_by_chars(history: List[Dict], max_chars: int) -> List[Dict]:
    """
    Trunca mensajes antiguos hasta que el total de caracteres esté por debajo de max_chars.
    Conserva siempre el primer mensaje de 'system'.
    """
    if max_chars <= 0:
        return history
    # calcular tamaño total
    total = sum(len(m["content"]) for m in history)
    if total <= max_chars:
        return history
    # eliminar mensajes antiguos (excepto el system initial) hasta cumplir
    new_hist = [h for h in history if h["role"] == "system"]  # mantener sistema
    # añadir mensajes desde el final (más recientes) hacia atrás hasta llenar el límite
    reversed_msgs = [m for m in history if m["role"] != "system"][::-1]
    current_chars = sum(len(m["content"]) for m in new_hist)
    for m in reversed_msgs:
        if current_chars + len(m["content"]) <= max_chars:
            new_hist.append(m)
            current_chars += len(m["content"])
        else:
            # si el mensaje es muy grande, puedes truncarlo parcialmente:
            remaining = max_chars - current_chars
            if remaining > 50:  # si queda espacio suficiente, truncar muy largo
                truncated = m.copy()
                truncated["content"] = m["content"][-remaining:]
                new_hist.append(truncated)
                current_chars += len(truncated["content"])
            break
    # devolver en orden cronológico
    return new_hist

# --- Llamada al API de Groq (plantilla) ---
def call_groq_api(messages: List[Dict], api_key: str, endpoint: str = None, timeout: int = 30) -> str:
    """
    Envía los mensajes a la API de Groq y devuelve la respuesta del asistente.
    *Nota*: Ajusta la URL, headers y el body al formato oficial de Groq. Aquí se muestra
    un cuerpo genérico similar a otros proveedores (modifica según documentación real).
    """
    if endpoint is None:
        # endpoint placeholder: reemplaza por el endpoint real de Groq
        endpoint = "https://api.groq.com/v1/generate"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        # parametros comunes (ajusta según Groq): max_tokens, temperature, etc.
        "max_tokens": 512,
        "temperature": 0.2,
        # "other_params": ...
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # dependiendo de la respuesta de Groq, extrae el texto:
        # Aquí intentamos buscar de forma robusta varios posibles caminos:
        if "output" in data:
            # ejemplo: {"output": [{"content": "texto"}]}
            out = data["output"]
            if isinstance(out, list) and len(out) > 0:
                # buscar contenido en la primera salida
                candidate = out[0]
                if isinstance(candidate, dict) and "content" in candidate:
                    return candidate["content"]
                elif isinstance(candidate, str):
                    return candidate
        # fallback: si hay 'choices' parecido a OpenAI:
        if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
            ch = data["choices"][0]
            if isinstance(ch, dict) and "message" in ch and "content" in ch["message"]:
                return ch["message"]["content"]
            if "text" in ch:
                return ch["text"]
        # si nada, devolver la serialización completa para ayuda debugging
        return json.dumps(data)[:4000]
    except requests.RequestException as e:
        # manejar errores de red / 4xx/5xx
        raise RuntimeError(f"Error al llamar a la API de Groq: {e}")

# --- UI lateral: configuraciones y secretos ---
with st.sidebar:
    st.header("Configuración")
    max_chars = st.number_input("Máx. caracteres de historial (truncado)", value=DEFAULT_MAX_CHARS_HISTORY, step=1000)
    show_raw = st.checkbox("Mostrar historial crudo", value=False)
    st.markdown("**Secrets:** usa `.streamlit/secrets.toml` con `groq_api_key` y opcionalmente `groq_endpoint`.")
    api_key = st.secrets.get("groq_api_key", "")  # si no existe, cadena vacía
    groq_endpoint = st.secrets.get("groq_endpoint", None)
    if api_key:
        st.success("Clave de Groq detectada en st.secrets.")
    else:
        st.warning("No se detectó clave en st.secrets — la app usará un simulador local.")

# --- Área principal: Chat UI ---
chat_col, aux_col = st.columns([3, 1])

with chat_col:
    st.subheader("Conversación")
    # mostrar mensajes en orden
    for msg in st.session_state.history:
        role = msg["role"]
        content = msg["content"]
        timestamp = msg.get("time", "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)
        elif role == "system":
            # no mostrar el system prompt al usuario (opcional)
            pass

    # Input del usuario con st.chat_input
    user_input = st.chat_input("Escribe tu mensaje...")
    if user_input:
        # añadir mensaje del usuario al historial
        add_message("user", user_input)

        # preparar historial a enviar (truncado)
        hist_for_api = truncate_history_by_chars(st.session_state.history, max_chars)
        messages_payload = history_to_messages_for_api(hist_for_api)

        # mostrar mensaje de espera en UI
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Generando respuesta..._")

        # llamar a la API si existe la clave, si no, usar simulador simple
        try:
            if api_key:
                assistant_text = call_groq_api(messages_payload, api_key=api_key, endpoint=groq_endpoint)
            else:
                # Simulador: eco simple + instrucción
                assistant_text = f"(SIMULADOR) He recibido tu mensaje: {user_input[:200]}. Responde como asistente."
            # reemplazar el placeholder con la respuesta real
            placeholder.markdown(assistant_text)
            add_message("assistant", assistant_text)
        except Exception as e:
            placeholder.markdown(f"**Error generando respuesta:** {e}")
            st.exception(e)

with aux_col:
    st.subheader("Acciones")
    if st.button("Limpiar memoria"):
        clear_history()
        st.experimental_rerun()

    if st.button("Descargar historial (JSON)"):
        json_hist = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
        st.download_button("Descargar JSON", data=json_hist, file_name="historial_chat.json", mime="application/json")

    st.markdown("---")
    st.markdown("Historial actual:")
    st.write(f"{len(st.session_state.history)} mensajes (incluye system prompt).")
    if show_raw:
        st.json(st.session_state.history)
