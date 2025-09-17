import streamlit as st
import requests
import time

# ----------------------------
# Función para detectar modelo disponible en Groq
# ----------------------------
def find_working_model(api_key: str, endpoint: str, candidates: list, timeout=8):
    """
    Prueba los modelos en 'candidates' con un payload mínimo.
    Devuelve el primero que funcione (status 200). Si ninguno sirve, devuelve None.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for model in candidates:
        payload = {
            "model": model,
            "input": "[SYSTEM]: prueba\n\n[USER]: ping",
            "max_output_tokens": 1,
            "temperature": 0.0
        }
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException:
            continue

        if resp.ok:
            return model

        # Puedes imprimir mensajes de error para depuración
        # print(f"{model} -> {resp.status_code}: {resp.text}")

        time.sleep(0.2)
    return None

# ----------------------------
# Configuración Streamlit
# ----------------------------
st.title("📝 Generador de texto con Groq")

# Cargar API key desde secrets
api_key = st.secrets.get("groq_api_key", "").strip()
if not api_key:
    st.error("No se encontró `groq_api_key` en tus secrets. Agrégalo en .streamlit/secrets.toml")
    st.stop()

# Endpoint de Groq
endpoint = "https://api.groq.com/openai/v1/responses"

# Buscar modelo funcional
candidates = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192"
]

st.info("Buscando un modelo disponible en Groq...")
MODEL_NAME = find_working_model(api_key, endpoint, candidates)
if not MODEL_NAME:
    st.error("No se encontró ningún modelo disponible. Revisa tu cuenta Groq.")
    st.stop()

st.success(f"Usando el modelo: {MODEL_NAME}")

# ----------------------------
# Entrada del usuario
# ----------------------------
user_prompt = st.text_area("Escribe tu texto:", "")

# ----------------------------
# Llamar a Groq API
# ----------------------------
def call_groq_api(prompt, api_key, endpoint, model):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "input": f"[SYSTEM]: Eres un asistente útil.\n\n[USER]: {prompt}",
        "max_output_tokens": 300,
        "temperature": 0.7
    }

    resp = requests.post(endpoint, headers=headers, json=payload)

    if not resp.ok:
        raise RuntimeError(f"Error {resp.status_code}: {resp.text}")

    data = resp.json()
    # Extraer el texto generado
    return data["output"][0]["content"][0]["text"]

# ----------------------------
# Ejecutar generación
# ----------------------------
if st.button("Generar texto"):
    if not user_prompt.strip():
        st.warning("Escribe algo antes de generar.")
    else:
        try:
            output = call_groq_api(user_prompt, api_key, endpoint, MODEL_NAME)
            st.subheader("✍️ Resultado:")
            st.write(output)
        except Exception as e:
            st.error(f"Error llamando a la API: {e}")
