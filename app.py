import streamlit as st
from groq import Groq

# Cargar API Key de manera segura
api_key = st.secrets["groq_api_key"]
client = Groq(api_key=api_key)

# Inicializar historial con mensaje de sistema
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "Eres un asistente útil y conversacional."}
    ]

st.title("🤖 Chatbot con Memoria (Groq + Streamlit)")

# Mostrar historial previo (excepto system prompt)
for msg in st.session_state["messages"]:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Entrada del usuario
if prompt := st.chat_input("Escribe tu mensaje..."):
    # Añadir mensaje del usuario al historial
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Enviar todo el historial al modelo actualizado
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # <--- modelo soportado
            messages=st.session_state["messages"]
        )
        respuesta = response.choices[0].message.content

        # Añadir respuesta del modelo al historial
        st.session_state["messages"].append({"role": "assistant", "content": respuesta})

        with st.chat_message("assistant"):
            st.markdown(respuesta)

    except Exception as e:
        st.error(f"Error con la API de Groq: {e}")
