import streamlit as st
from transformers import pipeline
import torch
import pandas as pd

st.set_page_config(page_title="Zero-shot Topic Classifier", layout="centered")

st.title("Clasificador de Tópicos (Zero-Shot)")
st.markdown(
    "Ingresa un texto y una lista de etiquetas separadas por comas. "
    "El modelo evaluará la afinidad de cada etiqueta (sin reentrenamiento)."
)

# Compatibilidad con diferentes versiones de streamlit: cache_resource si está disponible
if hasattr(st, "cache_resource"):
    cache_model = st.cache_resource
else:
    # st.cache con allow_output_mutation para versiones antiguas
    def cache_model(func):
        return st.cache(allow_output_mutation=True)(func)

@cache_model
def load_pipeline(model_name: str, device: int):
    """Carga y devuelve la pipeline de zero-shot. Está cacheada para no recargarla cada interacción."""
    return pipeline("zero-shot-classification", model=model_name, device=device)

# Selección de modelo (por defecto facebook/bart-large-mnli)
model_choice = st.selectbox(
    "Modelo (elige según idioma / recursos)",
    (
        "facebook/bart-large-mnli",                      # muy usado en Zero-shot (inglés)
        "joeddav/xlm-roberta-large-xnli"                 # multilingüe (mejor para español)
    ),
    index=0
)

# Detectar si hay GPU disponible para acelerar (si no, use CPU)
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    st.write("GPU detectada — usando CUDA para inferencia.")
else:
    st.write("No se detectó GPU — usando CPU (más lento).")

# Carga cacheada de la pipeline
with st.spinner("Cargando modelo (esto puede tardar la primera vez)..."):
    classifier = load_pipeline(model_choice, device)

# Inputs del usuario
text = st.text_area("Texto a clasificar", height=220, value="Escribe aquí el texto que quieres analizar...")
labels_input = st.text_input("Etiquetas (separadas por comas)", value="deportes, política, tecnología, salud")

hypothesis_template = st.text_input(
    "Plantilla de hipótesis (opcional)",
    value="Este texto trata sobre {}"
)
# Conversión segura de etiquetas a lista
labels = [lbl.strip() for lbl in labels_input.split(",") if lbl.strip()]

col1, col2 = st.columns([1, 1])
with col1:
    top_k = st.number_input("Mostrar top k etiquetas", min_value=1, max_value=max(1, len(labels)), value=min(5, len(labels)), step=1)
with col2:
    # Umbral visual opcional
    show_threshold = st.checkbox("Mostrar umbral mínimo (ocultar etiquetas con score < umbral)", value=False)
    threshold = 0.2
    if show_threshold:
        threshold = st.slider("Umbral mínimo", min_value=0.0, max_value=1.0, value=0.2)

if st.button("Clasificar"):
    if not text.strip():
        st.error("Por favor ingresa un texto.")
    elif not labels:
        st.error("Por favor ingresa al menos una etiqueta.")
    else:
        with st.spinner("Clasificando..."):
            # Llamada a la pipeline
            result = classifier(text, candidate_labels=labels, hypothesis_template=hypothesis_template)

        # result: {'sequence':..., 'labels':[...], 'scores':[...]}
        df = pd.DataFrame({
            "label": result["labels"],
            "score": result["scores"]
        })

        # Aplicar umbral si se pidió
        if show_threshold:
            df = df[df["score"] >= threshold]

        # Mostrar top_k
        df_top = df.head(top_k).reset_index(drop=True)

        st.subheader("Resultados (ordenados por afinidad)")
        st.write(df_top)

        # Barra (ordenar para visualización)
        df_plot = df_top.sort_values("score", ascending=True).set_index("label")
        st.bar_chart(df_plot)

        # Detalles
        st.markdown("**Detalles crudos del modelo:**")
        st.json(result)
