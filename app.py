import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================================================================
# CONFIGURACION DE PAGINA
# ======================================================================
st.set_page_config(
    page_title="Detector de Fake News",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ======================================================================
# CSS — solo para temas globales, NO para contenido dinamico
# ======================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, p, div, span, label, textarea, button {
    font-family: 'Inter', sans-serif !important;
}
.stApp > header { display: none !important; }
#MainMenu, footer { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1100px !important; }

/* Containers nativos dentro de columnas */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================================
# RUTAS DE MODELOS
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "AvancesTesisCronograma", "data")
LR_PATH  = os.path.join(DATA_DIR, "modelo_optimizado_lr.joblib")
SVM_PATH = os.path.join(DATA_DIR, "modelo_optimizado_svm.joblib")
BETO_DIR = os.path.join(DATA_DIR, "fold_1")

# ======================================================================
# CARGA DE MODELOS
# ======================================================================
@st.cache_resource(show_spinner=False)
def cargar_modelos_lineales():
    return joblib.load(LR_PATH), joblib.load(SVM_PATH)

@st.cache_resource(show_spinner=False)
def cargar_beto():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BETO_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BETO_DIR)
    model.to(device).eval()
    return tokenizer, model, device

# ======================================================================
# FUNCIONES DE PREDICCION
# ======================================================================
def predecir_lineal(pipeline, texto):
    import pandas as pd
    X = pd.DataFrame({"texto_lineal": [texto]})
    pred = int(pipeline.predict(X)[0])
    try:
        proba = pipeline.predict_proba(X)[0]
        confianza = float(proba[pred])
    except Exception:
        confianza = None
    return pred, confianza

def predecir_beto(tokenizer, model, device, texto):
    inputs = tokenizer(texto, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    proba = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred = int(np.argmax(proba))
    confianza = float(proba[pred])
    return pred, confianza

# ======================================================================
# TARJETAS usando SOLO componentes nativos de Streamlit
# ======================================================================
def mostrar_tarjeta_analitica(col, nombre, pred, confianza, error=None):
    """Usa st.container + componentes nativos. Sin HTML custom."""
    with col:
        with st.container(border=True):
            st.caption(nombre)
            if error:
                st.error("Error al cargar el modelo")
            elif pred is None:
                st.markdown(
                    "<span style='color:#94a3b8;font-size:0.9rem'>"
                    "Ingresa una noticia para analizar</span>",
                    unsafe_allow_html=True,
                )
            else:
                if pred == 1:
                    st.success("**VERDADERA**")
                else:
                    st.error("**FALSA**")

                if confianza is not None:
                    pct = int(confianza * 100)
                    st.metric(label="Confianza", value=f"{pct}%")
                    st.progress(pct)
                else:
                    st.info("Confianza no disponible para este modelo")


def mostrar_tarjeta_rapida(col, nombre, pred, error=None):
    """Tarjeta minimalista. Solo veredicto."""
    with col:
        with st.container(border=True):
            st.caption(nombre)
            if error:
                st.error("Error")
            elif pred is None:
                st.markdown(
                    "<span style='color:#94a3b8;font-size:0.9rem'>Esperando...</span>",
                    unsafe_allow_html=True,
                )
            elif pred == 1:
                st.success("## VERDADERA")
            else:
                st.error("## FALSA")


# ======================================================================
# ENCABEZADO
# ======================================================================
st.markdown(
    "<h1 style='text-align:center;font-size:2rem;font-weight:900;color:#1e293b;"
    "margin-bottom:0;letter-spacing:-0.03em'>"
    "Panel de <span style='color:#4f46e5'>Deteccion de Fake News</span></h1>"
    "<p style='text-align:center;color:#64748b;font-size:0.88rem;margin-top:6px'>"
    "Simulacion de despliegue &mdash; Proyecto Final ML</p>",
    unsafe_allow_html=True,
)

# ======================================================================
# CARGA DE MODELOS
# ======================================================================
with st.spinner("Cargando modelos... esto solo ocurre una vez."):
    lr_error = svm_error = beto_error = None
    lr_model = svm_model = beto_tok = beto_model = None
    try:
        lr_model, svm_model = cargar_modelos_lineales()
    except Exception as e:
        lr_error = svm_error = str(e)
    try:
        beto_tok, beto_model, beto_device = cargar_beto()
    except Exception as e:
        beto_error = str(e)

# ======================================================================
# TABS
# ======================================================================
tab1, tab2 = st.tabs(["  Analitico  ", "  Rapido  "])

# --------------------------------------------------------------- TAB 1
with tab1:
    noticia1 = st.text_area(
        "n1", placeholder="Pega aqui el texto de la noticia que deseas verificar...",
        height=130, label_visibility="collapsed", key="t1",
    )
    st.button("Verificar Noticia", key="b1")

    st.divider()

    c1, c2, c3 = st.columns(3, gap="medium")

    p_lr = c_lr = p_svm = c_svm = p_beto = c_beto = None
    e1 = lr_error; e2 = svm_error; e3 = beto_error

    if st.session_state.get("b1"):
        txt = noticia1.strip()
        if not txt:
            st.warning("Ingresa el texto de una noticia antes de verificar.")
        else:
            with st.spinner("Analizando con los tres modelos..."):
                if lr_model:
                    try: p_lr, c_lr = predecir_lineal(lr_model, txt)
                    except Exception as x: e1 = str(x)
                if svm_model:
                    try: p_svm, c_svm = predecir_lineal(svm_model, txt)
                    except Exception as x: e2 = str(x)
                if beto_model:
                    try: p_beto, c_beto = predecir_beto(beto_tok, beto_model, beto_device, txt)
                    except Exception as x: e3 = str(x)

    mostrar_tarjeta_analitica(c1, "REGRESION LOGISTICA", p_lr, c_lr, e1)
    mostrar_tarjeta_analitica(c2, "SVM", p_svm, c_svm, e2)
    mostrar_tarjeta_analitica(c3, "BETO (TRANSFORMER)", p_beto, c_beto, e3)

# --------------------------------------------------------------- TAB 2
with tab2:
    noticia2 = st.text_area(
        "n2", placeholder="Pega aqui el texto de la noticia que deseas verificar...",
        height=130, label_visibility="collapsed", key="t2",
    )
    st.button("Verificar Noticia", key="b2")

    st.divider()

    q1, q2, q3 = st.columns(3, gap="medium")

    p2_lr = p2_svm = p2_beto = None
    e2_lr = lr_error; e2_svm = svm_error; e2_beto = beto_error

    if st.session_state.get("b2"):
        txt2 = noticia2.strip()
        if not txt2:
            st.warning("Ingresa el texto de una noticia antes de verificar.")
        else:
            with st.spinner("Verificando..."):
                if lr_model:
                    try: p2_lr, _ = predecir_lineal(lr_model, txt2)
                    except Exception as x: e2_lr = str(x)
                if svm_model:
                    try: p2_svm, _ = predecir_lineal(svm_model, txt2)
                    except Exception as x: e2_svm = str(x)
                if beto_model:
                    try: p2_beto, _ = predecir_beto(beto_tok, beto_model, beto_device, txt2)
                    except Exception as x: e2_beto = str(x)

    mostrar_tarjeta_rapida(q1, "REGRESION LOGISTICA", p2_lr, e2_lr)
    mostrar_tarjeta_rapida(q2, "SVM", p2_svm, e2_svm)
    mostrar_tarjeta_rapida(q3, "BETO (TRANSFORMER)", p2_beto, e2_beto)
