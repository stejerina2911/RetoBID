import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Importar librerías para gráficos
import shap
from streamlit_shap import st_shap

# Configuración de la página
st.set_page_config(page_title="Predicción de Probabilidad de Empleo", layout="centered")

# Cargar el modelo
rf = joblib.load('random_forest_modelSTM.joblib')

st.title("🧑‍💼 Predicción de Probabilidad de Empleo")

st.write("Ingrese sus datos a continuación para calcular la probabilidad de que usted tenga empleo y descubra qué factores influyen más en su situación laboral.")

# Variables binarias (dummies)
jefehogar = st.selectbox("¿Es usted jefe de hogar?", ("No", "Sí"))
hombre = st.selectbox("¿Cuál es su género?", ("Mujer", "Hombre"))
rural = st.selectbox("¿Vive en una zona rural?", ("No", "Sí"))
HLENGUA = st.selectbox("¿Habla una lengua indígena?", ("No", "Sí"))
casado = st.selectbox("¿Está usted casado(a)?", ("No", "Sí"))
Ident_Indigena = st.selectbox("¿Se identifica como indígena?", ("No", "Sí"))

# Variables continuas
ESCOACUM = st.slider("Años de educación acumulada", min_value=0, max_value=30, value=12)
EDAD = st.slider("Edad", min_value=15, max_value=100, value=30)

# Calcular EDAD2
EDAD2 = EDAD ** 2

# Convertir entradas a 0 y 1
def convert_to_binary(value):
    return 1 if value == "Sí" or value == "Hombre" else 0

jefehogar_bin = convert_to_binary(jefehogar)
hombre_bin = convert_to_binary(hombre)
rural_bin = convert_to_binary(rural)
HLENGUA_bin = convert_to_binary(HLENGUA)
casado_bin = convert_to_binary(casado)
Ident_Indigena_bin = convert_to_binary(Ident_Indigena)

# Calcular 'hombrecasado' como una combinación de 'hombre' y 'casado'
hombrecasado_bin = hombre_bin * casado_bin  # Multiplicación lógica (AND)

# Crear el array de características en el orden correcto
features = np.array([[
    jefehogar_bin, hombre_bin, rural_bin, ESCOACUM, EDAD, EDAD2,
    HLENGUA_bin, hombrecasado_bin, casado_bin, Ident_Indigena_bin
]])

feature_names = ['jefehogar', 'hombre', 'rural', 'ESCOACUM', 'EDAD', 'EDAD2',
                 'HLENGUA', 'hombrecasado', 'casado', 'Ident_Indigena']

# Cargar el explainer de SHAP (cacheado para eficiencia)
@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(rf)

# Botón para realizar la predicción
if st.button("Calcular probabilidad de empleo"):
    # Realizar la predicción
    probabilidad = rf.predict_proba(features)[0][1]  # Probabilidad de tener empleo

    st.subheader(f"🔎 Su probabilidad de tener empleo es: **{probabilidad * 100:.2f}%**")

    # Calcular los valores SHAP para la instancia
    shap_values = explainer.shap_values(features)

    # Seleccionar los valores SHAP para la clase positiva, si está disponible
    if isinstance(shap_values, list) and len(shap_values) > 1:
        influencia = shap_values[1]  # Usar los valores SHAP de la clase 1
    else:
        influencia = shap_values[0]  # Usar los valores SHAP de la primera clase

    # Asegurarnos de que 'influencia' es un arreglo unidimensional
    if influencia.ndim > 1:
        influencia = influencia.flatten()

    # Depuración: Imprimir las longitudes de las variables
    st.write("### Depuración: Formas de las Variables")
    st.write(f"Longitud de 'feature_names': {len(feature_names)}")
    st.write(f"Longitud de 'features[0]': {len(features[0])}")
    st.write(f"Longitud de 'influencia': {len(influencia)}")

    # Crear un DataFrame para los valores SHAP solo si las longitudes coinciden
    if len(feature_names) == len(features[0]) == len(influencia):
        shap_df = pd.DataFrame({
            'Característica': feature_names,
            'Valor': features[0],
            'Influencia': influencia
        })

        # Ordenar por valor absoluto de SHAP
        shap_df['Influencia_abs'] = np.abs(shap_df['Influencia'])
        shap_df = shap_df.sort_values(by='Influencia_abs', ascending=False)
        
        # Mostrar tabla de influencias
        st.table(shap_df[['Característica', 'Valor', 'Influencia']].head(10))

        # Mostrar gráfico de SHAP values
        st.write("### Visualización de la influencia de cada factor:")
        shap.initjs()
        force_plot = shap.force_plot(explainer.expected_value[1], influencia, features[0], feature_names=feature_names)
        st_shap(force_plot)
    else:
        st.error("Las longitudes de las columnas no coinciden. Verifica el código.")

    st.info("Puede ajustar las características y volver a calcular para ver cómo cambia la probabilidad.")

