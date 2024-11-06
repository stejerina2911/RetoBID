import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Importar librerías para gráficos
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Predicción de Probabilidad de Empleo", layout="centered")

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load('random_forest_modelSTM2.joblib')

rf = load_model()

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

# Diccionario de nombres amigables
feature_name_mapping = {
    'jefehogar': 'Jefe de hogar',
    'hombre': 'Género masculino',
    'rural': 'Vive en zona rural',
    'ESCOACUM': 'Años de educación acumulada',
    'EDAD': 'Edad',
    'EDAD2': 'Edad al cuadrado',
    'HLENGUA': 'Habla lengua indígena',
    'hombrecasado': 'Hombre casado',
    'casado': 'Está casado(a)',
    'Ident_Indigena': 'Se identifica como indígena'
}

# Cargar el explainer de SHAP
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

    # Acceder a los valores SHAP para la clase positiva (clase 1)
    influencia = shap_values[1][0]

    # Crear un DataFrame para los valores SHAP con nombres amigables
    shap_df = pd.DataFrame({
        'Característica': [feature_name_mapping.get(name, name) for name in feature_names],
        'Valor': features[0],
        'Influencia': influencia
    })

    # Calcular el valor absoluto de las influencias
    shap_df['Influencia_abs'] = np.abs(shap_df['Influencia'])

    # Calcular el porcentaje de influencia
    total_influencia_abs = shap_df['Influencia_abs'].sum()
    shap_df['Influencia_%'] = (shap_df['Influencia_abs'] / total_influencia_abs) * 100
    shap_df['Influencia_%'] = shap_df['Influencia_%'].round(2)

    # Ordenar por porcentaje de influencia
    shap_df = shap_df.sort_values(by='Influencia_%', ascending=False)

    # Mostrar tabla de influencias con porcentajes
    st.write("### Factores que más influyen en su predicción:")
    st.table(shap_df[['Característica', 'Valor', 'Influencia_%']].head(10))

    # Agregar una nota explicativa
    st.caption("El porcentaje de influencia indica la contribución relativa de cada factor a la predicción, basado en los valores absolutos de influencia.")

    # Crear un gráfico de barras de las influencias en porcentaje
    fig, ax = plt.subplots()
    sns.barplot(x='Influencia_%', y='Característica', data=shap_df.head(10), palette='viridis', ax=ax)
    ax.set_xlabel('Influencia (%)')
    ax.set_ylabel('Característica')
    ax.set_title('Influencia de las características en la predicción')
    st.pyplot(fig)

    # Comparar probabilidades con y sin Identidad Indígena
    # Probabilidad con Ident_Indigena = 0
    features_no_indigena = features.copy()
    features_no_indigena[0][-1] = 0  # Establecer Ident_Indigena a 0
    prob_no_indigena = rf.predict_proba(features_no_indigena)[0][1]

    # Probabilidad con Ident_Indigena = 1
    features_si_indigena = features.copy()
    features_si_indigena[0][-1] = 1  # Establecer Ident_Indigena a 1
    prob_si_indigena = rf.predict_proba(features_si_indigena)[0][1]

    # Mostrar las probabilidades
    st.write("### Efecto de la Identidad Indígena en la Probabilidad de Empleo")
    st.write(f"- **Sin Identidad Indígena:** {prob_no_indigena * 100:.2f}%")
    st.write(f"- **Con Identidad Indígena:** {prob_si_indigena * 100:.2f}%")

    # Crear un gráfico comparativo
    probabilities = [prob_no_indigena * 100, prob_si_indigena * 100]
    labels = ['Sin Identidad Indígena', 'Con Identidad Indígena']

    fig2, ax2 = plt.subplots()
    ax2.bar(labels, probabilities, color=['blue', 'orange'])
    ax2.set_ylabel('Probabilidad de Empleo (%)')
    ax2.set_title('Impacto de la Identidad Indígena en la Probabilidad de Empleo')
    st.pyplot(fig2)

    st.info("Puede ajustar las características y volver a calcular para ver cómo cambia la probabilidad.")
