import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Importar librerías para gráficos
import shap
from streamlit_shap import st_shap

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Probabilidad de Empleo",
    layout="centered",
    initial_sidebar_state="auto"
)

# Función para cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load('random_forest_modelSTM2.joblib')

# Cargar el modelo
rf = load_model()

# Inicializar el estado de la aplicación
if 'app_started' not in st.session_state:
    st.session_state.app_started = False

# Título de la aplicación
st.title("🧑‍💼 Predicción de Probabilidad de Empleo")

# --- Página de inicio ---
if not st.session_state.app_started:
    # --- Introducción y Contexto ---
    st.header("Introducción y Contexto")
    st.markdown("""
    La discriminación lingüística hacia los hablantes de lenguas indígenas es una problemática significativa en México. A pesar de ser un país con 68 lenguas indígenas registradas y una riqueza cultural invaluable, los hablantes de estas lenguas enfrentan barreras estructurales y culturales que limitan su acceso a oportunidades laborales.

    Datos del Censo de Población y Vivienda 2020 muestran que la probabilidad de empleo disminuye un 10.98% para los hablantes de lenguas indígenas, incluso después de ajustar por factores sociodemográficos como educación, edad y género. De acuerdo con la Ley General de Derechos Lingüísticos de los Pueblos Indígenas, estas comunidades tienen el derecho de comunicarse en su lengua, sin ninguna forma de restricción en todas sus actividades sociales (Ley General de Derechos Lingüísticos de los Pueblos Indígenas, 2003, Art. 9). Esto significa que el idioma no debería representar una barrera para las oportunidades de empleo de quienes hablan lenguas indígenas. Sin embargo, en la práctica, esta teoría a menudo se aleja de la realidad.

    Esta exclusión no solo afecta la equidad social, sino que también perpetúa la desigualdad económica, particularmente entre las comunidades indígenas más vulnerables. En este contexto, comprender el impacto de las barreras lingüísticas en la inserción laboral y desarrollar estrategias para mitigar estas disparidades es esencial.

    ### Objetivo de la Herramienta Interactiva
    La calculadora de probabilidad de empleo diseñada como una aplicación tiene como objetivo:

    - **Visualizar** el impacto de factores lingüísticos y sociodemográficos sobre las probabilidades de empleo.
    - **Fomentar** una fácil comprensión de los resultados del análisis, al presentar de manera interactiva cómo distintas variables (como el bilingüismo o el nivel educativo) afectan las oportunidades laborales.
    - **Sensibilizar** a los usuarios acerca de la discriminación lingüística y su rol en perpetuar la desigualdad en el mercado laboral mexicano.

    Mediante esta herramienta, buscamos no solo presentar los resultados de nuestro modelo de análisis, sino también ofrecer una plataforma educativa que conecte estos datos con su contexto real, ayudando a generar conciencia y apoyar en el diseño de políticas públicas más inclusivas.
    """)

    # --- Presentación de los Autores ---
    st.header("Autores del Trabajo")

    # Información de los autores (reemplaza con tus datos y enlaces)
    autores = [
        {
            'nombre': 'Santiago Tejerina Marion',
            'descripcion': 'Carrera: Economía\nSemestre: 7mo',
            'imagen': 'https://media.licdn.com/dms/image/v2/D4E03AQFpwzaRL1NWfA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1676928415244?e=1738195200&v=beta&t=5nEyBjyvZWoab6voJqRWU7iLyX3h2OYLbjkjixyVUPM',  # Reemplaza con el enlace a la foto del autor
            'cv': 'https://www.linkedin.com/in/santiago-tejerina-marion/'   # Reemplaza con el enlace al CV del autor
        },
        {
            'nombre': 'Uriel Alejandro Zavala Arrambide',
            'descripcion': 'Carrera: Economía\nSemestre: 7mo',
            'imagen': 'https://media.licdn.com/dms/image/v2/D4E03AQFYUiG7AkIKTw/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1731301872073?e=1738195200&v=beta&t=ByKBe3tiowftTvXOWUlFxfTDQG2g4V-rg0DQutlguLc',
            'cv': 'https://www.linkedin.com/in/uriel-zavala/'
        },
        {
            'nombre': 'Edgar Karel Martínez Mendoza',
            'descripcion': 'Carrera: Economía\nSemestre: 7mo',
            'imagen': 'https://media.licdn.com/dms/image/v2/D5603AQG6f787eTDx-g/profile-displayphoto-shrink_800_800/B56ZN6RvJMGwAg-/0/1732923290996?e=1738800000&v=beta&t=5k9sRPAAMEJWLlctSMgmrrF6gMJVZ3-hUDje5o6OYe0',
            'cv': 'https://www.linkedin.com/in/edgar-karel-mart%C3%ADnez-mendoza-80a14b259/'
        },
        {
            'nombre': 'Mauricio Bernal Cisneros',
            'descripcion': 'Carrera: Economía\nSemestre: 7mo',
            'imagen': 'https://media.licdn.com/dms/image/v2/D5603AQGXNPSbFdvbUA/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1732419327764?e=1738195200&v=beta&t=aGNaCGqnRPsmgBoScbAYafattDyUsC0jApq4cs37boQ',
            'cv': 'https://www.linkedin.com/in/mauricio-bernal-cisneros-138a88249/'
        }
    ]

    # Mostrar información de los autores
    cols = st.columns(len(autores))

    for idx, col in enumerate(cols):
        autor = autores[idx]
        with col:
            st.image(autor['imagen'], width=150, caption=autor['nombre'])
            st.markdown(f"**{autor['nombre']}**")
            st.markdown(autor['descripcion'])
            st.markdown(f"[Ver Linkedin]({autor['cv']})")

    # Espacio adicional
    st.write("\n")

    # Centrar el botón "Iniciar Aplicación"
    col_empty1, col_button, col_empty2 = st.columns([1, 2, 1])
    with col_button:
        if st.button("Iniciar Aplicación"):
            st.session_state.app_started = True


# --- Aplicación Interactiva ---
if st.session_state.app_started:
    # Botón para regresar a la página de inicio
    if st.button("Regresar a la Página de Inicio"):
        st.session_state.app_started = False


    # --- Entradas del usuario ---
    st.header("Ingrese sus datos a continuación")

    st.write("Complete el siguiente formulario para calcular la probabilidad de que usted tenga empleo y descubra qué factores influyen más en su situación laboral.")

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

    # Lista de variables binarias
    binary_features = ['jefehogar', 'hombre', 'rural', 'HLENGUA', 'hombrecasado', 'casado', 'Ident_Indigena']
    binary_feature_names = [feature_name_mapping[name] for name in binary_features]

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
        influencia = shap_values[0][:, 1]  # Instancia 0, todas las características, clase 1

        # Crear una lista de nombres amigables de características
        user_friendly_feature_names = [feature_name_mapping.get(name, name) for name in feature_names]

        # Crear un DataFrame para los valores SHAP
        shap_df = pd.DataFrame({
            'Característica': user_friendly_feature_names,
            'Valor': features[0],
            'Influencia': influencia
        })

        # Mapear los valores de las variables binarias a 'Sí' o 'No'
        def map_binary_value(row):
            if row['Característica'] in binary_feature_names:
                return 'Sí' if row['Valor'] == 1 else 'No'
            else:
                return row['Valor']

        shap_df['Valor'] = shap_df.apply(map_binary_value, axis=1)

        # Redondear los valores de influencia
        shap_df['Influencia'] = shap_df['Influencia'].round(4)

        # Calcular el valor absoluto de las influencias
        shap_df['Influencia_abs'] = np.abs(shap_df['Influencia'])

        # Calcular el porcentaje de influencia
        total_influencia_abs = shap_df['Influencia_abs'].sum()
        shap_df['Influencia_%'] = (shap_df['Influencia_abs'] / total_influencia_abs) * 100
        shap_df['Influencia_%'] = shap_df['Influencia_%'].round(2)

        # Crear una interpretación para cada característica
        def interpretar_influencia(row):
            efecto = 'aumenta' if row['Influencia'] > 0 else 'disminuye'
            return f"{efecto.capitalize()} su probabilidad de empleo en un {row['Influencia_%']}%"

        shap_df['Interpretación'] = shap_df.apply(interpretar_influencia, axis=1)

        # Ordenar por valor absoluto de influencia
        shap_df = shap_df.sort_values(by='Influencia_abs', ascending=False)

        # Mostrar tabla de influencias con interpretaciones
        st.write("### Factores que más influyen en su predicción:")
        st.table(shap_df[['Característica', 'Valor', 'Influencia', 'Interpretación']].head(10))

        # Mostrar gráfico de SHAP values
        st.write("### Visualización de la influencia de cada factor:")
        shap.initjs()
        force_plot = shap.force_plot(explainer.expected_value[1], influencia, features[0], feature_names=user_friendly_feature_names)
        st_shap(force_plot)

        st.info("Puede ajustar las características y volver a calcular para ver cómo cambia la probabilidad.")

