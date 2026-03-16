"""
Sistema de Investigación de Operaciones
Aplicación Web Completa con Streamlit
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import json
import sys
import os

# -------------------------------------------------

# Configurar PATH dinámico para Streamlit Cloud

# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(**file**))
sys.path.append(BASE_DIR)

# -------------------------------------------------

# Imports de módulos del proyecto

# -------------------------------------------------

from modules.linear_programming import LinearProgrammingModule
from modules.network_analysis import (
TransportationProblem,
AssignmentProblem,
NetworkFlowProblems,
PertCpm
)

from modules.duality_sensitivity import (
DualityAnalysis,
SensitivityAnalysis
)

from modules.integer_programming import IntegerProgramming

from utils.visualizations import *
from utils.export import *

from app_modules import (
show_duality_sensitivity,
show_integer_programming,
show_network_analysis,
show_examples_library,
show_method_comparison,
show_history
)

# -------------------------------------------------

# Configuración de la página

# -------------------------------------------------

st.set_page_config(
page_title="Sistema de Investigación de Operaciones",
page_icon="📊",
layout="wide",
initial_sidebar_state="expanded"
)

# -------------------------------------------------

# CSS personalizado

# -------------------------------------------------

st.markdown(
"""

<style>

.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    padding: 1rem;
}

.module-card {
    background-color: #f0f2f6;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

</style>

""",
unsafe_allow_html=True,
)

# -------------------------------------------------

# Inicializar variables de sesión

# -------------------------------------------------

def initialize_session_state():

```
if "history" not in st.session_state:
    st.session_state.history = []

if "current_solution" not in st.session_state:
    st.session_state.current_solution = None

if "problem_data" not in st.session_state:
    st.session_state.problem_data = None

if "examples_loaded" not in st.session_state:
    st.session_state.examples_loaded = False
```

# -------------------------------------------------

# Función principal

# -------------------------------------------------

def main():

```
initialize_session_state()

# Header
st.markdown(
    '<h1 class="main-header">📊 Sistema de Investigación de Operaciones</h1>',
    unsafe_allow_html=True
)

st.markdown("### Universidad Autónoma de Sinaloa - Ingeniería en Software")
st.markdown("---")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/UAS_Logo.svg/1200px-UAS_Logo.svg.png",
        width=150
    )

    st.markdown("## 🎯 Módulos")

    module = st.radio(
        "Selecciona un módulo:",
        [
            "🏠 Inicio",
            "📈 Programación Lineal",
            "🔄 Dualidad y Sensibilidad",
            "🔢 Programación Entera",
            "🌐 Análisis de Redes",
            "📚 Biblioteca de Ejemplos",
            "📊 Comparar Métodos",
            "💾 Historial",
        ],
    )

    st.markdown("---")

    st.markdown("### ⚙️ Configuración")

    show_steps = st.checkbox("Mostrar pasos detallados", value=True)
    auto_export = st.checkbox("Auto-exportar resultados", value=False)

    st.markdown("---")

    st.markdown("### 📖 Ayuda")

    with st.expander("ℹ️ Guía rápida"):

        st.markdown(
            """
```

1. Selecciona un módulo
2. Ingresa los datos del problema
3. Elige el método de solución
4. Visualiza resultados
5. Exporta si lo deseas
   """
   )

   # -------------------------------------------------

   # Navegación principal

   # -------------------------------------------------

   if module == "🏠 Inicio":
   show_home()

   elif module == "📈 Programación Lineal":
   show_linear_programming()

   elif module == "🔄 Dualidad y Sensibilidad":
   show_duality_sensitivity()

   elif module == "🔢 Programación Entera":
   show_integer_programming()

   elif module == "🌐 Análisis de Redes":
   show_network_analysis()

   elif module == "📚 Biblioteca de Ejemplos":
   show_examples_library()

   elif module == "📊 Comparar Métodos":
   show_method_comparison()

   elif module == "💾 Historial":
   show_history()

# -------------------------------------------------

# Ejecutar app

# -------------------------------------------------

if **name** == "**main**":
main()
