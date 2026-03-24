"""
Sistema de Investigación de Operaciones
Aplicación Web Completa con Streamlit
"""
 
import streamlit as st
import numpy as np
import pandas as pd
import sys
from datetime import datetime
import os
 
# Configurar path dinámico
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
 
from modules.linear_programming import LinearProgrammingModule
from modules.network_analysis import (TransportationProblem, AssignmentProblem,
                                      NetworkFlowProblems, PertCpm)
from modules.duality_sensitivity import DualityAnalysis, SensitivityAnalysis
from modules.integer_programming import IntegerProgramming
from app_modules import (show_duality_sensitivity, show_integer_programming,
                         show_network_analysis, show_examples_library,
                         show_method_comparison, show_history, show_manual)
 
# Configuración de la página
st.set_page_config(
    page_title="Kit Práctico de Investigación de Operaciones",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# CSS personalizado
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; padding: 1rem; }
    .module-card { background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    .phase-header { background-color: #e8f4f8; padding: 0.5rem 1rem; border-left: 4px solid #1f77b4;
                    border-radius: 4px; margin: 1rem 0 0.5rem 0; }
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
# INICIALIZAR SESIÓN
# ─────────────────────────────────────────────
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_solution' not in st.session_state:
        st.session_state.current_solution = None
    if 'problem_data' not in st.session_state:
        st.session_state.problem_data = None
    if 'examples_loaded' not in st.session_state:
        st.session_state.examples_loaded = False
 
 
# ─────────────────────────────────────────────
# HELPER: mostrar tabla de iteraciones
# ─────────────────────────────────────────────
def _show_iterations_table(iterations, label):
    """Muestra las iteraciones de una fase en un expander."""
    num_iters = max(0, len(iterations) - 1)
    with st.expander(f"📊 {label} — {num_iters} iteración(es)"):
        for it in iterations:
            st.markdown(f"**Iteración {it['iteration']}** — {it['description']}")
            col_labels = it['var_names'] + ['RHS']
            row_labels = (
                [f"R{r+1}" for r in range(len(it['tableau']) - 1)] + ['Z']
            )
            df = pd.DataFrame(
                it['tableau'],
                columns=col_labels,
                index=row_labels
            )
            # Resaltar fila Z y columna RHS
            st.dataframe(df.style.format("{:.4f}"))
            st.markdown("---")
 
 
# ─────────────────────────────────────────────
# MÓDULO DE PROGRAMACIÓN LINEAL
# ─────────────────────────────────────────────
def show_linear_programming():
    st.markdown("## 📈 Programación Lineal")
 
    lp = LinearProgrammingModule()
 
    col1, col2 = st.columns([1, 2])
 
    with col1:
        st.markdown("### ⚙️ Configuración del Problema")
 
        n_vars = st.number_input("Número de variables", min_value=2, max_value=10, value=2, step=1)
        n_constraints = st.number_input("Número de restricciones", min_value=1, max_value=10, value=2, step=1)
 
        obj_type = st.radio("Tipo de optimización", ["Maximizar", "Minimizar"])
        maximize = obj_type == "Maximizar"
 
        method = st.selectbox(
            "Método de solución",
            ["simplex", "graphical", "two_phase", "pulp"],
            format_func=lambda x: {
                "simplex":    "Método Simplex",
                "graphical":  "Método Gráfico (solo 2 variables)",
                "two_phase":  "Método de las Dos Fases",
                "pulp":       "PuLP (Optimizador)"
            }[x]
        )
 
        if method == "graphical" and n_vars != 2:
            st.warning("⚠️ El método gráfico requiere exactamente 2 variables.")
 
        st.markdown("### 🎯 Función Objetivo")
        var_names = [f"x{i+1}" for i in range(n_vars)]
        c = np.array([
            st.number_input(f"Coeficiente {var_names[i]}", value=1.0, key=f"c_{i}")
            for i in range(n_vars)
        ])
 
        st.markdown("### 📋 Restricciones")
        A_list, b_list, ct_list = [], [], []
        for i in range(n_constraints):
            st.markdown(f"**Restricción {i+1}**")
            cols = st.columns(n_vars + 2)
            row = [
                cols[j].number_input(f"{var_names[j]}", value=1.0, key=f"A_{i}_{j}")
                for j in range(n_vars)
            ]
            ct  = cols[n_vars].selectbox("Tipo", ["<=", ">=", "="], key=f"ct_{i}")
            rhs = cols[n_vars + 1].number_input("RHS", value=10.0, key=f"b_{i}")
            A_list.append(row)
            b_list.append(rhs)
            ct_list.append(ct)
 
        solve_btn = st.button("🚀 Resolver", type="primary")
 
    # ── Resultados ──────────────────────────────
    with col2:
        if solve_btn:
            A = np.array(A_list)
            b = np.array(b_list)
 
            valid, msg = lp.validate_input(c, A, b)
            if not valid:
                st.error(f"❌ Error en los datos: {msg}")
                return
 
            with st.spinner("Resolviendo..."):
                solution = lp.solve(c, A, b, method, maximize, var_names, ct_list)
 
            # ── Resultado óptimo ─────────────────
            if solution['status'] == 'optimal':
                st.success("✅ Solución óptima encontrada")
 
                st.markdown("### 📐 Formulación")
                st.code(lp.get_problem_formulation(), language="")
 
                st.markdown("### 🏆 Resultado")
                res_cols = st.columns(1 + len(solution['variables']))
                res_cols[0].metric("Valor Óptimo (Z)", f"{solution['optimal_value']:.4f}")
                for idx, (var, val) in enumerate(solution['variables'].items()):
                    res_cols[idx + 1].metric(var, f"{val:.4f}")
 
                # Gráfico (método gráfico)
                if 'figure' in solution:
                    st.plotly_chart(solution['figure'], use_container_width=True)
 
                # ── Iteraciones Simplex estándar ─
                if method == 'simplex' and solution.get('iterations'):
                    st.markdown("### 🔢 Iteraciones del Simplex")
                    _show_iterations_table(solution['iterations'], "Simplex")
 
                # ── Dos Fases: Fase 1 arriba, Fase 2 abajo ─
                if method == 'two_phase':
                    # Resumen de fases
                    ph_col1, ph_col2, ph_col3 = st.columns(3)
                    ph_col1.metric("Iteraciones Fase 1", solution.get('num_phase1', 0))
                    ph_col2.metric("Iteraciones Fase 2", solution.get('num_phase2', 0))
                    ph_col3.metric("Total iteraciones",  solution.get('total_iterations', 0))
 
                    st.markdown("---")
 
                    # FASE 1
                    st.markdown(
                        '<div class="phase-header">🔵 FASE 1 — Encontrar Solución Básica Factible</div>',
                        unsafe_allow_html=True
                    )
                    st.info(
                        "En la Fase 1 se minimiza la suma de variables artificiales. "
                        "Si el óptimo es 0, existe una solución factible y se procede a la Fase 2."
                    )
                    if solution.get('phase1_iterations'):
                        _show_iterations_table(solution['phase1_iterations'], "Fase 1")
                    else:
                        st.caption("No hubo iteraciones en Fase 1.")
 
                    st.markdown("---")
 
                    # FASE 2
                    st.markdown(
                        '<div class="phase-header">🟢 FASE 2 — Optimizar Función Objetivo Original</div>',
                        unsafe_allow_html=True
                    )
                    st.info(
                        "En la Fase 2 se elimina las variables artificiales y se optimiza "
                        "con la función objetivo original partiendo de la base factible."
                    )
                    if solution.get('phase2_iterations'):
                        _show_iterations_table(solution['phase2_iterations'], "Fase 2")
                    else:
                        st.caption("No hubo iteraciones en Fase 2.")
 
                # Guardar en historial
                st.session_state.history.append({
                    'timestamp':     datetime.now().strftime("%H:%M:%S"),
                    'method':        solution.get('method', method),
                    'optimal_value': solution['optimal_value'],
                    'variables':     solution['variables']
                })
 
            # ── Infactible ───────────────────────
            elif solution['status'] == 'infeasible':
                st.error("❌ El problema no tiene solución factible.")
 
                # Mostrar Fase 1 aunque sea infactible (útil pedagógicamente)
                if method == 'two_phase' and solution.get('phase1_iterations'):
                    st.markdown("### 🔵 Fase 1 — Proceso (no alcanzó Z = 0)")
                    _show_iterations_table(solution['phase1_iterations'], "Fase 1")
 
            elif solution['status'] == 'unbounded':
                st.error("❌ El problema no está acotado (solución infinita).")
 
            elif solution['status'] == 'error':
                st.error(f"❌ {solution.get('message', 'Error desconocido')}")
 
            else:
                st.warning(f"⚠️ Estado inesperado: {solution['status']}")
 
 
# ─────────────────────────────────────────────
# PANTALLA DE INICIO
# ─────────────────────────────────────────────
def show_home():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("## 👋 Bienvenido al Sistema de IO")
        st.markdown(
            "Este sistema te permite resolver problemas de **Investigación de Operaciones** "
            "de manera interactiva, con visualizaciones y exportación de resultados."
        )
        modules_info = {
            "Programación Lineal":      ["Método Simplex", "Método Gráfico", "Método de las Dos Fases"],
            "Dualidad y Sensibilidad":  ["Problema Dual", "Precios Sombra", "Análisis de Sensibilidad"],
            "Programación Entera":      ["Branch and Bound", "Cortes de Gomory", "Problema de la Mochila"],
            "Análisis de Redes":        ["Problema de Transporte", "Problema de Asignación", "Flujo Máximo", "PERT-CPM"]
        }
        for module_name, topics in modules_info.items():
            with st.expander(f"📖 {module_name}"):
                for topic in topics:
                    st.markdown(f"- {topic}")
 
        # Logos institucionales
        st.markdown("---")
        st.markdown("#### 🏫 Instituciones")
        logo1, logo2, logo3 = st.columns(3)
        with logo1:
            st.image("assets/logo1.jpg", width=130, caption="UAS")
        with logo2:
            st.image("assets/logo2.jpg", width=130, caption="Facultad de Ingeniería")
        with logo3:
            st.image("assets/logo3.png", width=130, caption="UAS 2029")
 
        # Fecha de última actualización debajo de los logos
        st.caption("🔄 Última actualización: 20 de marzo de 2026")
 
    with col2:
        st.markdown("## 🚀 Comenzar")
        st.info("**Pasos rápidos:**\n1. Selecciona un módulo del menú\n2. Ingresa tus datos\n3. ¡Resuelve!")
        st.markdown("### 📊 Estadísticas")
        st.metric("Problemas Resueltos", len(st.session_state.history))
        if st.session_state.history:
            recent = st.session_state.history[-1]
            st.metric("Último Método", recent.get('method', 'N/A'))
 
 
# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────
def main():
    initialize_session_state()
 
    # Configurar locale en español
    try:
        import locale
        locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
    except Exception:
        pass
 
    # Header
    st.markdown(
        '<h1 class="main-header">📊 Kit Práctico de Investigación de Operaciones</h1>',
        unsafe_allow_html=True
    )
    st.markdown("### Universidad Autónoma de Sinaloa - Ingeniería en Desarrollo de Software")
    st.markdown("👨‍💻 Desarrollado por: **Cristóbal Lemus**")
    st.caption(f"📅 Fecha actual: {datetime.now().strftime('%d de %B de %Y')}")
    st.markdown("---")
 
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Módulos")
        module = st.radio("Selecciona un módulo:", [
            "🏠 Inicio",
            "📈 Programación Lineal",
            "🔄 Dualidad y Sensibilidad",
            "🔢 Programación Entera",
            "🌐 Análisis de Redes",
            "📚 Biblioteca de Ejemplos",
            "📊 Comparar Métodos",
            "💾 Historial",
            "📖 Manual de Ejercicios"
        ])
        st.markdown("---")
        st.markdown("### ⚙️ Configuración")
        st.checkbox("Mostrar pasos detallados", value=True)
        st.checkbox("Auto-exportar resultados", value=False)
        st.markdown("---")
        st.markdown("### 📖 Ayuda")
        with st.expander("ℹ️ Guía rápida"):
            st.markdown(
                "1. Selecciona un módulo\n"
                "2. Ingresa los datos del problema\n"
                "3. Elige el método de solución\n"
                "4. Visualiza resultados\n"
                "5. Exporta si lo deseas"
            )
 
    # Enrutador de módulos
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
    elif module == "📖 Manual de Ejercicios":
        show_manual()
 
 
if __name__ == "__main__":
    main()