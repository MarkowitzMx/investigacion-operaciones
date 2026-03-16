"""
Sistema de Investigación de Operaciones
Aplicación Web Completa con Streamlit
"""
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import json
 
# Configurar path

 
from modules.linear_programming import LinearProgrammingModule
from modules.network_analysis import (TransportationProblem, AssignmentProblem, 
                                      NetworkFlowProblems, PertCpm)
from modules.duality_sensitivity import DualityAnalysis, SensitivityAnalysis
from modules.integer_programming import IntegerProgramming
from utils.visualizations import *
from utils.export import *
from app_modules import (show_duality_sensitivity, show_integer_programming,
                        show_network_analysis, show_examples_library,
                        show_method_comparison, show_history)
 
# Configuración de la página
st.set_page_config(
    page_title="Sistema de Investigación de Operaciones",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# CSS personalizado
st.markdown("""
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
""", unsafe_allow_html=True)
 
 
def initialize_session_state():
    """Inicializa variables de sesión"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_solution' not in st.session_state:
        st.session_state.current_solution = None
    if 'problem_data' not in st.session_state:
        st.session_state.problem_data = None
    if 'examples_loaded' not in st.session_state:
        st.session_state.examples_loaded = False
 
 
def main():
    """Función principal"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📊 Sistema de Investigación de Operaciones</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Universidad Autónoma de Sinaloa - Ingeniería en Software")
    st.markdown("---")
    
    # Sidebar con navegación
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/UAS_Logo.svg/1200px-UAS_Logo.svg.png", 
                 width=150)
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
                "💾 Historial"
            ]
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuración")
        show_steps = st.checkbox("Mostrar pasos detallados", value=True)
        auto_export = st.checkbox("Auto-exportar resultados", value=False)
        
        st.markdown("---")
        st.markdown("### 📖 Ayuda")
        with st.expander("ℹ️ Guía rápida"):
            st.markdown("""
            1. Selecciona un módulo
            2. Ingresa los datos del problema
            3. Elige el método de solución
            4. Visualiza resultados
            5. Exporta si lo deseas
            """)
    
    # Contenido principal según módulo seleccionado
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
 
 
def show_home():
    """Página de inicio"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 👋 Bienvenido al Sistema de IO")
        st.markdown("""
        Este sistema te permite resolver problemas de **Investigación de Operaciones** 
        de manera interactiva, con visualizaciones y exportación de resultados.
        
        ### 📚 Contenido del Curso:
        """)
        
        modules_info = {
            "Programación Lineal": [
                "Método Simplex",
                "Método Gráfico",
                "Método de las Dos Fases"
            ],
            "Dualidad y Sensibilidad": [
                "Problema Dual",
                "Precios Sombra",
                "Análisis de Sensibilidad"
            ],
            "Programación Entera": [
                "Branch and Bound",
                "Cortes de Gomory",
                "Problema de la Mochila"
            ],
            "Análisis de Redes": [
                "Problema de Transporte",
                "Problema de Asignación",
                "Flujo Máximo",
                "PERT-CPM"
            ]
        }
        
        for module_name, topics in modules_info.items():
            with st.expander(f"📖 {module_name}"):
                for topic in topics:
                    st.markdown(f"- {topic}")
    
    with col2:
        st.markdown("## 🚀 Comenzar")
        st.info("""
        **Pasos rápidos:**
        1. Selecciona un módulo del menú
        2. Ingresa tus datos
        3. ¡Resuelve!
        """)
        
        st.markdown("### 📊 Estadísticas")
        st.metric("Problemas Resueltos", len(st.session_state.history))
        
        if st.session_state.history:
            recent = st.session_state.history[-1]
            st.metric("Último Método", recent.get('method', 'N/A'))
 
 
def show_linear_programming():
    """Módulo de Programación Lineal"""
    st.markdown("## 📈 Programación Lineal")
    
    tab1, tab2, tab3 = st.tabs(["📝 Definir Problema", "🔧 Resolver", "📊 Resultados"])
    
    with tab1:
        st.markdown("### Definición del Problema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            objective_type = st.radio("Tipo de optimización:", ["Maximizar", "Minimizar"])
            maximize = (objective_type == "Maximizar")
            
            num_vars = st.number_input("Número de variables:", min_value=2, max_value=10, value=2)
            num_constraints = st.number_input("Número de restricciones:", 
                                             min_value=1, max_value=20, value=2)
        
        with col2:
            method = st.selectbox(
                "Método de solución:",
                ["Simplex", "Gráfico (solo 2 variables)", "Dos Fases", "PuLP (Optimizador)"]
            )
        
        st.markdown("### Función Objetivo")
 
        # Primero recoger los coeficientes
        c_input = []
        cols = st.columns(num_vars)
        for i in range(num_vars):
            with cols[i]:
                coef = st.number_input(f"c{i+1}", value=1.0, key=f"c{i}")
                c_input.append(coef)
 
        # Luego mostrar la función objetivo construida en una sola línea
        terms = " + ".join([f"{c_input[i]:g}·x{i+1}" for i in range(num_vars)])
        st.markdown(f"**{objective_type}** &nbsp; Z = {terms}")
        
        st.markdown("### Restricciones")
        
        A_input = []
        b_input = []
        constraint_types_input = []
        
        for i in range(num_constraints):
            st.markdown(f"**Restricción {i+1}:**")
            cols = st.columns(num_vars + 2)
            
            row = []
            for j in range(num_vars):
                with cols[j]:
                    val = st.number_input(f"a{i+1}{j+1}", value=1.0, key=f"a{i}{j}")
                    row.append(val)
            
            with cols[num_vars]:
                constraint_type = st.selectbox("", ["<=", ">=", "="], 
                                               key=f"type{i}", label_visibility="collapsed")
                constraint_types_input.append(constraint_type)
            
            with cols[num_vars + 1]:
                b_val = st.number_input(f"b{i+1}", value=10.0, key=f"b{i}")
                b_input.append(b_val)
            
            A_input.append(row)
        
        # Botón para guardar problema
        if st.button("💾 Guardar Problema", type="primary"):
            st.session_state.problem_data = {
                'c': np.array(c_input),
                'A': np.array(A_input),
                'b': np.array(b_input),
                'maximize': maximize,
                'constraint_types': constraint_types_input,
                'var_names': [f"x{i+1}" for i in range(num_vars)],
                'method_name': method
            }
            st.success("✅ Problema guardado correctamente")
    
    with tab2:
        st.markdown("### Resolver Problema")
        
        if st.session_state.problem_data is None:
            st.warning("⚠️ Primero debes definir un problema en la pestaña 'Definir Problema'")
        else:
            # Mostrar formulación
            with st.expander("📋 Ver Formulación Matemática"):
                lp_module = LinearProgrammingModule()
                lp_module.problem_data = st.session_state.problem_data
                formulation = lp_module.get_problem_formulation()
                st.code(formulation)
            
            if st.button("🚀 Resolver", type="primary", use_container_width=True):
                with st.spinner("Resolviendo..."):
                    solve_linear_programming()
    
    with tab3:
        st.markdown("### Resultados")
        show_lp_results()
 
 
def solve_linear_programming():
    """Resuelve problema de programación lineal"""
    problem = st.session_state.problem_data
    
    # Mapear método
    method_map = {
        "Simplex": "simplex",
        "Gráfico (solo 2 variables)": "graphical",
        "Dos Fases": "two_phase",
        "PuLP (Optimizador)": "pulp"
    }
    
    method = method_map[problem['method_name']]
    
    # Resolver
    lp_module = LinearProgrammingModule()
    
    try:
        solution = lp_module.solve(
            c=problem['c'],
            A=problem['A'],
            b=problem['b'],
            method=method,
            maximize=problem['maximize'],
            var_names=problem['var_names'],
            constraint_types=problem['constraint_types']
        )
        
        st.session_state.current_solution = solution
        
        # Agregar al historial
        st.session_state.history.append({
            'timestamp': datetime.now(),
            'module': 'Programación Lineal',
            'method': problem['method_name'],
            'status': solution.get('status', 'unknown'),
            'objective_value': solution.get('optimal_value', None)
        })
        
        st.success(f"✅ Problema resuelto: {solution.get('status', 'unknown').upper()}")
        
    except Exception as e:
        st.error(f"❌ Error al resolver: {str(e)}")
        st.exception(e)
 
 
def show_lp_results():
    """Muestra resultados de programación lineal"""
    if st.session_state.current_solution is None:
        st.info("ℹ️ No hay resultados que mostrar. Resuelve un problema primero.")
        return
    
    solution = st.session_state.current_solution
    
    # Resumen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Estado", solution.get('status', 'N/A').upper())
    with col2:
        st.metric("Valor Óptimo", f"{solution.get('optimal_value', 0):.4f}")
    with col3:
        if 'num_iterations' in solution:
            st.metric("Iteraciones", solution.get('num_iterations', 'N/A'))
    
    # Variables
    if 'variables' in solution:
        st.markdown("### 🎯 Variables de Decisión")
        vars_df = pd.DataFrame([
            {'Variable': k, 'Valor': v}
            for k, v in solution['variables'].items()
        ])
        st.dataframe(vars_df, use_container_width=True)
    
    # Visualizaciones según método
    if solution.get('method') == 'Método Gráfico' and 'figure' in solution:
        st.markdown("### 📊 Visualización Gráfica")
        st.plotly_chart(solution['figure'], use_container_width=True)
    
    # Iteraciones del Simplex
    if 'iterations' in solution and len(solution['iterations']) > 0:
        st.markdown("### 📝 Iteraciones del Método Simplex")
        
        iteration_selector = st.selectbox(
            "Selecciona iteración:",
            range(len(solution['iterations'])),
            format_func=lambda x: f"Iteración {x}"
        )
        
        it_data = solution['iterations'][iteration_selector]
        fig_tableau = create_simplex_tableau_table(it_data)
        st.plotly_chart(fig_tableau, use_container_width=True)
        
        # Gráfico de convergencia
        if len(solution['iterations']) > 1:
            st.markdown("### 📈 Convergencia")
            fig_conv = create_convergence_plot(solution['iterations'])
            st.plotly_chart(fig_conv, use_container_width=True)
    
    # Exportar
    st.markdown("### 💾 Exportar Resultados")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        excel_data = export_to_excel(solution)
        st.download_button(
            label="📥 Descargar Excel",
            data=excel_data,
            file_name=f"solucion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        json_data = export_to_json(solution)
        st.download_button(
            label="📥 Descargar JSON",
            data=json_data,
            file_name=f"solucion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        report_text = export_report_text(solution)
        st.download_button(
            label="📥 Descargar Reporte",
            data=report_text,
            file_name=f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
 
 
if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 5af850f (update app structure and dependencies)
