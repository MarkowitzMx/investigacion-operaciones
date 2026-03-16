import streamlit as st

st.set_page_config(
    page_title="Investigación de Operaciones",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Plataforma de Investigación de Operaciones")

st.write("""
Bienvenido a la plataforma interactiva para aprender **Investigación de Operaciones**.

Aquí podrás explorar distintos métodos:

• Método gráfico  
• Método Simplex  
• Redes de transporte  
• PERT / CPM
""")

st.sidebar.title("Menú")

opcion = st.sidebar.selectbox(
    "Selecciona un método",
    [
        "Inicio",
        "Método gráfico",
        "Método Simplex",
        "Transporte",
        "PERT / CPM"
    ]
)

if opcion == "Inicio":
    st.success("Selecciona un método en el menú lateral.")

elif opcion == "Método gráfico":
    st.header("Método gráfico para programación lineal")

elif opcion == "Método Simplex":
    st.header("Método Simplex")

elif opcion == "Transporte":
    st.header("Problema de transporte")

elif opcion == "PERT / CPM":
    st.header("Análisis PERT / CPM")
