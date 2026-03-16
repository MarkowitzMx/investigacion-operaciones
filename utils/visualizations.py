"""
Módulo de visualizaciones para Investigación de Operaciones
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def create_simplex_tableau_table(iteration_data: Dict) -> go.Figure:
    """Crea tabla HTML del tableau del Simplex"""

    tableau = iteration_data['tableau']
    var_names = iteration_data['var_names']
    basis = iteration_data['basis']

    m, n = tableau.shape

    headers = ['Base'] + var_names + ['RHS']

    rows = []
    for i in range(m - 1):
        row = [var_names[basis[i]]] + [f"{tableau[i, j]:.4f}" for j in range(n)]
        rows.append(row)

    z_row = ['Z'] + [f"{tableau[-1, j]:.4f}" for j in range(n)]
    rows.append(z_row)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color='paleturquoise',
            align='center',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color='lavender',
            align='center',
            font=dict(size=11)
        )
    )])

    fig.update_layout(
        title=f"Iteración {iteration_data['iteration']}: {iteration_data['description']}",
        height=300 + 30 * len(rows)
    )

    return fig


def create_convergence_plot(iterations: List[Dict]) -> go.Figure:
    """Crea gráfico de convergencia del valor objetivo"""

    iteration_nums = []
    objective_values = []

    for it in iterations:
        iteration_nums.append(it['iteration'])
        objective_values.append(it['tableau'][-1, -1])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iteration_nums,
        y=objective_values,
        mode='lines+markers',
        name='Valor Objetivo',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))

    fig.update_layout(
        title='Convergencia del Método Simplex',
        xaxis_title='Iteración',
        yaxis_title='Valor de la Función Objetivo',
        hovermode='x unified',
        showlegend=True,
        height=400
    )

    return fig


def create_variable_evolution_plot(iterations: List[Dict], num_original_vars: int) -> go.Figure:
    """Muestra evolución de variables a través de las iteraciones"""

    fig = go.Figure()

    for var_idx in range(num_original_vars):

        values = []
        iteration_nums = []

        for it in iterations:
            iteration_nums.append(it['iteration'])
            tableau = it['tableau']
            basis = it['basis']

            if var_idx in basis:
                row_idx = basis.index(var_idx)
                values.append(tableau[row_idx, -1])
            else:
                values.append(0.0)

        var_name = iterations[0]['var_names'][var_idx] if var_idx < len(iterations[0]['var_names']) else f'x{var_idx+1}'

        fig.add_trace(go.Scatter(
            x=iteration_nums,
            y=values,
            mode='lines+markers',
            name=var_name,
            line=dict(width=2),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title='Evolución de Variables',
        xaxis_title='Iteración',
        yaxis_title='Valor',
        hovermode='x unified',
        height=400
    )

    return fig


def create_network_graph(nodes: List, edges: List[Tuple],
                         edge_labels: Dict = None,
                         node_colors: Dict = None) -> go.Figure:
    """Crea visualización de red para problemas de transporte/asignación"""

    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42, k=2)

    edge_traces = []

    for edge in G.edges():

        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color='#888'),
            hoverinfo='none'
        )

        edge_traces.append(edge_trace)

    node_x = []
    node_y = []
    node_text = []
    node_color_list = []

    for node in G.nodes():

        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

        if node_colors and node in node_colors:
            node_color_list.append(node_colors[node])
        else:
            node_color_list.append('lightblue')

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(
            size=30,
            color=node_color_list,
            line=dict(width=2, color='black')
        ),
        hoverinfo='text'
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title='Red de Flujo',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )

    return fig


def create_gantt_chart(tasks: List[Dict]) -> go.Figure:
    """Crea diagrama de Gantt para PERT/CPM"""

    df = pd.DataFrame(tasks)

    fig = px.timeline(
        df,
        x_start='start',
        x_end='finish',
        y='task',
        color='resource',
        title='Diagrama de Gantt'
    )

    fig.update_yaxes(categoryorder='total ascending')
    fig.update_layout(height=400)

    return fig


def create_sensitivity_plot(parameter_range: np.ndarray,
                            objective_values: np.ndarray,
                            parameter_name: str,
                            current_value: float) -> go.Figure:
    """Crea gráfico de análisis de sensibilidad"""

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=parameter_range,
        y=objective_values,
        mode='lines',
        name='Valor Objetivo',
        line=dict(color='blue', width=3)
    ))

    fig.add_vline(
        x=current_value,
        line_dash="dash",
        line_color="red",
        annotation_text="Valor Actual"
    )

    fig.update_layout(
        title=f'Análisis de Sensibilidad: {parameter_name}',
        xaxis_title=parameter_name,
        yaxis_title='Valor de Z',
        hovermode='x unified',
        height=400
    )

    return fig


def create_comparison_table(methods_results: Dict[str, Dict]) -> go.Figure:
    """Crea tabla comparativa de diferentes métodos"""

    methods = list(methods_results.keys())

    headers = ['Método', 'Valor Óptimo', 'Iteraciones', 'Tiempo (s)', 'Estado']

    rows = []

    for method in methods:

        result = methods_results[method]

        opt = result.get('optimal_value')
        opt_value = f"{opt:.4f}" if isinstance(opt, (int, float)) else "N/A"

        rows.append([
            method,
            opt_value,
            str(result.get('num_iterations', 'N/A')),
            f"{result.get('time', 0):.4f}",
            result.get('status', 'N/A')
        ])

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color='paleturquoise',
            align='center',
            font=dict(size=13, color='black')
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color='lavender',
            align='center',
            font=dict(size=12)
        )
    )])

    fig.update_layout(
        title='Comparación de Métodos',
        height=200 + 40 * len(methods)
    )

    return fig


def create_3d_surface(x_range: np.ndarray,
                      y_range: np.ndarray,
                      z_values: np.ndarray,
                      var_names: List[str]) -> go.Figure:
    """Crea superficie 3D para visualizar función objetivo"""

    fig = go.Figure(data=[go.Surface(
        x=x_range,
        y=y_range,
        z=z_values,
        colorscale='Viridis'
    )])

    fig.update_layout(
        title='Superficie de la Función Objetivo',
        scene=dict(
            xaxis_title=var_names[0] if len(var_names) > 0 else 'x1',
            yaxis_title=var_names[1] if len(var_names) > 1 else 'x2',
            zaxis_title='Z'
        ),
        height=600
    )

    return fig
