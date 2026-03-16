"""
Módulo de exportación de resultados a diferentes formatos
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import json
from datetime import datetime
import io

def export_to_excel(solution: Dict, filename: str = None) -> bytes:
    """
    Exporta solución a archivo Excel
    
    Returns:
        bytes: Contenido del archivo Excel
    """
    
    if filename is None:
        filename = f"solucion_IO_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Hoja 1: Resumen de la solución
        summary_data = {
            'Campo': ['Estado', 'Valor Óptimo', 'Método', 'Iteraciones', 'Fecha'],
            'Valor': [
                solution.get('status', 'N/A'),
                f"{solution.get('optimal_value', 0):.6f}",
                solution.get('method', 'N/A'),
                solution.get('num_iterations', 'N/A'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Resumen', index=False)
        
        # Hoja 2: Variables de decisión
        if 'variables' in solution:
            variables_data = {
                'Variable': list(solution['variables'].keys()),
                'Valor': list(solution['variables'].values())
            }
            df_variables = pd.DataFrame(variables_data)
            df_variables.to_excel(writer, sheet_name='Variables', index=False)
        
        # Hoja 3: Iteraciones (si están disponibles)
        if 'iterations' in solution:
            iterations_summary = []
            for it in solution['iterations']:
                iterations_summary.append({
                    'Iteración': it['iteration'],
                    'Descripción': it['description'],
                    'Valor Objetivo': it['tableau'][-1, -1] if 'tableau' in it else 'N/A'
                })
            
            if iterations_summary:
                df_iterations = pd.DataFrame(iterations_summary)
                df_iterations.to_excel(writer, sheet_name='Iteraciones', index=False)
        
        # Hoja 4: Puntos esquina (si es método gráfico)
        if 'corner_points' in solution and solution['corner_points']:
            corner_data = {
                'Punto': [f"P{i+1}" for i in range(len(solution['corner_points']))],
                'x1': [p[0] for p in solution['corner_points']],
                'x2': [p[1] for p in solution['corner_points']]
            }
            if 'evaluations' in solution:
                corner_data['Valor Z'] = [e['value'] for e in solution['evaluations']]
            
            df_corners = pd.DataFrame(corner_data)
            df_corners.to_excel(writer, sheet_name='Puntos Esquina', index=False)
        
        # Formatear hojas
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('A:A', 25)
            worksheet.set_column('B:Z', 15)
    
    output.seek(0)
    return output.getvalue()


def export_to_json(solution: Dict, filename: str = None) -> str:
    """
    Exporta solución a JSON
    
    Returns:
        str: Contenido JSON
    """
    
    # Convertir arrays numpy a listas
    solution_serializable = _make_json_serializable(solution)
    
    return json.dumps(solution_serializable, indent=2, ensure_ascii=False)


def export_to_csv(solution: Dict, filename: str = None) -> bytes:
    """
    Exporta solución a CSV
    
    Returns:
        bytes: Contenido del archivo CSV
    """
    
    output = io.StringIO()
    
    # Escribir resumen
    output.write("=== RESUMEN DE LA SOLUCIÓN ===\n")
    output.write(f"Estado,{solution.get('status', 'N/A')}\n")
    output.write(f"Valor Óptimo,{solution.get('optimal_value', 'N/A')}\n")
    output.write(f"Método,{solution.get('method', 'N/A')}\n")
    output.write(f"Iteraciones,{solution.get('num_iterations', 'N/A')}\n")
    output.write(f"Fecha,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output.write("\n")
    
    # Variables
    if 'variables' in solution:
        output.write("=== VARIABLES DE DECISIÓN ===\n")
        output.write("Variable,Valor\n")
        for var, val in solution['variables'].items():
            output.write(f"{var},{val}\n")
        output.write("\n")
    
    # Convertir a bytes
    return output.getvalue().encode('utf-8')


def export_report_text(solution: Dict, problem_description: str = None) -> str:
    """
    Genera reporte de texto completo
    
    Returns:
        str: Reporte en texto plano
    """
    
    report = []
    report.append("=" * 80)
    report.append("REPORTE DE INVESTIGACIÓN DE OPERACIONES")
    report.append("=" * 80)
    report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    if problem_description:
        report.append("DESCRIPCIÓN DEL PROBLEMA:")
        report.append("-" * 80)
        report.append(problem_description)
        report.append("")
    
    report.append("RESUMEN DE LA SOLUCIÓN:")
    report.append("-" * 80)
    report.append(f"Estado: {solution.get('status', 'N/A')}")
    report.append(f"Método: {solution.get('method', 'N/A')}")
    report.append(f"Valor Óptimo: {solution.get('optimal_value', 'N/A'):.6f}")
    report.append(f"Número de Iteraciones: {solution.get('num_iterations', 'N/A')}")
    report.append("")
    
    if 'variables' in solution:
        report.append("VARIABLES DE DECISIÓN:")
        report.append("-" * 80)
        for var, val in solution['variables'].items():
            report.append(f"  {var:15s} = {val:12.6f}")
        report.append("")
    
    if 'corner_points' in solution and solution['corner_points']:
        report.append("PUNTOS ESQUINA EVALUADOS:")
        report.append("-" * 80)
        for i, point in enumerate(solution['corner_points']):
            value = solution['evaluations'][i]['value'] if 'evaluations' in solution else 'N/A'
            report.append(f"  P{i+1}: ({point[0]:.4f}, {point[1]:.4f}) → Z = {value:.4f}")
        report.append("")
    
    if 'iterations' in solution and len(solution['iterations']) > 0:
        report.append("HISTORIAL DE ITERACIONES:")
        report.append("-" * 80)
        for it in solution['iterations']:
            report.append(f"  Iteración {it['iteration']}: {it['description']}")
            if 'tableau' in it:
                z_value = it['tableau'][-1, -1]
                report.append(f"    Valor Objetivo: {z_value:.6f}")
        report.append("")
    
    if 'message' in solution:
        report.append("MENSAJE:")
        report.append("-" * 80)
        report.append(solution['message'])
        report.append("")
    
    report.append("=" * 80)
    report.append("FIN DEL REPORTE")
    report.append("=" * 80)
    
    return "\n".join(report)


def _make_json_serializable(obj):
    """Convierte objetos no serializables a JSON"""
    
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


def create_latex_table(tableau: np.ndarray, var_names: List[str], 
                      basis: List[int]) -> str:
    """
    Genera código LaTeX para el tableau del Simplex
    
    Returns:
        str: Código LaTeX
    """
    
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{|c|" + "c|" * len(var_names) + "c|}")
    latex.append("\\hline")
    
    # Encabezado
    header = "Base & " + " & ".join(var_names) + " & RHS \\\\"
    latex.append(header)
    latex.append("\\hline")
    
    # Filas
    m, n = tableau.shape
    for i in range(m - 1):
        row_str = f"{var_names[basis[i]]} & "
        row_str += " & ".join([f"{tableau[i, j]:.4f}" for j in range(n - 1)])
        row_str += f" & {tableau[i, -1]:.4f} \\\\"
        latex.append(row_str)
    
    latex.append("\\hline")
    
    # Fila Z
    z_row = "Z & "
    z_row += " & ".join([f"{tableau[-1, j]:.4f}" for j in range(n - 1)])
    z_row += f" & {tableau[-1, -1]:.4f} \\\\"
    latex.append(z_row)
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Tableau del Método Simplex}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def save_session(problem_data: Dict, solution: Dict, filename: str = None) -> bytes:
    """
    Guarda problema y solución completa para cargar después
    
    Returns:
        bytes: Datos de sesión serializados
    """
    
    session_data = {
        'timestamp': datetime.now().isoformat(),
        'problem': _make_json_serializable(problem_data),
        'solution': _make_json_serializable(solution)
    }
    
    return json.dumps(session_data, indent=2).encode('utf-8')


def load_session(session_bytes: bytes) -> Dict:
    """
    Carga sesión guardada
    
    Returns:
        Dict: Problema y solución
    """
    
    session_data = json.loads(session_bytes.decode('utf-8'))
    return session_data
