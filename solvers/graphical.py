"""
Solver del Método Gráfico para problemas de 2 variables
"""
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from itertools import combinations

class GraphicalSolver:
    def __init__(self):
        self.feasible_region = None
        self.corner_points = []
        self.optimal_point = None
        
    def solve(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
              maximize: bool = True, var_names: List[str] = None,
              constraint_types: List[str] = None) -> Dict:
        """
        Resuelve problema de 2 variables usando método gráfico
        
        Args:
            c: Coeficientes función objetivo [c1, c2]
            A: Matriz de restricciones (m x 2)
            b: Vector lado derecho
            maximize: True para maximizar
            var_names: Nombres de variables
            constraint_types: Tipos de restricción
        """
        
        if len(c) != 2:
            return {
                'status': 'error',
                'message': 'El método gráfico solo funciona con 2 variables'
            }
        
        if var_names is None:
            var_names = ['x1', 'x2']
        
        if constraint_types is None:
            constraint_types = ['<='] * len(b)
        
        # Encontrar puntos esquina de la región factible
        self.corner_points = self._find_corner_points(A, b, constraint_types)
        
        if not self.corner_points:
            return {
                'status': 'infeasible',
                'message': 'No existe región factible',
                'corner_points': []
            }
        
        # Evaluar función objetivo en cada punto esquina
        evaluations = []
        for point in self.corner_points:
            value = np.dot(c, point)
            evaluations.append({
                'point': point,
                'value': value
            })
        
        # Encontrar óptimo
        if maximize:
            optimal = max(evaluations, key=lambda x: x['value'])
        else:
            optimal = min(evaluations, key=lambda x: x['value'])
        
        self.optimal_point = optimal['point']
        
        # Crear gráfico
        fig = self._create_plot(A, b, c, constraint_types, var_names, maximize)
        
        return {
            'status': 'optimal',
            'variables': {
                var_names[0]: self.optimal_point[0],
                var_names[1]: self.optimal_point[1]
            },
            'optimal_value': optimal['value'],
            'corner_points': self.corner_points,
            'evaluations': evaluations,
            'figure': fig
        }
    
    def _find_corner_points(self, A: np.ndarray, b: np.ndarray,
                           constraint_types: List[str]) -> List[np.ndarray]:
        """Encuentra puntos esquina de la región factible"""
        
        corner_points = []
        m = len(b)
        
        # Agregar restricciones de no negatividad
        A_extended = np.vstack([A, np.eye(2)])
        b_extended = np.hstack([b, [0, 0]])
        constraint_types_extended = constraint_types + ['>=', '>=']
        
        # Probar todas las combinaciones de 2 restricciones
        for i, j in combinations(range(len(b_extended)), 2):
            try:
                # Resolver sistema de 2x2
                A_sys = A_extended[[i, j]]
                b_sys = b_extended[[i, j]]
                
                if np.linalg.det(A_sys) == 0:
                    continue
                
                point = np.linalg.solve(A_sys, b_sys)
                
                # Verificar si el punto satisface todas las restricciones
                if self._is_feasible(point, A_extended, b_extended, constraint_types_extended):
                    # Evitar duplicados
                    is_duplicate = False
                    for existing_point in corner_points:
                        if np.allclose(point, existing_point, atol=1e-6):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        corner_points.append(point)
                        
            except np.linalg.LinAlgError:
                continue
        
        return corner_points
    
    def _is_feasible(self, point: np.ndarray, A: np.ndarray, b: np.ndarray,
                    constraint_types: List[str]) -> bool:
        """Verifica si un punto satisface todas las restricciones"""
        
        for i in range(len(b)):
            value = np.dot(A[i], point)
            
            if constraint_types[i] == '<=':
                if value > b[i] + 1e-6:
                    return False
            elif constraint_types[i] == '>=':
                if value < b[i] - 1e-6:
                    return False
            elif constraint_types[i] == '=':
                if abs(value - b[i]) > 1e-6:
                    return False
        
        return True
    
    def _create_plot(self, A: np.ndarray, b: np.ndarray, c: np.ndarray,
                    constraint_types: List[str], var_names: List[str],
                    maximize: bool) -> go.Figure:
        """Crea gráfico interactivo de la región factible"""
        
        fig = go.Figure()
        
        # Determinar límites del gráfico
        if self.corner_points:
            points_array = np.array(self.corner_points)
            x_min, x_max = points_array[:, 0].min(), points_array[:, 0].max()
            y_min, y_max = points_array[:, 1].min(), points_array[:, 1].max()
            
            margin = 0.2 * max(x_max - x_min, y_max - y_min, 1)
            x_range = [max(0, x_min - margin), x_max + margin]
            y_range = [max(0, y_min - margin), y_max + margin]
        else:
            x_range = [0, 10]
            y_range = [0, 10]
        
        # Graficar restricciones
        x_plot = np.linspace(x_range[0], x_range[1], 300)
        
        for i, (a_row, b_val) in enumerate(zip(A, b)):
            if a_row[1] != 0:
                y_plot = (b_val - a_row[0] * x_plot) / a_row[1]
                
                fig.add_trace(go.Scatter(
                    x=x_plot, y=y_plot,
                    mode='lines',
                    name=f'Restricción {i+1}',
                    line=dict(width=2)
                ))
        
        # Graficar región factible (polígono)
        if self.corner_points:
            # Ordenar puntos para formar polígono
            points = self._order_points_clockwise(self.corner_points)
            points_array = np.array(points)
            
            fig.add_trace(go.Scatter(
                x=np.append(points_array[:, 0], points_array[0, 0]),
                y=np.append(points_array[:, 1], points_array[0, 1]),
                fill='toself',
                fillcolor='rgba(0, 176, 246, 0.2)',
                line=dict(color='rgba(0, 176, 246, 0.8)', width=2),
                name='Región Factible'
            ))
        
        # Marcar puntos esquina
        if self.corner_points:
            points_array = np.array(self.corner_points)
            fig.add_trace(go.Scatter(
                x=points_array[:, 0],
                y=points_array[:, 1],
                mode='markers+text',
                marker=dict(size=10, color='red'),
                text=[f'({p[0]:.2f}, {p[1]:.2f})' for p in self.corner_points],
                textposition='top center',
                name='Puntos Esquina'
            ))
        
        # Marcar punto óptimo
        if self.optimal_point is not None:
            fig.add_trace(go.Scatter(
                x=[self.optimal_point[0]],
                y=[self.optimal_point[1]],
                mode='markers+text',
                marker=dict(size=15, color='green', symbol='star'),
                text=[f'Óptimo: ({self.optimal_point[0]:.2f}, {self.optimal_point[1]:.2f})'],
                textposition='bottom center',
                name='Solución Óptima'
            ))
        
        # Líneas de isovalor de función objetivo
        if self.optimal_point is not None:
            z_opt = np.dot(c, self.optimal_point)
            
            # Dibujar 3 líneas de isovalor
            for factor in [0.5, 0.75, 1.0]:
                z_val = z_opt * factor
                if c[1] != 0:
                    y_iso = (z_val - c[0] * x_plot) / c[1]
                    
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_iso,
                        mode='lines',
                        line=dict(dash='dash', width=1, color='green'),
                        name=f'Z = {z_val:.2f}',
                        showlegend=(factor == 1.0)
                    ))
        
        fig.update_layout(
            title=f'Método Gráfico - {"Maximizar" if maximize else "Minimizar"} Z = {c[0]}{var_names[0]} + {c[1]}{var_names[1]}',
            xaxis_title=var_names[0],
            yaxis_title=var_names[1],
            xaxis=dict(range=x_range, constrain='domain'),
            yaxis=dict(range=y_range, scaleanchor='x', scaleratio=1),
            hovermode='closest',
            width=800,
            height=700
        )
        
        return fig
    
    def _order_points_clockwise(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """Ordena puntos en sentido horario para formar polígono"""
        if len(points) < 3:
            return points
        
        # Calcular centroide
        centroid = np.mean(points, axis=0)
        
        # Calcular ángulos respecto al centroide
        def angle_from_centroid(point):
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
        
        # Ordenar por ángulo
        sorted_points = sorted(points, key=angle_from_centroid)
        
        return sorted_points
