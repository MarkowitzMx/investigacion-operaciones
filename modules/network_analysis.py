"""
Módulo de Análisis de Redes
Incluye: Transporte, Asignación, Camino más corto, Flujo máximo, PERT-CPM
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment


class TransportationProblem:
    """Problema de Transporte"""
    
    def __init__(self):
        self.solution = None
        
    def solve(self, supply: np.ndarray, demand: np.ndarray, 
              costs: np.ndarray, method: str = 'nw_corner') -> Dict:
        """
        Resuelve problema de transporte
        
        Args:
            supply: Oferta de cada origen
            demand: Demanda de cada destino
            costs: Matriz de costos (m x n)
            method: 'nw_corner' (Esquina Noroeste), 'vogel' (Vogel)
        
        Returns:
            Dict con solución
        """
        
        # Verificar balance
        total_supply = np.sum(supply)
        total_demand = np.sum(demand)
        
        if not np.isclose(total_supply, total_demand):
            return {
                'status': 'unbalanced',
                'message': f'No balanceado: Oferta={total_supply}, Demanda={total_demand}',
                'total_supply': total_supply,
                'total_demand': total_demand
            }
        
        if method == 'nw_corner':
            allocation, steps = self._northwest_corner(supply.copy(), demand.copy(), costs)
        elif method == 'vogel':
            allocation, steps = self._vogel_method(supply.copy(), demand.copy(), costs)
        else:
            return {'status': 'error', 'message': f'Método desconocido: {method}'}
        
        # Calcular costo total
        total_cost = np.sum(allocation * costs)
        
        return {
            'status': 'optimal',
            'allocation': allocation,
            'total_cost': total_cost,
            'method': method,
            'steps': steps
        }
    
    def _northwest_corner(self, supply: np.ndarray, demand: np.ndarray,
                         costs: np.ndarray) -> Tuple[np.ndarray, List]:
        """Método de la Esquina Noroeste"""
        
        m, n = costs.shape
        allocation = np.zeros((m, n))
        steps = []
        
        i, j = 0, 0
        
        while i < m and j < n:
            # Asignar el mínimo entre oferta y demanda
            quantity = min(supply[i], demand[j])
            allocation[i, j] = quantity
            
            steps.append({
                'from': i,
                'to': j,
                'quantity': quantity,
                'cost': costs[i, j]
            })
            
            supply[i] -= quantity
            demand[j] -= quantity
            
            # Mover a la siguiente celda
            if supply[i] == 0:
                i += 1
            if demand[j] == 0:
                j += 1
        
        return allocation, steps
    
    def _vogel_method(self, supply: np.ndarray, demand: np.ndarray,
                     costs: np.ndarray) -> Tuple[np.ndarray, List]:
        """Método de Aproximación de Vogel"""
        
        m, n = costs.shape
        allocation = np.zeros((m, n))
        steps = []
        
        supply_active = supply.copy()
        demand_active = demand.copy()
        rows_active = set(range(m))
        cols_active = set(range(n))
        
        while rows_active and cols_active:
            # Calcular penalizaciones
            row_penalties = []
            for i in rows_active:
                active_cols = sorted(cols_active)
                if len(active_cols) >= 2:
                    row_costs = sorted([costs[i, j] for j in active_cols])
                    penalty = row_costs[1] - row_costs[0]
                else:
                    penalty = 0
                row_penalties.append((penalty, i))
            
            col_penalties = []
            for j in cols_active:
                active_rows = sorted(rows_active)
                if len(active_rows) >= 2:
                    col_costs = sorted([costs[i, j] for i in active_rows])
                    penalty = col_costs[1] - col_costs[0]
                else:
                    penalty = 0
                col_penalties.append((penalty, j))
            
            # Seleccionar máxima penalización
            max_row_penalty = max(row_penalties, key=lambda x: x[0]) if row_penalties else (-1, -1)
            max_col_penalty = max(col_penalties, key=lambda x: x[0]) if col_penalties else (-1, -1)
            
            if max_row_penalty[0] >= max_col_penalty[0]:
                # Seleccionar fila con mayor penalización
                i = max_row_penalty[1]
                # Encontrar mínimo costo en la fila
                j = min(cols_active, key=lambda j: costs[i, j])
            else:
                # Seleccionar columna con mayor penalización
                j = max_col_penalty[1]
                # Encontrar mínimo costo en la columna
                i = min(rows_active, key=lambda i: costs[i, j])
            
            # Asignar
            quantity = min(supply_active[i], demand_active[j])
            allocation[i, j] = quantity
            
            steps.append({
                'from': i,
                'to': j,
                'quantity': quantity,
                'cost': costs[i, j],
                'penalty_selected': max(max_row_penalty[0], max_col_penalty[0])
            })
            
            supply_active[i] -= quantity
            demand_active[j] -= quantity
            
            if supply_active[i] == 0:
                rows_active.remove(i)
            if demand_active[j] == 0:
                cols_active.remove(j)
        
        return allocation, steps


class AssignmentProblem:
    """Problema de Asignación"""
    
    def solve(self, cost_matrix: np.ndarray, maximize: bool = False) -> Dict:
        """
        Resuelve problema de asignación usando Algoritmo Húngaro
        
        Args:
            cost_matrix: Matriz de costos (n x n)
            maximize: True para maximizar
        
        Returns:
            Dict con asignación óptima
        """
        
        if maximize:
            # Convertir a minimización
            matrix = cost_matrix.max() - cost_matrix
        else:
            matrix = cost_matrix.copy()
        
        # Usar scipy para resolver
        row_ind, col_ind = linear_sum_assignment(matrix)
        
        total_cost = cost_matrix[row_ind, col_ind].sum()
        
        assignments = []
        for i, j in zip(row_ind, col_ind):
            assignments.append({
                'from': int(i),
                'to': int(j),
                'cost': float(cost_matrix[i, j])
            })
        
        return {
            'status': 'optimal',
            'assignments': assignments,
            'total_cost': total_cost,
            'row_indices': row_ind,
            'col_indices': col_ind
        }


class NetworkFlowProblems:
    """Problemas de Flujo en Redes"""
    
    def shortest_path(self, graph: Dict, source: str, target: str) -> Dict:
        """
        Encuentra camino más corto
        
        Args:
            graph: {node: {neighbor: weight}}
            source: Nodo origen
            target: Nodo destino
        
        Returns:
            Dict con camino y distancia
        """
        
        G = nx.DiGraph()
        for node, edges in graph.items():
            for neighbor, weight in edges.items():
                G.add_edge(node, neighbor, weight=weight)
        
        try:
            path = nx.shortest_path(G, source, target, weight='weight')
            length = nx.shortest_path_length(G, source, target, weight='weight')
            
            return {
                'status': 'found',
                'path': path,
                'length': length
            }
        except nx.NetworkXNoPath:
            return {
                'status': 'no_path',
                'message': f'No existe camino de {source} a {target}'
            }
    
    def maximum_flow(self, graph: Dict, source: str, sink: str) -> Dict:
        """
        Encuentra flujo máximo
        
        Args:
            graph: {node: {neighbor: capacity}}
            source: Nodo fuente
            sink: Nodo sumidero
        
        Returns:
            Dict con flujo máximo
        """
        
        G = nx.DiGraph()
        for node, edges in graph.items():
            for neighbor, capacity in edges.items():
                G.add_edge(node, neighbor, capacity=capacity)
        
        flow_value, flow_dict = nx.maximum_flow(G, source, sink)
        
        return {
            'status': 'optimal',
            'max_flow': flow_value,
            'flow_dict': flow_dict
        }
    
    def minimum_spanning_tree(self, graph: Dict) -> Dict:
        """
        Encuentra árbol expandido mínimo
        
        Args:
            graph: {node: {neighbor: weight}}
        
        Returns:
            Dict con árbol
        """
        
        G = nx.Graph()
        for node, edges in graph.items():
            for neighbor, weight in edges.items():
                G.add_edge(node, neighbor, weight=weight)
        
        mst = nx.minimum_spanning_tree(G)
        
        edges = []
        total_weight = 0
        for u, v, data in mst.edges(data=True):
            edges.append({
                'from': u,
                'to': v,
                'weight': data['weight']
            })
            total_weight += data['weight']
        
        return {
            'status': 'optimal',
            'edges': edges,
            'total_weight': total_weight
        }


class PertCpm:
    """Análisis PERT-CPM"""
    
    def __init__(self):
        self.critical_path = None
        self.project_duration = None
        
    def solve(self, activities: List[Dict]) -> Dict:
        """
        Resuelve análisis PERT-CPM
        
        Args:
            activities: Lista de actividades con:
                - id: Identificador
                - duration: Duración
                - predecessors: Lista de predecesores
        
        Returns:
            Dict con ruta crítica y tiempos
        """
        
        # Crear grafo
        G = nx.DiGraph()
        
        for activity in activities:
            G.add_node(activity['id'], 
                      duration=activity['duration'])
            
            for pred in activity.get('predecessors', []):
                G.add_edge(pred, activity['id'])
        
        # Calcular tiempos más tempranos (ES, EF)
        es_times = {}
        ef_times = {}
        
        for node in nx.topological_sort(G):
            duration = G.nodes[node]['duration']
            
            # Tiempo más temprano de inicio
            if not list(G.predecessors(node)):
                es_times[node] = 0
            else:
                es_times[node] = max(ef_times[pred] for pred in G.predecessors(node))
            
            # Tiempo más temprano de fin
            ef_times[node] = es_times[node] + duration
        
        # Duración del proyecto
        self.project_duration = max(ef_times.values())
        
        # Calcular tiempos más tardíos (LS, LF)
        ls_times = {}
        lf_times = {}
        
        for node in reversed(list(nx.topological_sort(G))):
            duration = G.nodes[node]['duration']
            
            # Tiempo más tardío de fin
            if not list(G.successors(node)):
                lf_times[node] = self.project_duration
            else:
                lf_times[node] = min(ls_times[succ] for succ in G.successors(node))
            
            # Tiempo más tardío de inicio
            ls_times[node] = lf_times[node] - duration
        
        # Calcular holguras
        slacks = {node: ls_times[node] - es_times[node] for node in G.nodes()}
        
        # Identificar ruta crítica
        critical_activities = [node for node, slack in slacks.items() if slack == 0]
        self.critical_path = critical_activities
        
        # Preparar resultados
        activity_details = []
        for node in G.nodes():
            activity_details.append({
                'activity': node,
                'duration': G.nodes[node]['duration'],
                'ES': es_times[node],
                'EF': ef_times[node],
                'LS': ls_times[node],
                'LF': lf_times[node],
                'slack': slacks[node],
                'is_critical': slacks[node] == 0
            })
        
        return {
            'status': 'success',
            'project_duration': self.project_duration,
            'critical_path': self.critical_path,
            'activity_details': activity_details
        }
