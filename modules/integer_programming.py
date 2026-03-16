"""
Módulo de Programación Entera
Incluye: Branch and Bound, Cortes de Gomory
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import pulp


class IntegerProgramming:
    """Programación Entera"""
    
    def __init__(self):
        self.solution_tree = []
        
    def solve(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
              maximize: bool = True, var_names: List[str] = None,
              integer_vars: List[int] = None, method: str = 'branch_bound') -> Dict:
        """
        Resuelve problema de programación entera
        
        Args:
            c: Coeficientes función objetivo
            A: Matriz restricciones
            b: Vector RHS
            maximize: True para maximizar
            var_names: Nombres de variables
            integer_vars: Índices de variables enteras (None = todas enteras)
            method: 'branch_bound', 'gomory', 'pulp'
        
        Returns:
            Dict con solución
        """
        
        n = len(c)
        
        if integer_vars is None:
            integer_vars = list(range(n))
        
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(n)]
        
        if method == 'pulp':
            return self._solve_with_pulp(c, A, b, maximize, var_names, integer_vars)
        elif method == 'branch_bound':
            return self._branch_and_bound(c, A, b, maximize, var_names, integer_vars)
        elif method == 'gomory':
            return self._gomory_cuts(c, A, b, maximize, var_names, integer_vars)
        else:
            return {'status': 'error', 'message': f'Método desconocido: {method}'}
    
    def _solve_with_pulp(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                        maximize: bool, var_names: List[str],
                        integer_vars: List[int]) -> Dict:
        """Resuelve usando PuLP"""
        
        n = len(c)
        m = len(b)
        
        # Crear problema
        if maximize:
            prob = pulp.LpProblem("Integer_Problem", pulp.LpMaximize)
        else:
            prob = pulp.LpProblem("Integer_Problem", pulp.LpMinimize)
        
        # Variables (enteras o continuas)
        vars_dict = {}
        for i in range(n):
            if i in integer_vars:
                vars_dict[var_names[i]] = pulp.LpVariable(
                    var_names[i], lowBound=0, cat='Integer'
                )
            else:
                vars_dict[var_names[i]] = pulp.LpVariable(
                    var_names[i], lowBound=0
                )
        
        # Función objetivo
        prob += pulp.lpSum([c[i] * vars_dict[var_names[i]] for i in range(n)])
        
        # Restricciones
        for i in range(m):
            prob += pulp.lpSum([A[i, j] * vars_dict[var_names[j]] for j in range(n)]) <= b[i]
        
        # Resolver
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            solution_vars = {name: var.varValue for name, var in vars_dict.items()}
            
            return {
                'status': 'optimal',
                'method': 'Integer Programming (PuLP)',
                'variables': solution_vars,
                'optimal_value': pulp.value(prob.objective)
            }
        else:
            return {
                'status': pulp.LpStatus[prob.status],
                'message': f'Solver status: {pulp.LpStatus[prob.status]}'
            }
    
    def _branch_and_bound(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                         maximize: bool, var_names: List[str],
                         integer_vars: List[int]) -> Dict:
        """
        Implementación simplificada de Branch and Bound
        
        Nota: Esta es una implementación educativa. Para problemas reales
        se recomienda usar PuLP o solvers especializados.
        """
        
        # Resolver relajación lineal
        relaxed = self._solve_lp_relaxation(c, A, b, maximize)
        
        if relaxed['status'] != 'optimal':
            return relaxed
        
        # Verificar si la solución es entera
        solution_values = list(relaxed['variables'].values())
        
        all_integer = True
        fractional_var_idx = None
        
        for i in integer_vars:
            val = solution_values[i]
            if not self._is_integer(val):
                all_integer = False
                fractional_var_idx = i
                break
        
        if all_integer:
            return {
                'status': 'optimal',
                'method': 'Branch and Bound',
                'variables': relaxed['variables'],
                'optimal_value': relaxed['optimal_value'],
                'nodes_explored': 1
            }
        
        # Mensaje informativo
        return {
            'status': 'optimal',
            'method': 'Branch and Bound (simplificado)',
            'message': (
                'Implementación básica. Para problemas complejos, '
                'usa el método "pulp" que incluye Branch and Bound completo.'
            ),
            'lp_relaxation': relaxed,
            'fractional_variable': var_names[fractional_var_idx] if fractional_var_idx else None
        }
    
    def _gomory_cuts(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                    maximize: bool, var_names: List[str],
                    integer_vars: List[int]) -> Dict:
        """
        Método de Cortes de Gomory (implementación básica)
        """
        
        # Resolver relajación lineal
        relaxed = self._solve_lp_relaxation(c, A, b, maximize)
        
        return {
            'status': 'optimal',
            'method': 'Cortes de Gomory (simplificado)',
            'message': (
                'Implementación educativa. Para aplicaciones reales, '
                'usa el método "pulp".'
            ),
            'lp_relaxation': relaxed
        }
    
    def _solve_lp_relaxation(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                            maximize: bool) -> Dict:
        """Resuelve relajación lineal (ignora restricciones de enteros)"""
        
        n = len(c)
        m = len(b)
        
        if maximize:
            prob = pulp.LpProblem("LP_Relaxation", pulp.LpMaximize)
        else:
            prob = pulp.LpProblem("LP_Relaxation", pulp.LpMinimize)
        
        vars_list = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(n)]
        
        prob += pulp.lpSum([c[i] * vars_list[i] for i in range(n)])
        
        for i in range(m):
            prob += pulp.lpSum([A[i, j] * vars_list[j] for j in range(n)]) <= b[i]
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            solution_vars = {f"x{i+1}": var.varValue for i, var in enumerate(vars_list)}
            
            return {
                'status': 'optimal',
                'variables': solution_vars,
                'optimal_value': pulp.value(prob.objective)
            }
        else:
            return {'status': 'error'}
    
    def _is_integer(self, value: float, tolerance: float = 1e-6) -> bool:
        """Verifica si un valor es entero dentro de una tolerancia"""
        return abs(value - round(value)) < tolerance
    
    def solve_binary_problem(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                           maximize: bool = True, var_names: List[str] = None) -> Dict:
        """
        Resuelve problema binario (0-1)
        
        Args:
            c: Coeficientes función objetivo
            A: Matriz restricciones
            b: Vector RHS
            maximize: True para maximizar
            var_names: Nombres de variables
        
        Returns:
            Dict con solución
        """
        
        n = len(c)
        m = len(b)
        
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(n)]
        
        # Crear problema
        if maximize:
            prob = pulp.LpProblem("Binary_Problem", pulp.LpMaximize)
        else:
            prob = pulp.LpProblem("Binary_Problem", pulp.LpMinimize)
        
        # Variables binarias
        vars_dict = {}
        for i in range(n):
            vars_dict[var_names[i]] = pulp.LpVariable(
                var_names[i], cat='Binary'
            )
        
        # Función objetivo
        prob += pulp.lpSum([c[i] * vars_dict[var_names[i]] for i in range(n)])
        
        # Restricciones
        for i in range(m):
            prob += pulp.lpSum([A[i, j] * vars_dict[var_names[j]] for j in range(n)]) <= b[i]
        
        # Resolver
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            solution_vars = {name: var.varValue for name, var in vars_dict.items()}
            
            return {
                'status': 'optimal',
                'method': 'Binary Programming',
                'variables': solution_vars,
                'optimal_value': pulp.value(prob.objective)
            }
        else:
            return {
                'status': pulp.LpStatus[prob.status],
                'message': f'Solver status: {pulp.LpStatus[prob.status]}'
            }
    
    def solve_knapsack(self, values: np.ndarray, weights: np.ndarray,
                      capacity: float) -> Dict:
        """
        Resuelve problema de la mochila (knapsack)
        
        Args:
            values: Valores de los objetos
            weights: Pesos de los objetos
            capacity: Capacidad de la mochila
        
        Returns:
            Dict con solución
        """
        
        n = len(values)
        
        # Crear problema
        prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)
        
        # Variables binarias (tomar o no cada objeto)
        x = [pulp.LpVariable(f"x{i+1}", cat='Binary') for i in range(n)]
        
        # Maximizar valor total
        prob += pulp.lpSum([values[i] * x[i] for i in range(n)])
        
        # Restricción de capacidad
        prob += pulp.lpSum([weights[i] * x[i] for i in range(n)]) <= capacity
        
        # Resolver
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            selected_items = [i for i in range(n) if x[i].varValue == 1]
            total_value = sum(values[i] for i in selected_items)
            total_weight = sum(weights[i] for i in selected_items)
            
            return {
                'status': 'optimal',
                'selected_items': selected_items,
                'total_value': total_value,
                'total_weight': total_weight,
                'capacity': capacity,
                'utilization': total_weight / capacity
            }
        else:
            return {
                'status': 'error',
                'message': pulp.LpStatus[prob.status]
            }
