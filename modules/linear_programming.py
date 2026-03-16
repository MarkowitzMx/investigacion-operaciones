"""
Módulo de Programación Lineal - Integra todos los métodos
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os
 
# Path dinámico — funciona tanto local como en Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
 
from solvers.simplex import SimplexSolver, TwoPhaseSimplexSolver
from solvers.graphical import GraphicalSolver
import pulp
 
 
class LinearProgrammingModule:
    """Módulo principal para resolver problemas de Programación Lineal"""
 
    def __init__(self):
        self.problem_data = None
        self.solution = None
 
    def solve(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
              method: str = 'simplex', maximize: bool = True,
              var_names: List[str] = None,
              constraint_types: List[str] = None) -> Dict:
 
        n = len(c)
        m = len(b)
 
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(n)]
        if constraint_types is None:
            constraint_types = ['<='] * m
 
        self.problem_data = {
            'c': c.copy(),
            'A': A.copy(),
            'b': b.copy(),
            'method': method,
            'maximize': maximize,
            'var_names': var_names,
            'constraint_types': constraint_types
        }
 
        if method == 'simplex':
            solver = SimplexSolver()
            self.solution = solver.solve(c, A, b, maximize, var_names, constraint_types)
            self.solution['method'] = 'Método Simplex'
 
        elif method == 'graphical':
            if len(c) != 2:
                return {
                    'status': 'error',
                    'message': 'El método gráfico requiere exactamente 2 variables'
                }
            solver = GraphicalSolver()
            self.solution = solver.solve(c, A, b, maximize, var_names, constraint_types)
            self.solution['method'] = 'Método Gráfico'
 
        elif method == 'two_phase':
            solver = TwoPhaseSimplexSolver()
            # Pasar maximize=False internamente; el solver maneja el signo
            self.solution = solver.solve(
                c.copy(), A.copy(), b.copy(),
                maximize, var_names, constraint_types
            )
            self.solution['method'] = 'Método de las Dos Fases'
 
        elif method == 'pulp':
            self.solution = self._solve_with_pulp(c, A, b, maximize, var_names, constraint_types)
            self.solution['method'] = 'PuLP (Optimizador)'
 
        else:
            return {
                'status': 'error',
                'message': f'Método desconocido: {method}'
            }
 
        return self.solution
 
    def _solve_with_pulp(self, c, A, b, maximize, var_names, constraint_types):
        n = len(c)
        m = len(b)
 
        if maximize:
            prob = pulp.LpProblem("Problema_IO", pulp.LpMaximize)
        else:
            prob = pulp.LpProblem("Problema_IO", pulp.LpMinimize)
 
        vars_dict = {name: pulp.LpVariable(name, lowBound=0) for name in var_names}
        prob += pulp.lpSum([c[i] * vars_dict[var_names[i]] for i in range(n)])
 
        for i in range(m):
            expr = pulp.lpSum([A[i, j] * vars_dict[var_names[j]] for j in range(n)])
            if constraint_types[i] == '<=':
                prob += expr <= b[i], f"R{i+1}"
            elif constraint_types[i] == '>=':
                prob += expr >= b[i], f"R{i+1}"
            elif constraint_types[i] == '=':
                prob += expr == b[i], f"R{i+1}"
 
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
 
        if prob.status == pulp.LpStatusOptimal:
            return {
                'status': 'optimal',
                'variables': {name: var.varValue for name, var in vars_dict.items()},
                'optimal_value': pulp.value(prob.objective),
                'solver_status': pulp.LpStatus[prob.status]
            }
        else:
            return {
                'status': 'error',
                'message': f'Estado del solver: {pulp.LpStatus[prob.status]}'
            }
 
    def compare_methods(self, c, A, b, maximize=True, var_names=None, constraint_types=None):
        methods_to_compare = ['simplex', 'pulp']
        if len(c) == 2:
            methods_to_compare.append('graphical')
 
        results = {}
        for method in methods_to_compare:
            try:
                import time
                start = time.time()
                result = self.solve(c, A, b, method, maximize, var_names, constraint_types)
                result['time'] = time.time() - start
                results[method] = result
            except Exception as e:
                results[method] = {'status': 'error', 'message': str(e)}
 
        return results
 
    def get_problem_formulation(self) -> str:
        if self.problem_data is None:
            return "No hay problema definido"
 
        c  = self.problem_data.get('c')
        A  = self.problem_data.get('A')
        b  = self.problem_data.get('b')
 
        if c is None or A is None or b is None:
            return "Datos del problema incompletos"
 
        n = len(c)
        m = len(b)
 
        maximize         = self.problem_data.get('maximize', True)
        var_names        = self.problem_data.get('var_names') or [f"x{i+1}" for i in range(n)]
        constraint_types = self.problem_data.get('constraint_types') or ['<='] * m
 
        formulation = []
        obj_type = "Maximizar" if maximize else "Minimizar"
        obj_terms = []
        for i, (coef, var) in enumerate(zip(c, var_names)):
            if i == 0:
                obj_terms.append(f"{coef}{var}")
            else:
                sign = '+' if coef >= 0 else ''
                obj_terms.append(f"{sign}{coef}{var}")
 
        formulation.append(f"{obj_type} Z = {' '.join(obj_terms)}")
        formulation.append("")
        formulation.append("Sujeto a:")
 
        for i in range(m):
            terms = []
            for j, (coef, var) in enumerate(zip(A[i], var_names)):
                if j == 0:
                    terms.append(f"{coef}{var}")
                else:
                    sign = '+' if coef >= 0 else ''
                    terms.append(f"{sign}{coef}{var}")
            formulation.append(f"  {' '.join(terms)} {constraint_types[i]} {b[i]}")
 
        formulation.append("")
        formulation.append(f"  {', '.join(var_names)} >= 0")
 
        return "\n".join(formulation)
 
    def validate_input(self, c, A, b):
        if len(c.shape) != 1:
            return False, "El vector c debe ser unidimensional"
        if len(A.shape) != 2:
            return False, "A debe ser una matriz bidimensional"
        if len(b.shape) != 1:
            return False, "El vector b debe ser unidimensional"
 
        n = len(c)
        m, n_A = A.shape
 
        if n_A != n:
            return False, f"A tiene {n_A} columnas pero c tiene {n} elementos"
        if len(b) != m:
            return False, f"A tiene {m} filas pero b tiene {len(b)} elementos"
        if np.any(b < 0):
            return False, "Todos los elementos de b deben ser no negativos"
 
        return True, "Datos válidos"
 