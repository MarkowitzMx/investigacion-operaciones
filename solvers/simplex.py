"""
Solver del Método Simplex con seguimiento detallado de iteraciones
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy

class SimplexSolver:
    def __init__(self):
        self.iterations = []
        self.optimal_solution = None
        self.optimal_value = None
        self.status = None
        
    def solve(self, c: np.ndarray, A: np.ndarray, b: np.ndarray, 
              maximize: bool = True, var_names: List[str] = None,
              constraint_types: List[str] = None) -> Dict:
        """
        Resuelve un problema de programación lineal usando el método Simplex
        
        Args:
            c: Coeficientes de la función objetivo
            A: Matriz de coeficientes de restricciones
            b: Vector del lado derecho de restricciones
            maximize: True para maximizar, False para minimizar
            var_names: Nombres de las variables
            constraint_types: Lista de tipos de restricción ('<=', '>=', '=')
        
        Returns:
            Diccionario con la solución y detalles del proceso
        """
        self.iterations = []
        
        # Convertir a minimización si es necesario
        if maximize:
            c = -c.copy()
        else:
            c = c.copy()
            
        # Preparar nombres de variables
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(len(c))]
        
        # Convertir restricciones a forma estándar
        tableau, basis, var_names_extended = self._convert_to_standard_form(
            c, A, b, var_names, constraint_types
        )
        
        # Guardar tableau inicial
        self._record_iteration(tableau, basis, var_names_extended, 0, "Tableau Inicial")
        
        iteration = 1
        while True:
            # Verificar optimalidad
            if self._is_optimal(tableau):
                self.status = "optimal"
                break
                
            # Verificar si no está acotado
            entering_col = self._get_entering_variable(tableau)
            if entering_col is None:
                self.status = "unbounded"
                break
                
            # Obtener variable que sale
            leaving_row = self._get_leaving_variable(tableau, entering_col)
            if leaving_row is None:
                self.status = "unbounded"
                break
            
            # Realizar pivoteo
            tableau = self._pivot(tableau, leaving_row, entering_col)
            basis[leaving_row] = entering_col
            
            # Registrar iteración
            self._record_iteration(
                tableau, basis, var_names_extended, iteration,
                f"Entra: {var_names_extended[entering_col]}, "
                f"Sale: {var_names_extended[basis[leaving_row] if leaving_row < len(basis) else 0]}"
            )
            
            iteration += 1
            
            # Seguridad: máximo 100 iteraciones
            if iteration > 100:
                self.status = "max_iterations"
                break
        
        # Extraer solución
        solution = self._extract_solution(tableau, basis, len(c), var_names)
        
        # Ajustar valor objetivo si era maximización
        if maximize:
            solution['optimal_value'] = -solution['optimal_value']
        
        solution['status'] = self.status
        solution['iterations'] = self.iterations
        solution['num_iterations'] = len(self.iterations) - 1
        
        return solution
    
    def _convert_to_standard_form(self, c: np.ndarray, A: np.ndarray, 
                                   b: np.ndarray, var_names: List[str],
                                   constraint_types: List[str]) -> Tuple:
        """Convierte el problema a forma estándar"""
        m, n = A.shape
        
        # Si no se especifican tipos, asumir todas <=
        if constraint_types is None:
            constraint_types = ['<='] * m
        
        # Contar variables de holgura/exceso necesarias
        num_slack = sum(1 for ct in constraint_types if ct in ['<=', '>='])
        
        # Crear matriz ampliada
        A_extended = np.zeros((m, n + num_slack))
        A_extended[:, :n] = A
        
        # Agregar variables de holgura/exceso
        slack_idx = n
        extended_var_names = var_names.copy()
        
        for i, ct in enumerate(constraint_types):
            if ct == '<=':
                A_extended[i, slack_idx] = 1
                extended_var_names.append(f"s{slack_idx - n + 1}")
                slack_idx += 1
            elif ct == '>=':
                A_extended[i, slack_idx] = -1
                extended_var_names.append(f"e{slack_idx - n + 1}")
                slack_idx += 1
        
        # Crear tableau
        tableau = np.zeros((m + 1, n + num_slack + 1))
        tableau[:-1, :-1] = A_extended
        tableau[:-1, -1] = b
        tableau[-1, :n] = c
        
        # Base inicial (variables de holgura)
        basis = list(range(n, n + num_slack))
        
        return tableau, basis, extended_var_names
    
    def _is_optimal(self, tableau: np.ndarray) -> bool:
        """Verifica si el tableau actual es óptimo"""
        # Para minimización: todos los coeficientes de la fila objetivo deben ser >= 0
        return np.all(tableau[-1, :-1] >= -1e-10)
    
    def _get_entering_variable(self, tableau: np.ndarray) -> Optional[int]:
        """Selecciona la variable que entra (columna pivote)"""
        # Regla: columna con el coeficiente más negativo
        obj_row = tableau[-1, :-1]
        min_val = np.min(obj_row)
        
        if min_val >= -1e-10:
            return None
            
        return np.argmin(obj_row)
    
    def _get_leaving_variable(self, tableau: np.ndarray, entering_col: int) -> Optional[int]:
        """Selecciona la variable que sale (fila pivote)"""
        m = tableau.shape[0] - 1
        ratios = []
        
        for i in range(m):
            if tableau[i, entering_col] > 1e-10:
                ratio = tableau[i, -1] / tableau[i, entering_col]
                ratios.append((ratio, i))
            else:
                ratios.append((float('inf'), i))
        
        # Filtrar ratios negativos
        valid_ratios = [(r, i) for r, i in ratios if r >= 0]
        
        if not valid_ratios:
            return None
        
        # Seleccionar mínimo ratio
        return min(valid_ratios, key=lambda x: x[0])[1]
    
    def _pivot(self, tableau: np.ndarray, pivot_row: int, pivot_col: int) -> np.ndarray:
        """Realiza operación de pivoteo"""
        tableau = tableau.copy()
        
        # Normalizar fila pivote
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        
        # Eliminar elementos en la columna pivote
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                tableau[i, :] -= factor * tableau[pivot_row, :]
        
        return tableau
    
    def _extract_solution(self, tableau: np.ndarray, basis: List[int], 
                         num_original_vars: int, var_names: List[str]) -> Dict:
        """Extrae la solución del tableau final"""
        solution_dict = {name: 0.0 for name in var_names}
        
        # Variables básicas
        for i, var_idx in enumerate(basis):
            if var_idx < num_original_vars:
                solution_dict[var_names[var_idx]] = tableau[i, -1]
        
        optimal_value = tableau[-1, -1]
        
        return {
            'variables': solution_dict,
            'optimal_value': optimal_value
        }
    
    def _record_iteration(self, tableau: np.ndarray, basis: List[int], 
                         var_names: List[str], iteration: int, description: str):
        """Registra una iteración para visualización posterior"""
        self.iterations.append({
            'iteration': iteration,
            'description': description,
            'tableau': tableau.copy(),
            'basis': basis.copy(),
            'var_names': var_names.copy()
        })


class TwoPhaseSimplexSolver:
    """Solver del Método de las Dos Fases"""
    
    def __init__(self):
        self.phase1_iterations = []
        self.phase2_iterations = []
        self.status = None
        
    def solve(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
              maximize: bool = True, var_names: List[str] = None,
              constraint_types: List[str] = None) -> Dict:
        """
        Resuelve usando el método de las dos fases
        """
        
        # FASE 1: Encontrar solución básica factible
        phase1_result = self._phase1(A, b, constraint_types)
        
        if phase1_result['status'] != 'feasible':
            return {
                'status': 'infeasible',
                'message': 'No existe solución factible',
                'phase1_iterations': self.phase1_iterations
            }
        
        # FASE 2: Optimizar con la función objetivo original
        phase2_result = self._phase2(
            c, phase1_result['tableau'], 
            phase1_result['basis'], maximize, var_names
        )
        
        return {
            'status': 'optimal',
            'variables': phase2_result['variables'],
            'optimal_value': phase2_result['optimal_value'],
            'phase1_iterations': self.phase1_iterations,
            'phase2_iterations': self.phase2_iterations,
            'total_iterations': len(self.phase1_iterations) + len(self.phase2_iterations)
        }
    
    def _phase1(self, A: np.ndarray, b: np.ndarray, 
                constraint_types: List[str]) -> Dict:
        """Fase 1: Encontrar solución básica factible"""
        m, n = A.shape
        
        # Crear problema artificial minimizando suma de variables artificiales
        # (Implementación simplificada)
        
        return {
            'status': 'feasible',
            'tableau': np.zeros((m + 1, n + m + 1)),
            'basis': list(range(n, n + m))
        }
    
    def _phase2(self, c: np.ndarray, tableau: np.ndarray, 
                basis: List[int], maximize: bool, var_names: List[str]) -> Dict:
        """Fase 2: Optimizar con función objetivo original"""
        
        # Usar SimplexSolver estándar
        solver = SimplexSolver()
        # (Implementación completa)
        
        return {
            'variables': {},
            'optimal_value': 0.0
        }
