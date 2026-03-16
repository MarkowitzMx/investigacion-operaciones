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
        self.iterations = []
 
        if maximize:
            c = -c.copy()
        else:
            c = c.copy()
 
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(len(c))]
 
        tableau, basis, var_names_extended = self._convert_to_standard_form(
            c, A, b, var_names, constraint_types
        )
 
        self._record_iteration(tableau, basis, var_names_extended, 0, "Tableau Inicial")
 
        iteration = 1
        while True:
            if self._is_optimal(tableau):
                self.status = "optimal"
                break
 
            entering_col = self._get_entering_variable(tableau)
            if entering_col is None:
                self.status = "unbounded"
                break
 
            leaving_row = self._get_leaving_variable(tableau, entering_col)
            if leaving_row is None:
                self.status = "unbounded"
                break
 
            prev_leaving = basis[leaving_row]
            tableau = self._pivot(tableau, leaving_row, entering_col)
            basis[leaving_row] = entering_col
 
            self._record_iteration(
                tableau, basis, var_names_extended, iteration,
                f"Entra: {var_names_extended[entering_col]}, "
                f"Sale: {var_names_extended[prev_leaving]}"
            )
 
            iteration += 1
            if iteration > 100:
                self.status = "max_iterations"
                break
 
        solution = self._extract_solution(tableau, basis, len(c), var_names)
 
        if maximize:
            solution['optimal_value'] = -solution['optimal_value']
 
        solution['status'] = self.status
        solution['iterations'] = self.iterations
        solution['num_iterations'] = len(self.iterations) - 1
 
        return solution
 
    def _convert_to_standard_form(self, c, A, b, var_names, constraint_types):
        m, n = A.shape
 
        if constraint_types is None:
            constraint_types = ['<='] * m
 
        num_slack = sum(1 for ct in constraint_types if ct in ['<=', '>='])
 
        A_extended = np.zeros((m, n + num_slack))
        A_extended[:, :n] = A
 
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
 
        tableau = np.zeros((m + 1, n + num_slack + 1))
        tableau[:-1, :-1] = A_extended
        tableau[:-1, -1] = b
        tableau[-1, :n] = c
 
        basis = list(range(n, n + num_slack))
 
        return tableau, basis, extended_var_names
 
    def _is_optimal(self, tableau):
        return np.all(tableau[-1, :-1] >= -1e-10)
 
    def _get_entering_variable(self, tableau):
        obj_row = tableau[-1, :-1]
        min_val = np.min(obj_row)
        if min_val >= -1e-10:
            return None
        return int(np.argmin(obj_row))
 
    def _get_leaving_variable(self, tableau, entering_col):
        m = tableau.shape[0] - 1
        ratios = []
 
        for i in range(m):
            if tableau[i, entering_col] > 1e-10:
                ratio = tableau[i, -1] / tableau[i, entering_col]
                ratios.append((ratio, i))
 
        if not ratios:
            return None
 
        return min(ratios, key=lambda x: x[0])[1]
 
    def _pivot(self, tableau, pivot_row, pivot_col):
        tableau = tableau.copy()
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
 
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                tableau[i, :] -= factor * tableau[pivot_row, :]
 
        return tableau
 
    def _extract_solution(self, tableau, basis, num_original_vars, var_names):
        solution_dict = {name: 0.0 for name in var_names}
 
        for i, var_idx in enumerate(basis):
            if var_idx < num_original_vars:
                solution_dict[var_names[var_idx]] = tableau[i, -1]
 
        optimal_value = tableau[-1, -1]
 
        return {
            'variables': solution_dict,
            'optimal_value': optimal_value
        }
 
    def _record_iteration(self, tableau, basis, var_names, iteration, description):
        self.iterations.append({
            'iteration': iteration,
            'description': description,
            'tableau': tableau.copy(),
            'basis': basis.copy(),
            'var_names': var_names.copy()
        })
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MÉTODO DE LAS DOS FASES — IMPLEMENTACIÓN COMPLETA
# ─────────────────────────────────────────────────────────────────────────────
class TwoPhaseSimplexSolver:
    """Solver del Método de las Dos Fases — implementación completa"""
 
    def __init__(self):
        self.phase1_iterations = []
        self.phase2_iterations = []
        self.status = None
 
    # ── API pública ──────────────────────────────────────────────────────────
    def solve(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
              maximize: bool = True, var_names: List[str] = None,
              constraint_types: List[str] = None) -> Dict:
 
        self.phase1_iterations = []
        self.phase2_iterations = []
 
        m, n = A.shape
 
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(n)]
        if constraint_types is None:
            constraint_types = ['<='] * m
 
        # Asegurar b >= 0 (multiplicar filas negativas por -1)
        A, b, constraint_types = self._ensure_nonneg_rhs(A, b, constraint_types)
 
        # ── FASE 1 ───────────────────────────────────────────────────────────
        phase1_result = self._phase1(A, b, constraint_types, var_names, n, m)
 
        if phase1_result['status'] != 'feasible':
            return {
                'status': 'infeasible',
                'message': 'No existe solución básica factible (Fase 1 no alcanzó Z=0)',
                'phase1_iterations': self.phase1_iterations,
                'phase2_iterations': []
            }
 
        # ── FASE 2 ───────────────────────────────────────────────────────────
        phase2_result = self._phase2(
            c, A, b, maximize, var_names, constraint_types,
            phase1_result['basis'], phase1_result['tableau'], n, m
        )
 
        return {
            'status': phase2_result['status'],
            'variables': phase2_result.get('variables', {}),
            'optimal_value': phase2_result.get('optimal_value', 0.0),
            'phase1_iterations': self.phase1_iterations,
            'phase2_iterations': self.phase2_iterations,
            'num_phase1': len(self.phase1_iterations) - 1,
            'num_phase2': len(self.phase2_iterations) - 1,
            'total_iterations': (
                max(0, len(self.phase1_iterations) - 1) +
                max(0, len(self.phase2_iterations) - 1)
            ),
            'method': 'Método de las Dos Fases'
        }
 
    # ── Fase 1 ───────────────────────────────────────────────────────────────
    def _phase1(self, A, b, constraint_types, var_names, n, m):
        """
        Construye tableau con variables artificiales y minimiza su suma.
        Si el óptimo es 0, existe SBF.
        """
        # Número de variables de holgura/exceso
        num_slack = sum(1 for ct in constraint_types if ct in ['<=', '>='])
        # Variables artificiales para restricciones >= y =
        artificial_rows = [i for i, ct in enumerate(constraint_types) if ct in ['>=', '=']]
        num_art = len(artificial_rows)
 
        total_cols = n + num_slack + num_art  # sin RHS
 
        # Construir A extendida con holguras
        A_ext = np.zeros((m, n + num_slack))
        A_ext[:, :n] = A
        slack_idx = n
        slack_names = []
        for i, ct in enumerate(constraint_types):
            if ct == '<=':
                A_ext[i, slack_idx] = 1
                slack_names.append(f"s{slack_idx - n + 1}")
                slack_idx += 1
            elif ct == '>=':
                A_ext[i, slack_idx] = -1
                slack_names.append(f"e{slack_idx - n + 1}")
                slack_idx += 1
 
        # Agregar columnas de variables artificiales
        A_full = np.zeros((m, total_cols))
        A_full[:, :n + num_slack] = A_ext
        art_names = []
        basis = [-1] * m
 
        # Asignar base inicial: holguras para <=, artificiales para >= y =
        slack_ptr = n
        art_ptr = n + num_slack
        basis_slack = {}  # row -> slack col para <=
 
        for i, ct in enumerate(constraint_types):
            if ct == '<=':
                basis[i] = slack_ptr
                slack_ptr += 1
            elif ct == '>=':
                A_full[i, art_ptr] = 1
                art_names.append(f"a{len(art_names)+1}")
                basis[i] = art_ptr
                art_ptr += 1
            elif ct == '=':
                A_full[i, art_ptr] = 1
                art_names.append(f"a{len(art_names)+1}")
                basis[i] = art_ptr
                art_ptr += 1
 
        all_var_names = var_names + slack_names + art_names
 
        # Función objetivo Fase 1: minimizar sum(artificiales)
        # Fila Z = sum de filas con artificiales en base (para eliminar artificiales)
        tableau = np.zeros((m + 1, total_cols + 1))
        tableau[:m, :total_cols] = A_full
        tableau[:m, -1] = b
 
        # Fila objetivo: coeficiente 1 para artificiales
        for j in range(n + num_slack, total_cols):
            tableau[-1, j] = 1.0
 
        # Eliminar artificiales de la fila objetivo (ya están en base)
        for i, bi in enumerate(basis):
            if bi >= n + num_slack:  # es artificial
                tableau[-1, :] -= tableau[i, :]
 
        self._record_phase(self.phase1_iterations, tableau, basis, all_var_names,
                           0, "Tableau Inicial — Fase 1")
 
        # Iterar simplex minimizando
        iteration = 1
        while True:
            if np.all(tableau[-1, :-1] >= -1e-10):
                break
 
            entering = int(np.argmin(tableau[-1, :-1]))
            leaving = self._min_ratio(tableau, entering)
 
            if leaving is None:
                self.status = "unbounded"
                return {'status': 'unbounded'}
 
            prev = basis[leaving]
            tableau = self._pivot_op(tableau, leaving, entering)
            basis[leaving] = entering
 
            self._record_phase(self.phase1_iterations, tableau, basis, all_var_names,
                               iteration,
                               f"Entra: {all_var_names[entering]}, "
                               f"Sale: {all_var_names[prev]}")
            iteration += 1
            if iteration > 200:
                break
 
        # Verificar factibilidad: Z fase1 debe ser ~0
        z1 = abs(tableau[-1, -1])
        if z1 > 1e-6:
            return {'status': 'infeasible'}
 
        return {
            'status': 'feasible',
            'tableau': tableau,
            'basis': basis,
            'all_var_names': all_var_names,
            'num_slack': num_slack,
            'num_art': num_art
        }
 
    # ── Fase 2 ───────────────────────────────────────────────────────────────
    def _phase2(self, c, A, b, maximize, var_names, constraint_types,
                basis, tableau_p1, n, m):
        """
        Toma el tableau final de Fase 1, elimina columnas artificiales
        y optimiza con la función objetivo original.
        """
        num_slack = sum(1 for ct in constraint_types if ct in ['<=', '>='])
        num_art   = sum(1 for ct in constraint_types if ct in ['>=', '='])
        total_orig = n + num_slack  # columnas sin artificiales
 
        # Reconstruir tableau sin columnas artificiales
        # Filas de restricciones: tomar solo columnas originales + RHS
        T = np.zeros((m + 1, total_orig + 1))
        T[:m, :total_orig] = tableau_p1[:m, :total_orig]
        T[:m, -1]          = tableau_p1[:m, -1]
 
        # Nombre de variables (sin artificiales)
        slack_names = []
        for i, ct in enumerate(constraint_types):
            if ct in ['<=', '>=']:
                slack_names.append(f"s{len(slack_names)+1}" if ct == '<=' else f"e{len(slack_names)+1}")
        all_var_names = var_names + slack_names
 
        # Construir fila objetivo Fase 2
        # Internamente siempre minimizamos:
        # - maximizar: negamos c
        # - minimizar: usamos c directamente
        if maximize:
            c_phase2 = -c.copy()
        else:
            c_phase2 = c.copy()
 
        T[-1, :n] = c_phase2
        T[-1, n:total_orig] = 0.0
        T[-1, -1] = 0.0
 
        # Eliminar variables básicas de la fila objetivo
        for i, bi in enumerate(basis):
            if bi < total_orig and abs(T[-1, bi]) > 1e-10:
                T[-1, :] -= T[-1, bi] * T[i, :]
 
        self._record_phase(self.phase2_iterations, T, basis, all_var_names,
                           0, "Tableau Inicial — Fase 2")
 
        iteration = 1
        status = "optimal"
        while True:
            # Solo revisar optimalidad en variables originales (0..n-1)
            # Las variables de exceso/holgura pueden tener coeficientes
            # negativos espurios que no indican mejora real
            obj_row = T[-1, :n]
            if np.all(obj_row >= -1e-10):
                status = "optimal"
                break
 
            entering = int(np.argmin(obj_row))
            leaving  = self._min_ratio(T, entering)
 
            if leaving is None:
                status = "unbounded"
                break
 
            prev = basis[leaving]
            T = self._pivot_op(T, leaving, entering)
            basis[leaving] = entering
 
            self._record_phase(self.phase2_iterations, T, basis, all_var_names,
                               iteration,
                               f"Entra: {all_var_names[entering]}, "
                               f"Sale: {all_var_names[prev]}")
            iteration += 1
            if iteration > 200:
                status = "max_iterations"
                break
 
        if status != "optimal":
            return {'status': status}
 
        # Extraer solución
        solution = {name: 0.0 for name in var_names}
        for i, bi in enumerate(basis):
            if bi < n:
                solution[var_names[bi]] = T[i, -1]
 
        # Signo del valor óptimo:
        # - Minimizar: c directo,  T[-1,-1] = -Z*  → opt = -T[-1,-1]
        # - Maximizar: c negado,   T[-1,-1] = +Z*  → opt =  T[-1,-1]
        if maximize:
            opt_val = T[-1, -1]
        else:
            opt_val = -T[-1, -1]
 
        return {
            'status': 'optimal',
            'variables': solution,
            'optimal_value': opt_val
        }
 
    # ── Utilidades internas ──────────────────────────────────────────────────
    def _ensure_nonneg_rhs(self, A, b, constraint_types):
        A = A.copy().astype(float)
        b = b.copy().astype(float)
        ct = list(constraint_types)
        for i in range(len(b)):
            if b[i] < 0:
                A[i, :] *= -1
                b[i]    *= -1
                if ct[i] == '<=':
                    ct[i] = '>='
                elif ct[i] == '>=':
                    ct[i] = '<='
        return A, b, ct
 
    def _min_ratio(self, tableau, entering_col):
        m = tableau.shape[0] - 1
        ratios = []
        for i in range(m):
            if tableau[i, entering_col] > 1e-10:
                ratios.append((tableau[i, -1] / tableau[i, entering_col], i))
        if not ratios:
            return None
        return min(ratios, key=lambda x: x[0])[1]
 
    def _pivot_op(self, tableau, pivot_row, pivot_col):
        T = tableau.copy()
        T[pivot_row, :] /= T[pivot_row, pivot_col]
        for i in range(T.shape[0]):
            if i != pivot_row:
                T[i, :] -= T[i, pivot_col] * T[pivot_row, :]
        return T
 
    def _record_phase(self, iterations_list, tableau, basis, var_names,
                      iteration, description):
        iterations_list.append({
            'iteration':   iteration,
            'description': description,
            'tableau':     tableau.copy(),
            'basis':       basis.copy(),
            'var_names':   var_names.copy()
        })