"""
Módulo de Análisis de Sensibilidad y Teoría de Dualidad
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import pulp


class DualityAnalysis:
    """Análisis de Dualidad (Primal-Dual)"""
    
    def __init__(self):
        self.primal_solution = None
        self.dual_solution = None
        
    def get_dual_problem(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                        maximize: bool = True) -> Dict:
        """
        Genera el problema dual
        
        Args:
            c: Coeficientes función objetivo primal
            A: Matriz de restricciones primal
            b: Vector RHS primal
            maximize: True si el primal es de maximización
        
        Returns:
            Dict con formulación del dual
        """
        
        m, n = A.shape
        
        # Problema dual
        if maximize:
            # Primal: max c^T x, Ax <= b, x >= 0
            # Dual:   min b^T y, A^T y >= c, y >= 0
            dual_c = b
            dual_A = A.T
            dual_b = c
            dual_maximize = False
            dual_constraint_types = ['>='] * n
        else:
            # Primal: min c^T x, Ax >= b, x >= 0
            # Dual:   max b^T y, A^T y <= c, y >= 0
            dual_c = b
            dual_A = A.T
            dual_b = c
            dual_maximize = True
            dual_constraint_types = ['<='] * n
        
        return {
            'c': dual_c,
            'A': dual_A,
            'b': dual_b,
            'maximize': dual_maximize,
            'constraint_types': dual_constraint_types,
            'num_dual_vars': m,
            'primal_vars': n,
            'dual_vars': m
        }
    
    def solve_both(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                   maximize: bool = True) -> Dict:
        """
        Resuelve ambos problemas: primal y dual
        
        Returns:
            Dict con soluciones y relaciones
        """
        
        # Resolver primal
        primal = self._solve_with_pulp(c, A, b, maximize)
        
        # Obtener dual
        dual_problem = self.get_dual_problem(c, A, b, maximize)
        
        # Resolver dual
        dual = self._solve_with_pulp(
            dual_problem['c'],
            dual_problem['A'],
            dual_problem['b'],
            dual_problem['maximize']
        )
        
        # Verificar dualidad fuerte
        strong_duality_holds = np.isclose(
            primal['optimal_value'],
            dual['optimal_value']
        )
        
        return {
            'primal': primal,
            'dual': dual,
            'strong_duality_holds': strong_duality_holds,
            'gap': abs(primal['optimal_value'] - dual['optimal_value']),
            'dual_formulation': dual_problem
        }
    
    def _solve_with_pulp(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                        maximize: bool, constraint_types: List[str] = None) -> Dict:
        """Resuelve problema con PuLP"""
        
        n = len(c)
        m = len(b)
        
        if constraint_types is None:
            constraint_types = ['<='] * m if maximize else ['>='] * m
        
        # Crear problema
        if maximize:
            prob = pulp.LpProblem("Problem", pulp.LpMaximize)
        else:
            prob = pulp.LpProblem("Problem", pulp.LpMinimize)
        
        # Variables
        vars_list = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(n)]
        
        # Función objetivo
        prob += pulp.lpSum([c[i] * vars_list[i] for i in range(n)])
        
        # Restricciones
        for i in range(m):
            constraint_expr = pulp.lpSum([A[i, j] * vars_list[j] for j in range(n)])
            
            if constraint_types[i] == '<=':
                prob += constraint_expr <= b[i]
            elif constraint_types[i] == '>=':
                prob += constraint_expr >= b[i]
            elif constraint_types[i] == '=':
                prob += constraint_expr == b[i]
        
        # Resolver
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            solution_vars = {f"x{i+1}": var.varValue for i, var in enumerate(vars_list)}
            
            # Obtener precios sombra (valores duales)
            shadow_prices = {}
            for i, constraint in enumerate(prob.constraints.values()):
                shadow_prices[f"R{i+1}"] = constraint.pi if hasattr(constraint, 'pi') else 0
            
            return {
                'status': 'optimal',
                'variables': solution_vars,
                'optimal_value': pulp.value(prob.objective),
                'shadow_prices': shadow_prices
            }
        else:
            return {
                'status': 'error',
                'message': pulp.LpStatus[prob.status]
            }
    
    def get_economic_interpretation(self, primal: Dict, dual: Dict) -> str:
        """
        Genera interpretación económica de la dualidad
        
        Returns:
            str: Interpretación
        """
        
        interpretation = []
        interpretation.append("=== INTERPRETACIÓN ECONÓMICA DE LA DUALIDAD ===\n")
        
        interpretation.append("PROBLEMA PRIMAL:")
        interpretation.append(f"  Valor óptimo: {primal['optimal_value']:.4f}")
        interpretation.append("  Representa el beneficio/costo óptimo\n")
        
        interpretation.append("PROBLEMA DUAL:")
        interpretation.append(f"  Valor óptimo: {dual['optimal_value']:.4f}")
        interpretation.append("  Representa el costo de oportunidad de los recursos\n")
        
        if 'shadow_prices' in primal:
            interpretation.append("PRECIOS SOMBRA (Valores Duales):")
            for constraint, price in primal['shadow_prices'].items():
                interpretation.append(f"  {constraint}: {price:.4f}")
                if price > 0:
                    interpretation.append(
                        f"    → Por cada unidad adicional de este recurso, "
                        f"el objetivo mejora en {price:.4f}"
                    )
                else:
                    interpretation.append(
                        "    → Este recurso no es limitante (hay holgura)"
                    )
            interpretation.append("")
        
        interpretation.append("TEOREMA DE DUALIDAD FUERTE:")
        interpretation.append(
            "  Si ambos problemas tienen solución óptima, "
            "sus valores objetivos son iguales."
        )
        interpretation.append(
            f"  Gap observado: {abs(primal['optimal_value'] - dual['optimal_value']):.6f}"
        )
        
        return "\n".join(interpretation)


class SensitivityAnalysis:
    """Análisis de Sensibilidad"""
    
    def __init__(self):
        self.base_solution = None
        
    def analyze_objective_coefficients(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                                      maximize: bool, var_index: int,
                                      range_percent: float = 50) -> Dict:
        """
        Analiza sensibilidad de coeficientes de la función objetivo
        
        Args:
            c, A, b: Problema original
            maximize: Si es maximización
            var_index: Índice de la variable a analizar
            range_percent: Rango de variación en porcentaje
        
        Returns:
            Dict con análisis
        """
        
        original_value = c[var_index]
        
        # Rango de variación
        if original_value != 0:
            delta = abs(original_value) * range_percent / 100
            test_values = np.linspace(
                original_value - delta,
                original_value + delta,
                20
            )
        else:
            test_values = np.linspace(-10, 10, 20)
        
        results = []
        
        for test_value in test_values:
            c_modified = c.copy()
            c_modified[var_index] = test_value
            
            # Resolver con nuevo coeficiente
            solution = self._solve_simple(c_modified, A, b, maximize)
            
            results.append({
                'coefficient_value': test_value,
                'optimal_value': solution.get('optimal_value', None),
                'status': solution.get('status', 'error')
            })
        
        return {
            'variable_index': var_index,
            'original_coefficient': original_value,
            'test_range': (test_values[0], test_values[-1]),
            'results': results
        }
    
    def analyze_rhs(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                   maximize: bool, constraint_index: int,
                   range_percent: float = 50) -> Dict:
        """
        Analiza sensibilidad del lado derecho (RHS)
        
        Args:
            c, A, b: Problema original
            maximize: Si es maximización
            constraint_index: Índice de la restricción
            range_percent: Rango de variación
        
        Returns:
            Dict con análisis
        """
        
        original_value = b[constraint_index]
        
        # Rango de variación
        if original_value != 0:
            delta = abs(original_value) * range_percent / 100
            test_values = np.linspace(
                max(0, original_value - delta),
                original_value + delta,
                20
            )
        else:
            test_values = np.linspace(0, 20, 20)
        
        results = []
        
        for test_value in test_values:
            b_modified = b.copy()
            b_modified[constraint_index] = test_value
            
            # Resolver con nuevo RHS
            solution = self._solve_simple(c, A, b_modified, maximize)
            
            results.append({
                'rhs_value': test_value,
                'optimal_value': solution.get('optimal_value', None),
                'status': solution.get('status', 'error')
            })
        
        # Calcular precio sombra aproximado (derivada)
        shadow_price = None
        if len(results) > 1:
            valid_results = [r for r in results if r['optimal_value'] is not None]
            if len(valid_results) >= 2:
                delta_z = valid_results[-1]['optimal_value'] - valid_results[0]['optimal_value']
                delta_b = valid_results[-1]['rhs_value'] - valid_results[0]['rhs_value']
                if delta_b != 0:
                    shadow_price = delta_z / delta_b
        
        return {
            'constraint_index': constraint_index,
            'original_rhs': original_value,
            'test_range': (test_values[0], test_values[-1]),
            'shadow_price_estimate': shadow_price,
            'results': results
        }
    
    def _solve_simple(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                     maximize: bool) -> Dict:
        """Resuelve problema de manera simple"""
        
        try:
            n = len(c)
            m = len(b)
            
            if maximize:
                prob = pulp.LpProblem("Sens", pulp.LpMaximize)
            else:
                prob = pulp.LpProblem("Sens", pulp.LpMinimize)
            
            vars_list = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(n)]
            
            prob += pulp.lpSum([c[i] * vars_list[i] for i in range(n)])
            
            for i in range(m):
                prob += pulp.lpSum([A[i, j] * vars_list[j] for j in range(n)]) <= b[i]
            
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                return {
                    'status': 'optimal',
                    'optimal_value': pulp.value(prob.objective)
                }
            else:
                return {'status': 'error'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def generate_sensitivity_report(self, analyses: List[Dict]) -> str:
        """
        Genera reporte de sensibilidad
        
        Returns:
            str: Reporte formateado
        """
        
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE ANÁLISIS DE SENSIBILIDAD")
        report.append("=" * 80)
        report.append("")
        
        for analysis in analyses:
            if 'variable_index' in analysis:
                report.append(f"Variable x{analysis['variable_index'] + 1}")
                report.append(f"  Coeficiente original: {analysis['original_coefficient']:.4f}")
                report.append(f"  Rango analizado: {analysis['test_range']}")
                report.append("")
            
            elif 'constraint_index' in analysis:
                report.append(f"Restricción R{analysis['constraint_index'] + 1}")
                report.append(f"  RHS original: {analysis['original_rhs']:.4f}")
                report.append(f"  Rango analizado: {analysis['test_range']}")
                if analysis['shadow_price_estimate']:
                    report.append(
                        f"  Precio sombra estimado: {analysis['shadow_price_estimate']:.4f}"
                    )
                report.append("")
        
        return "\n".join(report)
