
Copy

"""
Biblioteca de Ejemplos Predefinidos
"""
import numpy as np
 
EXAMPLES = {
    "Programación Lineal": {
        "Problema de Producción Básico": {
            'c': np.array([3.0, 5.0]),
            'A': np.array([[2.0, 1.0], [1.0, 2.0], [1.0, 0.0]]),
            'b': np.array([20.0, 16.0, 8.0]),
            'maximize': True,
            'constraint_types': ['<=', '<=', '<='],
            'var_names': ['x1', 'x2'],
            'description': """
            Una empresa produce dos productos: A y B.
            - Producto A genera $3 de utilidad
            - Producto B genera $5 de utilidad
            
            Restricciones:
            - Tiempo de máquina: 2A + 1B ≤ 20 horas
            - Mano de obra: 1A + 2B ≤ 16 horas
            - Disponibilidad A: 1A ≤ 8 unidades
            """
        },
 
        "Problema de Dieta": {
            'c': np.array([0.12, 0.10, 0.08]),
            'A': np.array([
                [100, 150, 80],
                [50,  80, 120],
                [20,  30,  40]
            ]),
            'b': np.array([500, 600, 200]),
            'maximize': False,
            'constraint_types': ['>=', '>=', '>='],
            'var_names': ['Alimento A', 'Alimento B', 'Alimento C'],
            'description': """
            Minimizar el costo de una dieta que cumpla:
            - Al menos 500g de proteína
            - Al menos 600g de carbohidratos
            - Al menos 200g de grasas
            """
        },
 
        "Asignación de Recursos": {
            'c': np.array([40.0, 30.0, 50.0]),
            'A': np.array([
                [2.0, 1.0, 3.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0]
            ]),
            'b': np.array([100.0, 80.0, 90.0]),
            'maximize': True,
            'constraint_types': ['<=', '<=', '<='],
            'var_names': ['Proyecto A', 'Proyecto B', 'Proyecto C'],
            'description': """
            Maximizar beneficios de inversión en proyectos.
            Restricciones de presupuesto, personal y tiempo.
            """
        },
 
        "Mezcla de Productos": {
            'c': np.array([25.0, 30.0]),
            'A': np.array([
                [0.5, 0.4],
                [0.3, 0.6],
                [1.0, 0.0],
                [0.0, 1.0]
            ]),
            'b': np.array([40.0, 45.0, 60.0, 50.0]),
            'maximize': True,
            'constraint_types': ['<=', '<=', '<=', '<='],
            'var_names': ['Producto X', 'Producto Y'],
            'description': """
            Mezcla óptima de productos considerando:
            - Capacidad de línea 1
            - Capacidad de línea 2
            - Demanda máxima X
            - Demanda máxima Y
            """
        },
 
        "Dos Fases — Minimización": {
            'c': np.array([2000.0, 500.0]),
            'A': np.array([
                [2.0, 3.0],
                [3.0, 6.0]
            ]),
            'b': np.array([36.0, 60.0]),
            'maximize': False,
            'constraint_types': ['>=', '>='],
            'var_names': ['x1', 'x2'],
            'description': """
            Minimizar Z = 2000x1 + 500x2
            Sujeto a:
            - 2x1 + 3x2 >= 36
            - 3x1 + 6x2 >= 60
            - x1, x2 >= 0
 
            Solución óptima: x1=12, x2=4, Z=26,000
            ⚠️ Usar método: Dos Fases
            """
        },
 
        "Dos Fases — Maximización": {
            'c': np.array([5.0, 4.0]),
            'A': np.array([
                [6.0, 4.0],
                [1.0, 2.0]
            ]),
            'b': np.array([24.0, 6.0]),
            'maximize': True,
            'constraint_types': ['<=', '<='],
            'var_names': ['x1', 'x2'],
            'description': """
            Maximizar Z = 5x1 + 4x2
            Sujeto a:
            - 6x1 + 4x2 <= 24
            - x1 + 2x2 <= 6
            - x1, x2 >= 0
 
            Solución óptima: x1=3, x2=1.5, Z=21
            ⚠️ Usar método: Dos Fases o Simplex
            """
        },
    },
 
    "Transporte": {
        "Distribución Clásica": {
            'supply': np.array([100.0, 150.0, 200.0]),
            'demand': np.array([120.0, 180.0, 150.0]),
            'costs': np.array([
                [10, 15, 20],
                [12, 10, 18],
                [11, 14, 16]
            ]),
            'description': """
            Tres almacenes (Oferta: 100, 150, 200)
            Tres tiendas (Demanda: 120, 180, 150)
            Minimizar costos de transporte
            """
        },
 
        "Red de Distribución": {
            'supply': np.array([500.0, 400.0, 300.0]),
            'demand': np.array([350.0, 450.0, 400.0]),
            'costs': np.array([
                [ 8, 12, 15],
                [10,  9, 11],
                [14, 13, 10]
            ]),
            'description': """
            Red de distribución con 3 centros y 3 destinos
            """
        }
    },
 
    "Asignación": {
        "Asignación de Tareas": {
            'cost_matrix': np.array([
                [9, 2, 7, 8],
                [6, 4, 3, 7],
                [5, 8, 1, 8],
                [7, 6, 9, 4]
            ]),
            'maximize': False,
            'description': """
            4 trabajadores, 4 tareas
            Minimizar tiempo total de ejecución
            """
        },
 
        "Selección de Proyectos": {
            'cost_matrix': np.array([
                [15, 20, 25],
                [18, 22, 19],
                [21, 17, 23]
            ]),
            'maximize': True,
            'description': """
            3 equipos, 3 proyectos
            Maximizar beneficio total
            """
        }
    },
 
    "Programación Entera": {
        "Problema de la Mochila": {
            'values':   np.array([60, 100, 120, 80, 90]),
            'weights':  np.array([10,  20,  30, 15, 25]),
            'capacity': 50.0,
            'description': """
            5 objetos con valores y pesos
            Capacidad: 50 unidades
            Maximizar valor total
            """
        },
 
        "Selección de Proyectos": {
            'c': np.array([10, 15, 20, 12, 18]),
            'A': np.array([
                [5, 7, 10, 6,  8],
                [3, 4,  6, 3,  5],
                [1, 1,  1, 1,  1]
            ]),
            'b': np.array([40, 20, 3]),
            'maximize': True,
            'integer_vars': [0, 1, 2, 3, 4],
            'description': """
            Selección de proyectos (0 o 1)
            Restricciones de presupuesto, personal y cantidad
            """
        }
    },
 
    "PERT-CPM": {
        "Proyecto de Construcción": {
            'activities': [
                {'id': 'A', 'duration': 3, 'predecessors': []},
                {'id': 'B', 'duration': 4, 'predecessors': []},
                {'id': 'C', 'duration': 2, 'predecessors': ['A']},
                {'id': 'D', 'duration': 5, 'predecessors': ['A']},
                {'id': 'E', 'duration': 3, 'predecessors': ['B', 'C']},
                {'id': 'F', 'duration': 2, 'predecessors': ['D', 'E']},
            ],
            'description': """
            Proyecto con 6 actividades
            Determinar ruta crítica y duración total
            """
        },
 
        "Desarrollo de Software": {
            'activities': [
                {'id': 'Análisis',       'duration':  5, 'predecessors': []},
                {'id': 'Diseño',         'duration':  7, 'predecessors': ['Análisis']},
                {'id': 'Codificación',   'duration': 10, 'predecessors': ['Diseño']},
                {'id': 'Pruebas',        'duration':  6, 'predecessors': ['Codificación']},
                {'id': 'Documentación',  'duration':  4, 'predecessors': ['Diseño']},
                {'id': 'Despliegue',     'duration':  2, 'predecessors': ['Pruebas', 'Documentación']},
            ],
            'description': """
            Fases de desarrollo de software
            Identificar actividades críticas
            """
        }
    }
}
 
 
def get_example(category: str, name: str):
    """Obtiene un ejemplo específico"""
    if category in EXAMPLES and name in EXAMPLES[category]:
        return EXAMPLES[category][name]
    return None
 
 
def get_categories():
    """Obtiene lista de categorías"""
    return list(EXAMPLES.keys())
 
 
def get_examples_in_category(category: str):
    """Obtiene ejemplos de una categoría"""
    if category in EXAMPLES:
        return list(EXAMPLES[category].keys())
    return []
 
 
def get_all_examples_summary():
    """Obtiene resumen de todos los ejemplos"""
    summary = []
    for category, examples in EXAMPLES.items():
        for name, data in examples.items():
            summary.append({
                'category': category,
                'name': name,
                'description': data.get('description', 'Sin descripción').strip()[:100]
            })
    return summary
 