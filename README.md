# 📖 Guía de Usuario - Sistema de Investigación de Operaciones

## Tabla de Contenidos
1. [Introducción](#introducción)
2. [Primeros Pasos](#primeros-pasos)
3. [Módulos](#módulos)
4. [Casos de Uso](#casos-de-uso)
5. [Tips y Trucos](#tips-y-trucos)
6. [Solución de Problemas](#solución-de-problemas)

## Introducción

Este sistema te permite resolver problemas de Investigación de Operaciones de manera visual e interactiva.

### ¿Para quién es este sistema?
- Estudiantes de Ingeniería
- Profesores de Investigación de Operaciones
- Profesionales que necesitan resolver problemas de optimización

## Primeros Pasos

### 1. Acceder al Sistema
- **Local**: Ejecuta `streamlit run app.py`
- **Web**: Accede a la URL compartida

### 2. Navegación
- **Barra Lateral**: Selecciona el módulo que necesitas
- **Pestañas**: Navega entre definir problema, resolver y ver resultados
- **Botones**: Interactúa con el sistema

### 3. Flujo Típico
```
Seleccionar Módulo → Definir Problema → Resolver → Ver Resultados → Exportar
```

## Módulos

### 📈 Programación Lineal

#### ¿Cuándo usarlo?
- Maximizar beneficios
- Minimizar costos
- Asignar recursos limitados
- Planificación de producción

#### Pasos:
1. **Definir el problema**:
   - Número de variables (productos, recursos, etc.)
   - Número de restricciones
   - Tipo de optimización (Max/Min)

2. **Ingresar datos**:
   - Coeficientes de la función objetivo
   - Coeficientes de las restricciones
   - Valores del lado derecho (RHS)

3. **Elegir método**:
   - **Simplex**: Para cualquier problema
   - **Gráfico**: Solo 2 variables (visual)
   - **Dos Fases**: Cuando las restricciones son complejas
   - **PuLP**: Usa optimizador industrial

4. **Interpretar resultados**:
   - Valor óptimo de Z
   - Valores de las variables
   - Iteraciones (si aplica)

#### Ejemplo Práctico:
**Problema**: Una panadería produce pan y pasteles.
- Pan: $3 de utilidad, 2h de horno, 1h de mano de obra
- Pastel: $5 de utilidad, 1h de horno, 2h de mano de obra
- Disponible: 20h de horno, 16h de mano de obra

**En el sistema**:
```
Maximizar Z = 3x₁ + 5x₂

Restricciones:
2x₁ + 1x₂ ≤ 20  (horno)
1x₁ + 2x₂ ≤ 16  (mano de obra)
x₁, x₂ ≥ 0
```

### 🔄 Dualidad y Sensibilidad

#### ¿Cuándo usarlo?
- Entender el valor de los recursos (precios sombra)
- Analizar cambios en parámetros
- Verificar robustez de la solución

#### Características:
1. **Problema Dual**:
   - Genera automáticamente el dual
   - Compara soluciones primal y dual
   - Interpreta económicamente

2. **Análisis de Sensibilidad**:
   - ¿Qué pasa si cambio un coeficiente?
   - ¿Hasta cuánto puedo variar el RHS?
   - ¿Cuánto vale un recurso adicional?

#### Interpretación:
- **Precio Sombra > 0**: El recurso es valioso, si aumentas su disponibilidad, mejora Z
- **Precio Sombra = 0**: El recurso tiene holgura, no está siendo limitante

### 🔢 Programación Entera

#### ¿Cuándo usarlo?
- Las variables deben ser números enteros
- Decisiones binarias (sí/no)
- No puedes producir 2.7 unidades

#### Tipos de Problemas:

1. **Programación Entera General**:
   - Variables pueden ser 0, 1, 2, 3, ...
   - Ejemplo: Número de máquinas a comprar

2. **Programación Binaria (0-1)**:
   - Variables solo pueden ser 0 o 1
   - Ejemplo: ¿Aceptar proyecto? Sí (1) o No (0)

3. **Problema de la Mochila**:
   - Seleccionar objetos con valor y peso
   - Maximizar valor sin exceder capacidad
   - Ejemplo: ¿Qué productos llevar al mercado?

#### Ejemplo: Mochila
```
Objetos:
- Laptop: Valor=1000, Peso=3kg
- Cámara: Valor=500, Peso=1kg
- Libros: Valor=200, Peso=2kg

Capacidad: 4kg
¿Qué llevar?
```

### 🌐 Análisis de Redes

#### Tipos de Problemas:

1. **Transporte**:
   - Enviar desde varios orígenes a varios destinos
   - Minimizar costos de envío
   - Métodos: Esquina Noroeste, Vogel

2. **Asignación**:
   - Asignar n trabajadores a n tareas
   - Minimizar tiempo o maximizar eficiencia
   - Algoritmo Húngaro

3. **Flujo Máximo**:
   - ¿Cuánto puede fluir por una red?
   - Ejemplo: Tráfico en red de carreteras

4. **PERT-CPM**:
   - Planificación de proyectos
   - Identificar ruta crítica
   - Calcular holguras

#### Ejemplo: Transporte
```
Almacenes → Tiendas

Almacén A (100 unidades) → Tienda 1 (120 unid): $10/unid
                         → Tienda 2 (180 unid): $15/unid
                         
Almacén B (150 unidades) → Tienda 1: $12/unid
                         → Tienda 2: $10/unid
```

## Casos de Uso

### Caso 1: Planificación de Producción

**Problema**: Fábrica produce sillas y mesas
- Silla: $50 utilidad, 2h de trabajo, 4 unidades de madera
- Mesa: $80 utilidad, 3h de trabajo, 6 unidades de madera
- Disponible: 100h de trabajo, 200 unidades de madera

**Solución en el sistema**:
1. Módulo: Programación Lineal
2. Variables: 2 (sillas, mesas)
3. Restricciones: 2 (trabajo, madera)
4. Función objetivo: Max 50s + 80m
5. Resolver con Simplex

### Caso 2: Asignación de Proyectos

**Problema**: 4 equipos, 4 proyectos, minimizar tiempo total

**Solución**:
1. Módulo: Análisis de Redes → Asignación
2. Matriz 4x4 con tiempos
3. Resolver
4. Ver asignación óptima

### Caso 3: Ruta Crítica de Proyecto

**Problema**: Construir casa con 10 actividades

**Solución**:
1. Módulo: PERT-CPM
2. Ingresar actividades con duraciones y predecesores
3. Ver ruta crítica
4. Identificar actividades que NO pueden retrasarse

## Tips y Trucos

### 💡 Mejores Prácticas

1. **Empieza Simple**:
   - Prueba con 2 variables primero
   - Usa el método gráfico para visualizar

2. **Verifica los Datos**:
   - ¿Están balanceados oferta y demanda?
   - ¿Son coherentes las restricciones?

3. **Compara Métodos**:
   - Usa la función "Comparar Métodos"
   - Verifica que dan el mismo resultado

4. **Guarda el Trabajo**:
   - Exporta a Excel regularmente
   - Usa el historial

5. **Interpreta**:
   - No solo mires números
   - ¿Tiene sentido la solución?
   - ¿Es práctica?

### ⚡ Atajos

- **Ejemplos predefinidos**: Usa la biblioteca para aprender
- **Copiar y pegar**: Puedes pegar matrices desde Excel
- **Exportar JSON**: Para reusar problemas después

## Solución de Problemas

### Problema: "No existe solución factible"
**Causa**: Las restricciones se contradicen
**Solución**: Revisa que las restricciones sean compatibles

### Problema: "Solución no acotada"
**Causa**: Falta una restricción
**Solución**: Agrega límites superiores a las variables

### Problema: El resultado no tiene sentido
**Causa**: Error en los datos de entrada
**Solución**: 
- Verifica signos de las restricciones
- Confirma si es maximizar o minimizar
- Revisa los coeficientes

### Problema: Tarda mucho en resolver
**Causa**: Problema muy grande
**Solución**: 
- Usa el método PuLP (más eficiente)
- Reduce el número de variables si es posible

## Glosario

- **Función Objetivo**: Lo que quieres maximizar o minimizar
- **Restricciones**: Limitaciones del problema
- **Variable de Decisión**: Lo que el sistema calculará
- **Solución Factible**: Solución que cumple todas las restricciones
- **Solución Óptima**: La mejor solución factible
- **Precio Sombra**: Valor marginal de un recurso
- **Holgura**: Recurso no utilizado
- **Ruta Crítica**: Secuencia de actividades que determina la duración mínima

## Recursos Adicionales

- Libro de texto del curso
- Videos tutoriales (próximamente)
- Foro de preguntas
- Oficina del profesor

---

**¿Necesitas más ayuda?**
Contacta al instructor o usa el botón de "ayuda" en la aplicación.
