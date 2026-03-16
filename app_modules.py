"""
Funciones adicionales para app.py - Módulos restantes
"""
import streamlit as st
import numpy as np
import pandas as pd
import json
import sys
from datetime import datetime
sys.path.append('/home/claude/investigacion-operaciones')

from modules.network_analysis import (TransportationProblem, AssignmentProblem,
                                      NetworkFlowProblems, PertCpm)
from modules.duality_sensitivity import DualityAnalysis, SensitivityAnalysis
from modules.integer_programming import IntegerProgramming
from utils.visualizations import *


# ─────────────────────────────────────────────────────────────────────────────
#  DUALIDAD Y SENSIBILIDAD
# ─────────────────────────────────────────────────────────────────────────────

def show_duality_sensitivity():
    st.markdown("## 🔄 Dualidad y Sensibilidad")
    tab1, tab2 = st.tabs(["🔄 Análisis Dual", "📊 Sensibilidad"])

    with tab1:
        st.markdown("### Análisis de Dualidad")
        if st.session_state.problem_data is None:
            st.warning("Primero resuelve un problema de Programación Lineal")
            return
        if st.button("🔄 Generar Problema Dual"):
            with st.spinner("Analizando..."):
                p = st.session_state.problem_data
                duality = DualityAnalysis()
                result  = duality.solve_both(p['c'], p['A'], p['b'], p['maximize'])
                c1, c2  = st.columns(2)
                with c1:
                    st.markdown("### Primal")
                    st.metric("Óptimo", f"{result['primal']['optimal_value']:.4f}")
                    st.json(result['primal']['variables'])
                with c2:
                    st.markdown("### Dual")
                    st.metric("Óptimo", f"{result['dual']['optimal_value']:.4f}")
                    st.json(result['dual']['variables'])
                st.markdown("### Interpretación Económica")
                st.code(duality.get_economic_interpretation(result['primal'], result['dual']))
                if result['strong_duality_holds']:
                    st.success("Se cumple la Dualidad Fuerte")
                else:
                    st.warning(f"Gap de dualidad: {result['gap']:.6f}")

    with tab2:
        st.markdown("### Análisis de Sensibilidad")
        if st.session_state.problem_data is None:
            st.warning("Primero resuelve un problema de Programación Lineal")
            return
        analysis_type = st.radio("Tipo:", ["Coeficientes de la FO", "Lado Derecho (RHS)"])
        p = st.session_state.problem_data

        if analysis_type == "Coeficientes de la FO":
            var_idx = st.selectbox("Variable:", range(len(p['c'])),
                                   format_func=lambda x: f"x{x+1}")
            rng = st.slider("Rango (%)", 10, 100, 50)
            if st.button("Analizar"):
                with st.spinner("..."):
                    s = SensitivityAnalysis()
                    r = s.analyze_objective_coefficients(
                        p['c'], p['A'], p['b'], p['maximize'], var_idx, rng)
                    xv = [i['coefficient_value'] for i in r['results'] if i['status']=='optimal']
                    yv = [i['optimal_value']     for i in r['results'] if i['status']=='optimal']
                    st.plotly_chart(
                        create_sensitivity_plot(np.array(xv), np.array(yv),
                                                f"Coef x{var_idx+1}", r['original_coefficient']),
                        use_container_width=True)
        else:
            ci  = st.selectbox("Restricción:", range(len(p['b'])),
                               format_func=lambda x: f"R{x+1}")
            rng = st.slider("Rango (%)", 10, 100, 50)
            if st.button("Analizar"):
                with st.spinner("..."):
                    s = SensitivityAnalysis()
                    r = s.analyze_rhs(p['c'], p['A'], p['b'], p['maximize'], ci, rng)
                    if r['shadow_price_estimate']:
                        st.metric("Precio Sombra", f"{r['shadow_price_estimate']:.4f}")
                    xv = [i['rhs_value']    for i in r['results'] if i['status']=='optimal']
                    yv = [i['optimal_value'] for i in r['results'] if i['status']=='optimal']
                    st.plotly_chart(
                        create_sensitivity_plot(np.array(xv), np.array(yv),
                                                f"RHS R{ci+1}", r['original_rhs']),
                        use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PROGRAMACIÓN ENTERA  (dispatcher)
# ─────────────────────────────────────────────────────────────────────────────

def show_integer_programming():
    st.markdown("## 🔢 Programación Entera")
    problem_type = st.radio(
        "Tipo de problema:",
        ["Programación Entera General", "Problema Binario (0-1)", "Problema de la Mochila"]
    )
    if problem_type == "Problema de la Mochila":
        _show_knapsack()
    else:
        _show_integer_input(is_binary=(problem_type == "Problema Binario (0-1)"))


# ─────────────────────────────────────────────────────────────────────────────
#  MOCHILA
# ─────────────────────────────────────────────────────────────────────────────

def _show_knapsack():
    st.markdown("### Problema de la Mochila")
    n = st.number_input("Objetos:", min_value=2, max_value=20, value=5)
    values, weights = [], []
    for i in range(n):
        c1, c2 = st.columns(2)
        with c1:
            values.append(st.number_input(f"Valor {i+1}", value=10.0, key=f"val{i}"))
        with c2:
            weights.append(st.number_input(f"Peso {i+1}", value=5.0, key=f"weight{i}"))
    cap = st.number_input("Capacidad:", value=20.0, min_value=0.0)
    if st.button("Resolver", type="primary"):
        with st.spinner("..."):
            sol = IntegerProgramming().solve_knapsack(np.array(values), np.array(weights), cap)
            st.success(f"Estado: {sol['status']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor",      f"{sol['total_value']:.2f}")
            c2.metric("Peso",       f"{sol['total_weight']:.2f}")
            c3.metric("Utiliz.",    f"{sol['utilization']*100:.1f}%")
            st.dataframe(pd.DataFrame([
                {'Obj': i+1, 'Valor': values[i], 'Peso': weights[i]}
                for i in sol['selected_items']
            ]), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PE GENERAL / BINARIA  —  interfaz con 3 pestañas
# ─────────────────────────────────────────────────────────────────────────────

def _show_integer_input(is_binary: bool):
    label = "PE Binaria (0-1)" if is_binary else "PE General"
    st.markdown(f"### {label}")
    if is_binary:
        st.info("Variables: solo **0 o 1**.")
    else:
        st.info("Variables: **enteras no negativas** (0, 1, 2, ...).")

    tab1, tab2, tab3 = st.tabs(["Definir Problema", "Resolver", "Resultados"])

    # ── Definir ──────────────────────────────────────────────────
    with tab1:
        st.markdown("### Definición")
        c1, c2 = st.columns(2)
        with c1:
            obj_type = st.radio("Optimización:", ["Maximizar","Minimizar"], key="ip_obj_type")
            maximize = (obj_type == "Maximizar")
            nv = int(st.number_input("Variables:", min_value=2, max_value=10, value=2, key="ip_nv"))
            nc = int(st.number_input("Restricciones:", min_value=1, max_value=20, value=2, key="ip_nc"))
        with c2:
            methods = ["Método Gráfico (2 variables)",
                       "Branch and Bound", "Cortes de Gomory", "PuLP (Optimizador)"]
            meth = st.selectbox("Método:", methods, key="ip_method")
            if meth == "Método Gráfico (2 variables)" and nv != 2:
                st.warning("El método gráfico requiere exactamente 2 variables.")

        # Función objetivo
        st.markdown("### Función Objetivo")
        c_inp = []
        cols = st.columns(nv)
        for i in range(nv):
            with cols[i]:
                c_inp.append(st.number_input(f"c{i+1} (x{i+1})", value=1.0, key=f"ip_c{i}"))
        terms = " + ".join([f"{c_inp[i]:g}·x{i+1}" for i in range(nv)])
        st.markdown(f"**{obj_type}**  Z = {terms}")

        # Restricciones
        st.markdown("### Restricciones")
        A_inp, b_inp, ct_inp = [], [], []
        for i in range(nc):
            st.markdown(f"**R{i+1}:**")
            cols = st.columns(nv + 2)
            row = []
            for j in range(nv):
                with cols[j]:
                    row.append(st.number_input(f"a{i+1}{j+1}", value=1.0, key=f"ip_a{i}_{j}"))
            with cols[nv]:
                ct = st.selectbox("", ["<=",">=","="], key=f"ip_ct{i}",
                                  label_visibility="collapsed")
                ct_inp.append(ct)
            with cols[nv+1]:
                bv = st.number_input(f"b{i+1}", value=10.0, key=f"ip_b{i}")
                b_inp.append(bv)
            A_inp.append(row)
            preview = " + ".join([f"{row[j]:g}·x{j+1}" for j in range(nv)])
            st.caption(f"  {preview} {ct} {bv:g}")

        # Integralidad
        st.markdown("### Integralidad")
        vlist = ", ".join([f"x{i+1}" for i in range(nv)])
        st.markdown(f"**{vlist}** ∈ {{0,1}}" if is_binary else f"**{vlist}** ∈ ℤ⁺ ∪ {{0}}")

        if st.button("💾 Guardar", type="primary", key="ip_save"):
            st.session_state.ip_problem_data = {
                'c': np.array(c_inp), 'A': np.array(A_inp), 'b': np.array(b_inp),
                'maximize': maximize, 'constraint_types': ct_inp,
                'var_names': [f"x{i+1}" for i in range(nv)],
                'method_name': meth, 'is_binary': is_binary,
            }
            st.session_state.ip_solution  = None
            st.session_state.ip_graphical = None
            st.success("✅ Guardado. Ve a **Resolver**.")

    # ── Resolver ─────────────────────────────────────────────────
    with tab2:
        st.markdown("### Resolver")
        if not st.session_state.get('ip_problem_data'):
            st.warning("Primero define y guarda el problema.")
            return
        prob = st.session_state.ip_problem_data

        with st.expander("Ver formulación"):
            otype = "Maximizar" if prob['maximize'] else "Minimizar"
            fo    = " + ".join([f"{prob['c'][i]:g}·{prob['var_names'][i]}" for i in range(len(prob['c']))])
            ls    = [f"{otype}  Z = {fo}", "", "Sujeto a:"]
            for i in range(len(prob['b'])):
                rt = " + ".join([f"{prob['A'][i,j]:g}·{prob['var_names'][j]}" for j in range(len(prob['c']))])
                ls.append(f"  {rt} {prob['constraint_types'][i]} {prob['b'][i]:g}")
            ls += ["", "  " + ", ".join(prob['var_names']) +
                   (" ∈ {0,1}" if prob['is_binary'] else " ∈ ℤ⁺ ∪ {0}")]
            st.code("\n".join(ls))

        if prob['method_name'] == "Método Gráfico (2 variables)":
            if len(prob['c']) != 2:
                st.error("El método gráfico requiere exactamente 2 variables.")
                return
            st.info("Se graficará la región factible, los puntos reticulares y la solución óptima.")

        if st.button("🚀 Resolver", type="primary", use_container_width=True, key="ip_solve"):
            with st.spinner("Resolviendo..."):
                if prob['method_name'] == "Método Gráfico (2 variables)":
                    _solve_graphical_ip()
                else:
                    _solve_integer_programming()

    # ── Resultados ───────────────────────────────────────────────
    with tab3:
        st.markdown("### Resultados")
        prob = st.session_state.get('ip_problem_data', {})
        if prob.get('method_name') == "Método Gráfico (2 variables)":
            _show_graphical_ip_results()
        else:
            _show_ip_results()


# ─────────────────────────────────────────────────────────────────────────────
#  MÉTODO GRÁFICO PE
# ─────────────────────────────────────────────────────────────────────────────

def _solve_graphical_ip():
    prob = st.session_state.ip_problem_data
    c, A, b, ct = prob['c'], prob['A'], prob['b'], prob['constraint_types']
    maximize = prob['maximize']

    # Rango de búsqueda
    max_x1 = max_x2 = 0
    for i in range(len(b)):
        if A[i,0] > 0: max_x1 = max(max_x1, b[i]/A[i,0])
        if A[i,1] > 0: max_x2 = max(max_x2, b[i]/A[i,1])
    max_x1 = int(np.ceil(max_x1)) + 2
    max_x2 = int(np.ceil(max_x2)) + 2

    def feasible(x1, x2):
        for i in range(len(b)):
            lhs = A[i,0]*x1 + A[i,1]*x2
            if ct[i]=='<=' and lhs > b[i]+1e-9: return False
            if ct[i]=='>=' and lhs < b[i]-1e-9: return False
            if ct[i]=='='  and abs(lhs-b[i])>1e-9: return False
        if prob.get('is_binary') and (x1 not in (0,1) or x2 not in (0,1)): return False
        return True

    r1 = range(0,2) if prob.get('is_binary') else range(0, max_x1+1)
    r2 = range(0,2) if prob.get('is_binary') else range(0, max_x2+1)
    pts = [{'x1':x1,'x2':x2,'z':c[0]*x1+c[1]*x2} for x1 in r1 for x2 in r2 if feasible(x1,x2)]

    if not pts:
        st.session_state.ip_graphical = {'status':'infeasible'}
        st.warning("No se encontraron puntos enteros factibles.")
        return

    opt = max(pts, key=lambda p: p['z']) if maximize else min(pts, key=lambda p: p['z'])
    vertices = _compute_vertices(A, b, ct, max_x1, max_x2)

    st.session_state.ip_graphical = {
        'status':'optimal', 'feasible_points':pts, 'optimal':opt,
        'vertices':vertices, 'max_x1':max_x1, 'max_x2':max_x2, 'prob':prob
    }
    st.session_state.ip_solution = {
        'status':'optimal', 'variables':{'x1':opt['x1'],'x2':opt['x2']},
        'optimal_value':opt['z'], 'method':'Método Gráfico PE'
    }
    st.session_state.history.append({
        'timestamp':datetime.now(), 'module':'Programación Entera',
        'method':'Método Gráfico', 'status':'optimal', 'objective_value':opt['z']
    })
    st.success(f"✅  x1={opt['x1']},  x2={opt['x2']},  Z={opt['z']:g}")


def _compute_vertices(A, b, ct, max_x1, max_x2):
    from itertools import combinations
    A_ext = np.vstack([A, [[-1,0]], [[0,-1]]])
    b_ext = np.append(b, [0,0])
    ct_e  = list(ct)+['>=','>=']
    verts = []
    for i, j in combinations(range(len(b_ext)), 2):
        try:
            A2 = np.array([[A_ext[i,0],A_ext[i,1]],[A_ext[j,0],A_ext[j,1]]])
            if abs(np.linalg.det(A2)) < 1e-10: continue
            x = np.linalg.solve(A2, np.array([b_ext[i],b_ext[j]]))
            if x[0]<-1e-6 or x[1]<-1e-6: continue
            ok = True
            for k in range(len(b)):
                lhs = A[k,0]*x[0]+A[k,1]*x[1]
                if ct[k]=='<=' and lhs>b[k]+1e-6: ok=False; break
                if ct[k]=='>=' and lhs<b[k]-1e-6: ok=False; break
                if ct[k]=='='  and abs(lhs-b[k])>1e-6: ok=False; break
            if ok: verts.append((round(x[0],4), round(x[1],4)))
        except Exception:
            continue
    uniq = []
    for v in verts:
        if not any(abs(v[0]-u[0])<1e-4 and abs(v[1]-u[1])<1e-4 for u in uniq):
            uniq.append(v)
    return uniq


def _show_graphical_ip_results():
    if not st.session_state.get('ip_graphical') or st.session_state.ip_graphical.get('status')!='optimal':
        st.info("Resuelve el problema primero.")
        return

    import plotly.graph_objects as go

    data = st.session_state.ip_graphical
    prob, pts, opt, verts = data['prob'], data['feasible_points'], data['optimal'], data['vertices']
    c, A, b, ct = prob['c'], prob['A'], prob['b'], prob['constraint_types']
    maximize, max_x1, max_x2 = prob['maximize'], data['max_x1'], data['max_x2']

    fig = go.Figure()

    # Región factible
    if len(verts) >= 3:
        cx = np.mean([v[0] for v in verts]); cy = np.mean([v[1] for v in verts])
        sv = sorted(verts, key=lambda v: np.arctan2(v[1]-cy, v[0]-cx))
        fig.add_trace(go.Scatter(
            x=[v[0] for v in sv]+[sv[0][0]], y=[v[1] for v in sv]+[sv[0][1]],
            fill='toself', fillcolor='rgba(37,99,235,0.10)',
            line=dict(color='rgba(37,99,235,0.35)', dash='dash', width=1.5),
            name='Región factible (continua)', hoverinfo='skip'
        ))

    # Restricciones
    x_plot = np.linspace(0, max_x1, 400)
    pal = ['#3b82f6','#f59e0b','#8b5cf6','#10b981','#ef4444']
    for i in range(len(b)):
        col = pal[i % len(pal)]
        if abs(A[i,1]) > 1e-9:
            fig.add_trace(go.Scatter(
                x=x_plot, y=(b[i]-A[i,0]*x_plot)/A[i,1], mode='lines',
                line=dict(color=col, width=2.5, dash='dot'),
                name=f"R{i+1}: {A[i,0]:g}x₁+{A[i,1]:g}x₂ {ct[i]} {b[i]:g}"
            ))

    # Puntos infactibles
    inf_x = [x1 for x1 in range(max_x1+1) for x2 in range(max_x2+1)
             if not any(p['x1']==x1 and p['x2']==x2 for p in pts)]
    inf_y = [x2 for x1 in range(max_x1+1) for x2 in range(max_x2+1)
             if not any(p['x1']==x1 and p['x2']==x2 for p in pts)]
    if inf_x:
        fig.add_trace(go.Scatter(x=inf_x, y=inf_y, mode='markers',
            marker=dict(color='#94a3b8', size=5, opacity=0.35),
            name='Infactibles', hoverinfo='skip'))

    # Puntos factibles (no óptimos)
    no = [p for p in pts if not(p['x1']==opt['x1'] and p['x2']==opt['x2'])]
    if no:
        fig.add_trace(go.Scatter(
            x=[p['x1'] for p in no], y=[p['x2'] for p in no],
            mode='markers+text',
            marker=dict(color='#22c55e', size=9, line=dict(color='white',width=1)),
            text=[f"Z={p['z']:g}" for p in no],
            textposition='top center', textfont=dict(size=9, color='#64748b'),
            name='Puntos enteros factibles'
        ))

    # Isoprofit
    if abs(c[1]) > 1e-9:
        fig.add_trace(go.Scatter(
            x=x_plot, y=(opt['z']-c[0]*x_plot)/c[1], mode='lines',
            line=dict(color='rgba(234,88,12,0.65)', width=2, dash='dashdot'),
            name=f"Isoprofit Z={opt['z']:g}"
        ))

    # Óptimo
    fig.add_trace(go.Scatter(
        x=[opt['x1']], y=[opt['x2']], mode='markers+text',
        marker=dict(color='#ea580c', size=18, symbol='star',
                    line=dict(color='white', width=2)),
        text=[f"★ ({opt['x1']},{opt['x2']})<br>Z={opt['z']:g}"],
        textposition='top right', textfont=dict(size=11, color='#ea580c'),
        name=f"Óptimo Z={opt['z']:g}"
    ))

    obj_str = "Maximizar" if maximize else "Minimizar"
    fig.update_layout(
        title=f"Método Gráfico PE  |  {obj_str} Z = {c[0]:g}x₁ + {c[1]:g}x₂",
        xaxis=dict(title='x₁', range=[-0.3, max_x1+0.5], gridcolor='#e2e8f0',
                   zeroline=True, zerolinecolor='#94a3b8'),
        yaxis=dict(title='x₂', range=[-0.3, max_x2+0.5], gridcolor='#e2e8f0',
                   zeroline=True, zerolinecolor='#94a3b8'),
        legend=dict(orientation='h', yanchor='bottom', y=-0.38, xanchor='left', x=0),
        plot_bgcolor='white', height=520, margin=dict(t=55, b=130)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Métricas
    c1, c2, c3 = st.columns(3)
    c1.metric("x₁*", opt['x1']); c2.metric("x₂*", opt['x2']); c3.metric("Z*", f"{opt['z']:g}")

    # Tabla puntos
    st.markdown("### Todos los Puntos Enteros Factibles")
    df = pd.DataFrame(pts).sort_values('z', ascending=not maximize)
    df.columns = ['x₁','x₂','Z']
    df['★'] = df.apply(lambda r: '★' if r['x₁']==opt['x1'] and r['x₂']==opt['x2'] else '', axis=1)
    st.dataframe(df, use_container_width=True, height=260)

    # Verificación
    st.markdown("### Verificación de Restricciones")
    rows = []
    for i in range(len(b)):
        lhs = A[i,0]*opt['x1']+A[i,1]*opt['x2']
        ok  = ((ct[i]=='<='and lhs<=b[i]+1e-6)or(ct[i]=='>='and lhs>=b[i]-1e-6)
               or(ct[i]=='='and abs(lhs-b[i])<=1e-6))
        rows.append({'R': f"R{i+1}",
                     'Cálculo': f"{A[i,0]:g}({opt['x1']})+{A[i,1]:g}({opt['x2']})={lhs:g}",
                     'Tipo': ct[i], 'RHS': f"{b[i]:g}", '✓': '✅' if ok else '❌'})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Paso a paso
    with st.expander("Ver procedimiento paso a paso"):
        rest = "\n".join([f"  {A[i,0]:g}x₁ + {A[i,1]:g}x₂ {ct[i]} {b[i]:g}" for i in range(len(b))])
        intercepts = "\n".join([
            f"- R{i+1}: x₁=0→x₂={b[i]/A[i,1]:.4g},  x₂=0→x₁={b[i]/A[i,0]:.4g}"
            if A[i,0]!=0 and A[i,1]!=0 else f"- R{i+1}: ver gráfica"
            for i in range(len(b))
        ])
        vstr = ", ".join([f"({v[0]:.3g},{v[1]:.3g})" for v in verts])
        st.markdown(f"""
**Paso 1 — Modelo**
```
{obj_str}  Z = {c[0]:g}x₁ + {c[1]:g}x₂
Sujeto a:
{rest}
  x₁, x₂ ≥ 0,  enteros
```
**Paso 2 — Interceptos**
{intercepts}

**Paso 3 — Región factible**
Vértices de la región continua: {vstr}

**Paso 4 — Puntos reticulares factibles encontrados:** {len(pts)}

**Paso 5 — Evaluación de Z**
Se evaluó Z = {c[0]:g}x₁ + {c[1]:g}x₂ en los {len(pts)} puntos (ver tabla).

**Paso 6 — Solución óptima entera**
**(x₁={opt['x1']}, x₂={opt['x2']})** produce Z = **{opt['z']:g}**
({'máximo' if maximize else 'mínimo'} entre todos los puntos enteros factibles).

**Paso 7 — Verificación ✅**
Todas las restricciones se cumplen (ver tabla de verificación).
        """)

    # Exportar
    st.markdown("### Exportar")
    c1, c2 = st.columns(2)
    with c1:
        out = json.dumps({'solucion':opt,'puntos':pts,'vertices':verts},
                         indent=2, default=str)
        st.download_button("📥 JSON", out,
                           f"pe_grafico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                           "application/json")
    with c2:
        lines = ["MÉTODO GRÁFICO PE","="*35,
                 f"x1*={opt['x1']}  x2*={opt['x2']}  Z*={opt['z']:g}","","Puntos factibles:"]
        for p in sorted(pts, key=lambda x: -x['z'] if maximize else x['z']):
            lines.append(f"  ({p['x1']},{p['x2']}) Z={p['z']:g}")
        st.download_button("📥 TXT", "\n".join(lines),
                           f"pe_grafico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                           "text/plain")


# ─────────────────────────────────────────────────────────────────────────────
#  SOLVER PE (B&B, Gomory, PuLP)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_integer_programming():
    prob = st.session_state.ip_problem_data
    method_map = {"Branch and Bound":"branch_and_bound",
                  "Cortes de Gomory":"gomory", "PuLP (Optimizador)":"pulp"}
    method = method_map.get(prob['method_name'], "pulp")
    try:
        sol = IntegerProgramming().solve(
            c=prob['c'], A=prob['A'], b=prob['b'], method=method,
            maximize=prob['maximize'], var_names=prob['var_names'],
            constraint_types=prob['constraint_types'], binary=prob['is_binary']
        )
        st.session_state.ip_solution = sol
        st.session_state.history.append({
            'timestamp':datetime.now(), 'module':'Programación Entera',
            'method':prob['method_name'], 'status':sol.get('status','unknown'),
            'objective_value':sol.get('optimal_value',None)
        })
        if sol.get('status') == 'optimal':
            st.success(f"✅ Solución óptima — Estado: {sol['status'].upper()}")
        else:
            st.warning(f"Estado: {sol.get('status','?').upper()}")
    except Exception as e:
        st.error(f"Error: {e}"); st.exception(e)


def _show_ip_results():
    if not st.session_state.get('ip_solution'):
        st.info("Resuelve el problema primero.")
        return
    sol  = st.session_state.ip_solution
    prob = st.session_state.get('ip_problem_data', {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Estado", sol.get('status','N/A').upper())
    val = sol.get('optimal_value', 0)
    c2.metric("Z*", f"{val:.4f}" if isinstance(val,float) else str(val))
    if 'num_iterations' in sol:
        c3.metric("Iteraciones", sol['num_iterations'])

    if 'variables' in sol:
        st.markdown("### Variables de Decisión")
        st.dataframe(pd.DataFrame([
            {'Variable':n, 'Valor entero': int(round(v)) if isinstance(v,float) else v,
             'Tipo':'∈{0,1}' if prob.get('is_binary') else '∈ℤ⁺'}
            for n, v in sol['variables'].items()
        ]), use_container_width=True)

        if prob.get('A') is not None:
            st.markdown("### Verificación")
            xv = np.array([sol['variables'].get(f"x{i+1}",0) for i in range(len(prob['c']))])
            Ax = prob['A'] @ xv
            rows = []
            for i in range(len(prob['b'])):
                lhs,rhs,ct2 = Ax[i],prob['b'][i],prob['constraint_types'][i]
                ok = ((ct2=='<='and lhs<=rhs+1e-6)or(ct2=='>='and lhs>=rhs-1e-6)
                      or(ct2=='='and abs(lhs-rhs)<=1e-6))
                rows.append({'R':f"R{i+1}",'LHS':f"{lhs:.4g}",'Tipo':ct2,
                              'RHS':f"{rhs:.4g}",'✓':'✅' if ok else '❌'})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if sol.get('branch_and_bound_tree'):
        with st.expander("Árbol B&B"): st.json(sol['branch_and_bound_tree'])
    if sol.get('cuts'):
        with st.expander(f"Cortes de Gomory ({len(sol['cuts'])})"):
            for i,cut in enumerate(sol['cuts'],1): st.markdown(f"**{i}:** `{cut}`")

    st.markdown("### Exportar")
    c1, c2 = st.columns(2)
    with c1:
        out = json.dumps({k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in sol.items()},
                         indent=2, default=str)
        st.download_button("📥 JSON", out,
                           f"pe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json","application/json")
    with c2:
        lines = ["SOLUCIÓN PE","="*30,
                 f"Estado: {sol.get('status','?').upper()}",
                 f"Z*: {sol.get('optimal_value','?')}","","Variables:"]
        for n,v in sol.get('variables',{}).items(): lines.append(f"  {n} = {v}")
        st.download_button("📥 TXT", "\n".join(lines),
                           f"pe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt","text/plain")


# ─────────────────────────────────────────────────────────────────────────────
#  REDES
# ─────────────────────────────────────────────────────────────────────────────

def show_network_analysis():
    st.markdown("## 🌐 Análisis de Redes")
    pt = st.selectbox("Tipo:", ["Transporte","Asignación","Camino más Corto","Flujo Máximo","PERT-CPM"])
    if pt == "Transporte":        show_transportation_problem()
    elif pt == "Asignación":      show_assignment_problem()
    elif pt == "PERT-CPM":        show_pert_cpm()
    else:                          st.info(f"🚧 '{pt}' en construcción...")


def show_transportation_problem():
    st.markdown("### 🚚 Transporte")

    # Leer datos precargados desde la biblioteca de ejemplos (si existen)
    pre = st.session_state.get('transport_example')
    if pre:
        st.success("📥 Ejemplo precargado — puedes modificar los valores antes de resolver.")

    c1, c2 = st.columns(2)
    with c1:
        no = st.number_input("Orígenes:", min_value=2, max_value=10,
                             value=int(pre['no']) if pre else 3)
    with c2:
        nd = st.number_input("Destinos:", min_value=2, max_value=10,
                             value=int(pre['nd']) if pre else 3)

    no, nd = int(no), int(nd)

    st.markdown("#### Oferta:")
    supply = []
    cols = st.columns(no)
    for i in range(no):
        default = float(pre['supply'][i]) if pre and i < len(pre['supply']) else 100.0
        with cols[i]:
            supply.append(st.number_input(f"O{i+1}", value=default, key=f"supply{i}"))

    st.markdown("#### Demanda:")
    demand = []
    cols = st.columns(nd)
    for j in range(nd):
        default = float(pre['demand'][j]) if pre and j < len(pre['demand']) else 100.0
        with cols[j]:
            demand.append(st.number_input(f"D{j+1}", value=default, key=f"demand{j}"))

    st.markdown("#### Matriz de Costos:")
    costs = []
    for i in range(no):
        row = []; cols = st.columns(nd); st.markdown(f"**O{i+1}:**")
        for j in range(nd):
            default = float(pre['costs'][i][j]) if pre and i < len(pre['costs']) and j < len(pre['costs'][i]) else 10.0
            with cols[j]:
                row.append(st.number_input(f"→D{j+1}", value=default, key=f"cost{i}{j}"))
        costs.append(row)

    meth = st.selectbox("Método:", ["Esquina Noroeste", "Vogel"])

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        resolver = st.button("🚀 Resolver", type="primary", use_container_width=True)
    with col_btn2:
        if pre and st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.transport_example = None
            st.rerun()

    if resolver:
        with st.spinner("Resolviendo..."):
            sol = TransportationProblem().solve(
                np.array(supply), np.array(demand), np.array(costs),
                "nw_corner" if meth == "Esquina Noroeste" else "vogel"
            )
            if sol['status'] == 'optimal':
                st.success("✅ Solución encontrada")
                st.metric("Costo Total", f"{sol['total_cost']:.2f}")
                st.markdown("#### Asignación óptima:")
                alloc_df = pd.DataFrame(
                    sol['allocation'],
                    columns=[f"D{j+1}" for j in range(nd)],
                    index=[f"O{i+1}" for i in range(no)]
                )
                st.dataframe(alloc_df, use_container_width=True)
                # Guardar en historial
                st.session_state.history.append({
                    'timestamp': datetime.now(), 'module': 'Transporte',
                    'method': meth, 'status': 'optimal',
                    'objective_value': sol['total_cost']
                })
            else:
                st.error(f"❌ {sol.get('message', 'Error')}")


def show_assignment_problem():
    st.markdown("### 👥 Asignación")

    # Leer datos precargados desde la biblioteca de ejemplos (si existen)
    pre = st.session_state.get('assignment_example')
    if pre:
        st.success("📥 Ejemplo precargado — puedes modificar los valores antes de resolver.")

    sz = st.number_input("Tamaño n×n:", min_value=2, max_value=10,
                         value=int(pre['size']) if pre else 3)
    sz = int(sz)

    default_max = bool(pre.get('maximize', False)) if pre else False
    mx = st.checkbox("Maximizar (en lugar de minimizar)", value=default_max)

    st.markdown("#### Matriz de Costos/Beneficios:")
    cost_matrix = []
    for i in range(sz):
        row = []; cols = st.columns(sz); st.markdown(f"**Agente {i+1}:**")
        for j in range(sz):
            default = float(pre['costs'][i][j]) if pre and i < len(pre['costs']) and j < len(pre['costs'][i]) else 10.0
            with cols[j]:
                row.append(st.number_input(f"→T{j+1}", value=default, key=f"assign{i}{j}"))
        cost_matrix.append(row)

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        resolver = st.button("🚀 Resolver", type="primary", use_container_width=True)
    with col_btn2:
        if pre and st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.assignment_example = None
            st.rerun()

    if resolver:
        with st.spinner("Resolviendo..."):
            sol = AssignmentProblem().solve(np.array(cost_matrix), mx)
            st.success("✅ Asignación óptima encontrada")
            st.metric("Costo/Beneficio Total", f"{sol['total_cost']:.2f}")
            st.markdown("#### Asignaciones:")
            df = pd.DataFrame(sol['assignments'])
            df['from'] = df['from'].apply(lambda x: f"Agente {x+1}")
            df['to']   = df['to'].apply(lambda x: f"Tarea {x+1}")
            st.dataframe(df, use_container_width=True)
            # Guardar en historial
            st.session_state.history.append({
                'timestamp': datetime.now(), 'module': 'Asignación',
                'method': 'Húngaro', 'status': 'optimal',
                'objective_value': sol['total_cost']
            })


def show_pert_cpm():
    st.markdown("### PERT-CPM")
    na = st.number_input("Actividades:", min_value=2, max_value=20, value=5)
    acts = []
    for i in range(na):
        with st.expander(f"Actividad {chr(65+i)}"):
            c1,c2 = st.columns(2)
            with c1: dur = st.number_input("Duración:", value=5.0, key=f"dur{i}")
            with c2: ps  = st.text_input("Predecesores:", key=f"pred{i}", placeholder="A,B")
            acts.append({'id':chr(65+i),'duration':dur,
                         'predecessors':[p.strip() for p in ps.split(',')] if ps else []})
    if st.button("Analizar"):
        with st.spinner("..."):
            sol = PertCpm().solve(acts)
            st.success("✅"); st.metric("Duración", f"{sol['project_duration']:.0f} días")
            st.info(f"Ruta crítica: **{' → '.join(sol['critical_path'])}**")
            st.dataframe(pd.DataFrame(sol['activity_details']), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  BIBLIOTECA DE EJEMPLOS
# ─────────────────────────────────────────────────────────────────────────────

def show_examples_library():
    st.markdown("## 📚 Biblioteca de Ejemplos")
    cat = st.selectbox("Categoría:", [
        "Programación Lineal", "Programación Entera", "Transporte", "Asignación"
    ])

    # ── Ejemplos de PL y PE ───────────────────────────────────────
    lp_pe_examples = {
        "Programación Lineal": {
                        "Dos Fases — Minimización": {
                'c': np.array([2000.0, 500.0]),
                'A': np.array([[2.0, 3.0],[3.0, 6.0]]),
                'b': np.array([36.0, 60.0]),
                'maximize': False,
                'var_names': ['x1', 'x2'],
                'constraint_types': ['>=', '>='],
                'description': "Min Z=2000x1+500x2  s.a. 2x1+3x2>=36, 3x1+6x2>=60  →  x1=12, x2=4, Z=26000"
            },
            "Dos Fases — Maximización": {
                'c': np.array([5.0, 4.0]),
                'A': np.array([[6.0, 4.0],[1.0, 2.0]]),
                'b': np.array([24.0, 6.0]),
                'maximize': True,
                'var_names': ['x1', 'x2'],
                'constraint_types': ['<=', '<='],
                'description': "Max Z=5x1+4x2  s.a. 6x1+4x2<=24, x1+2x2<=6  →  x1=3, x2=1.5, Z=21"
            },
            "Producción Simple": {
                'c': np.array([3., 5.]), 'A': np.array([[2., 1.], [1., 2.]]),
                'b': np.array([20., 16.]), 'maximize': True,
                'var_names': ['x1', 'x2'], 'constraint_types': ['<=', '<='],
                'description': "Max Z = 3x₁ + 5x₂  s.a. 2x₁+x₂≤20, x₁+2x₂≤16"
            }
        },
        "Programación Entera": {
            "Muebles — Método Gráfico": {
                'c': np.array([5., 8.]), 'A': np.array([[2., 4.], [3., 2.]]),
                'b': np.array([16., 12.]), 'maximize': True,
                'var_names': ['x1', 'x2'], 'constraint_types': ['<=', '<='],
                'is_binary': False, 'method_name': 'Método Gráfico (2 variables)',
                'description': "Max Z=5x₁+8x₂  →  Óptimo (2,3) Z=34"
            },
            "Muebles — Branch and Bound": {
                'c': np.array([5., 8.]), 'A': np.array([[2., 4.], [3., 2.]]),
                'b': np.array([16., 12.]), 'maximize': True,
                'var_names': ['x1', 'x2'], 'constraint_types': ['<=', '<='],
                'is_binary': False, 'method_name': 'Branch and Bound',
                'description': "Mismo problema de muebles resuelto con B&B"
            },
            "Inversión Binaria": {
                'c': np.array([25., 30., 15.]), 'A': np.array([[20., 25., 15.]]),
                'b': np.array([40.]), 'maximize': True,
                'var_names': ['x1', 'x2', 'x3'], 'constraint_types': ['<='],
                'is_binary': True, 'method_name': 'PuLP (Optimizador)',
                'description': "Selección de proyectos con presupuesto limitado"
            }
        }
    }

    # ── Ejemplos de Transporte ────────────────────────────────────
    transport_examples = {
        "Distribución 3×3 (balanceado)": {
            'no': 3, 'nd': 3,
            'supply':  [120., 80., 80.],
            'demand':  [150., 70., 60.],
            'costs':   [[2., 3., 1.], [5., 4., 8.], [5., 6., 8.]],
            'description': "3 orígenes, 3 destinos · Oferta=Demanda=280 · Minimizar costo"
        },
        "Distribución 3×4 (desbalanceado)": {
            'no': 3, 'nd': 4,
            'supply':  [100., 200., 150.],
            'demand':  [80., 90., 120., 160.],
            'costs':   [[3., 2., 7., 6.], [7., 5., 2., 3.], [2., 5., 4., 5.]],
            'description': "3 orígenes, 4 destinos · Oferta>Demanda · Minimizar costo"
        },
        "Fábrica — Tiendas 2×3": {
            'no': 2, 'nd': 3,
            'supply':  [200., 300.],
            'demand':  [150., 200., 150.],
            'costs':   [[4., 8., 8.], [16., 24., 16.]],
            'description': "2 fábricas, 3 tiendas · Sistema balanceado"
        },
    }

    # ── Ejemplos de Asignación ────────────────────────────────────
    assignment_examples = {
        "Asignación 3×3 — Minimizar costo": {
            'size': 3, 'maximize': False,
            'costs': [[9., 2., 7.], [3., 6., 4.], [5., 8., 1.]],
            'description': "3 agentes, 3 tareas · Minimizar costo total"
        },
        "Asignación 4×4 — Minimizar tiempo": {
            'size': 4, 'maximize': False,
            'costs': [[13., 4., 7., 6.], [1., 11., 5., 4.],
                      [6., 7., 2., 8.],  [1., 3., 5., 9.]],
            'description': "4 trabajadores, 4 trabajos · Minimizar tiempo"
        },
        "Asignación 3×3 — Maximizar beneficio": {
            'size': 3, 'maximize': True,
            'costs': [[10., 5., 8.], [6., 7., 9.], [4., 8., 6.]],
            'description': "3 vendedores, 3 regiones · Maximizar ventas"
        },
        "Asignación 4×4 — Maximizar eficiencia": {
            'size': 4, 'maximize': True,
            'costs': [[8., 4., 2., 6.], [0., 9., 5., 5.],
                      [3., 8., 9., 2.], [4., 3., 2., 4.]],
            'description': "4 máquinas, 4 trabajos · Maximizar eficiencia"
        },
    }

    # ── Renderizado según categoría ───────────────────────────────
    if cat in lp_pe_examples:
        examples = lp_pe_examples[cat]
        name = st.selectbox("Ejemplo:", list(examples.keys()))
        ex   = examples[name]
        st.markdown(f"**{ex['description']}**")

        with st.expander("👁️ Ver formulación"):
            c, A, b = ex['c'], ex['A'], ex['b']
            vn = ex.get('var_names', [f"x{i+1}" for i in range(len(c))])
            ct = ex.get('constraint_types', ['<='] * len(b))
            fo = " + ".join([f"{c[i]:g}·{vn[i]}" for i in range(len(c))])
            obj = "Max" if ex['maximize'] else "Min"
            ls  = [f"{obj} Z = {fo}", "", "Sujeto a:"]
            for i in range(len(b)):
                r = " + ".join([f"{A[i,j]:g}·{vn[j]}" for j in range(len(c))])
                ls.append(f"  {r} {ct[i]} {b[i]:g}")
            st.code("\n".join(ls))

        target = st.radio("Cargar en:", ["Programación Lineal", "Programación Entera"],
                          horizontal=True)
        if st.button("📥 Cargar Ejemplo"):
            if target == "Programación Lineal":
                st.session_state.problem_data    = ex
            else:
                st.session_state.ip_problem_data = ex
                st.session_state.ip_graphical    = None
                st.session_state.ip_solution     = None
            st.success(f"✅ Cargado en **{target}**. Ve al módulo para resolverlo.")

    elif cat == "Transporte":
        examples = transport_examples
        name = st.selectbox("Ejemplo:", list(examples.keys()))
        ex   = examples[name]
        st.markdown(f"**{ex['description']}**")

        # Vista previa de la matriz
        with st.expander("👁️ Ver datos del ejemplo"):
            no, nd = ex['no'], ex['nd']
            df_costs = pd.DataFrame(
                ex['costs'],
                columns=[f"D{j+1}" for j in range(nd)],
                index=[f"O{i+1}" for i in range(no)]
            )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Oferta:**")
                st.dataframe(pd.DataFrame({'Origen': [f"O{i+1}" for i in range(no)],
                                           'Oferta': ex['supply']}),
                             use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**Demanda:**")
                st.dataframe(pd.DataFrame({'Destino': [f"D{j+1}" for j in range(nd)],
                                           'Demanda': ex['demand']}),
                             use_container_width=True, hide_index=True)
            st.markdown("**Matriz de costos:**")
            st.dataframe(df_costs, use_container_width=True)

            total_oferta  = sum(ex['supply'])
            total_demanda = sum(ex['demand'])
            bal = "✅ Balanceado" if abs(total_oferta - total_demanda) < 1e-6 else \
                  f"⚠️ Desbalanceado (Oferta={total_oferta:.0f}, Demanda={total_demanda:.0f})"
            st.info(bal)

        if st.button("📥 Cargar en Análisis de Redes → Transporte"):
            st.session_state.transport_example = ex
            st.success("✅ Ejemplo cargado. Ve a **Análisis de Redes → Transporte**.")

    elif cat == "Asignación":
        examples = assignment_examples
        name = st.selectbox("Ejemplo:", list(examples.keys()))
        ex   = examples[name]
        st.markdown(f"**{ex['description']}**")

        with st.expander("👁️ Ver matriz de costos/beneficios"):
            sz = ex['size']
            df_costs = pd.DataFrame(
                ex['costs'],
                columns=[f"Tarea {j+1}" for j in range(sz)],
                index=[f"Agente {i+1}" for i in range(sz)]
            )
            st.dataframe(df_costs, use_container_width=True)
            obj_lbl = "Maximizar beneficio" if ex['maximize'] else "Minimizar costo"
            st.info(f"Objetivo: **{obj_lbl}**")

        if st.button("📥 Cargar en Análisis de Redes → Asignación"):
            st.session_state.assignment_example = ex
            st.success("✅ Ejemplo cargado. Ve a **Análisis de Redes → Asignación**.")


# ─────────────────────────────────────────────────────────────────────────────
#  COMPARAR MÉTODOS
# ─────────────────────────────────────────────────────────────────────────────

def show_method_comparison():
    st.markdown("## 📊 Comparar Métodos")
    if st.session_state.problem_data is None:
        st.warning("Primero define un problema de Programación Lineal"); return
    if st.button("🔄 Comparar"):
        with st.spinner("..."):
            from modules.linear_programming import LinearProgrammingModule
            p = st.session_state.problem_data
            r = LinearProgrammingModule().compare_methods(
                p['c'],p['A'],p['b'],p['maximize'],
                p.get('var_names'),p.get('constraint_types'))
            st.dataframe(pd.DataFrame([
                {'Método':m,'Estado':v.get('status','?'),
                 'Z':v.get('optimal_value','?'),'t(s)':f"{v.get('time',0):.4f}"}
                for m,v in r.items()
            ]), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HISTORIAL
# ─────────────────────────────────────────────────────────────────────────────

def show_history():
    st.markdown("## 💾 Historial")
    if not st.session_state.history:
        st.info("No hay problemas resueltos aún"); return
    st.markdown(f"**{len(st.session_state.history)} problemas resueltos**")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    if st.button("🗑️ Limpiar"):
        st.session_state.history = []
        st.success("✅ Limpiado")
        st.rerun()
