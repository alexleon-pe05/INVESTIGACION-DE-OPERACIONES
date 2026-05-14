# ================================================================
# SISTEMA PERT/CPM - NOTACIÓN AOA (v5 - render con Graphviz)
import re
from collections import defaultdict
from io import BytesIO
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Librería para el nuevo motor de layout (graphviz)
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


# Helper para formatear números (entero si es exacto, sino 2 decimales)
def _fmt_num(x):
    """Formatea int si es entero, sino con 2 decimales. Robusto a None/NaN."""
    if x is None:
        return ""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(x - round(x)) < 1e-6:
        return int(round(x))
    return round(x, 2)


# ================================================================
# 1. PARSEO (con soporte de "A < B, C" y predecesoras tradicionales)
# ================================================================
def parse_table(df, warning_list=None):
    """
    Normaliza el DataFrame.
    - Si la columna de precedencias contiene '<', se interpreta como
      'actividad < sucesor1, sucesor2' (la actividad precede a esos).
    - En caso contrario, se interpreta como lista de predecesores.
    Ahora devuelve también una lista de warnings.
    """
    if warning_list is None:
        warning_list = []
    df = df.copy()
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ('actividad', 'activity', 'tarea', 'task', 'nombre', 'act',
                  'actividades'):
            col_map[c] = 'Actividad'
        elif cl in ('duración', 'duracion', 'duration', 'tiempo', 'time',
                    'dur', 'd', 'duración (semanas)', 'duracion (semanas)',
                    'tiempo (días)', 'tiempo (dias)'):
            col_map[c] = 'Duración'
        elif cl in ('predecesoras', 'predecessors', 'precedencia',
                    'predecesor', 'predecesores', 'pred', 'requisitos',
                    'antecesoras', 'prec', 'predecesoras inmediatas',
                    'precedencias'):
            col_map[c] = 'Predecesoras'
    df.rename(columns=col_map, inplace=True)

    for col in ['Actividad', 'Duración', 'Predecesoras']:
        if col not in df.columns:
            raise ValueError(
                f"Columna '{col}' no encontrada. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    # Eliminar filas completamente vacías
    df = df.dropna(how='all').reset_index(drop=True)
    df = df[df['Actividad'].astype(str).str.strip() != ''].reset_index(drop=True)

    df['Actividad'] = df['Actividad'].astype(str).str.strip().str.upper()
    df['Duración'] = pd.to_numeric(df['Duración'], errors='coerce').fillna(0).clip(lower=0)

    # Función local para parsear predecesoras (usada en ambos branches)
    def parse_preds(s):
        if pd.isna(s):
            return []
        s = str(s).strip()
        if s in ('', '-', 'nan', 'none', 'ninguna', 'n/a', '—', '–', '0'):
            return []
        parts = re.split(r'[,;/]+|\s+y\s+|\s*&\s*', s)
        if len(parts) == 1 and ' ' in s.strip():
            cand = s.strip().split()
            if all(len(p.strip()) <= 3 for p in cand):
                parts = cand
        result, seen = [], set()
        for p in parts:
            p = p.strip().upper()
            if p and p not in seen and p not in ('', '-', 'NINGUNA',
                                                  'NONE', 'N/A'):
                seen.add(p)
                result.append(p)
        return result

    # Detectar formato con '<'
    use_successors = any(
        '<' in str(v) for v in df['Predecesoras'] if pd.notna(v) and str(v).strip()
    )

    if use_successors:
        succ_map = defaultdict(list)
        direct_preds = defaultdict(list)  # para filas sin '<'
        for _, row in df.iterrows():
            act = row['Actividad']
            s = str(row['Predecesoras']).strip()
            if pd.isna(s) or s in ('', '-', 'nan', 'none', 'ninguna'):
                continue
            if '<' in s:
                # formato sucesores o "predecesor(es) < sucesor(es)"
                src_str, succs_str = s.split('<', 1)
                # Dividir el lado izquierdo en múltiples predecesores
                src_parts = [p.strip().upper() for p in re.split(r'[,;/]+', src_str)]
                src_parts = [p for p in src_parts if p]
                # Dividir el lado derecho en sucesores
                succs = [p.strip().upper() for p in re.split(r'[,;/]+', succs_str)]
                succs = [p for p in succs if p]

                for src in src_parts:
                    for succ in succs:
                        if src == succ:
                            warning_list.append(
                                f"Auto-precedencia '{src} < {succ}' detectada en fila '{act}' — ignorada."
                            )
                            continue
                        succ_map[succ].append(src)
            else:
                # fila sin '<': interpretar como predecesoras tradicionales
                preds = parse_preds(s)
                direct_preds[act].extend(preds)
        # fusionar succ_map y direct_preds
        df['Predecesoras'] = df['Actividad'].apply(
            lambda a: list(set(succ_map.get(a, []) + direct_preds.get(a, [])))
        )
    else:
        df['Predecesoras'] = df['Predecesoras'].apply(parse_preds)

    # Resolver duplicados
    seen = {}
    new_names = []
    for name in df['Actividad']:
        if name in seen:
            seen[name] += 1
            new_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            new_names.append(name)
    df['Actividad'] = new_names

    # Validación de duración cero solo para filas con actividad declarada
    for _, row in df.iterrows():
        act = str(row.get('Actividad', '')).strip()
        if not act:
            continue   # fila vacía, no avisar
        if row.get('Duración', 0) == 0:
            warning_list.append(
                f"Actividad '{act}' tiene duración 0 — comportamiento equivalente a un dummy."
            )
    return df, warning_list


# ================================================================
# 2. ALGORITMO AON → AOA — Predecessor Signature (mínimo dummies)
# ================================================================
def build_aoa(df):
    """
    Convierte AON a AOA usando 'firma de predecesoras':
      - Mismo conjunto de predecesoras → mismo evento de inicio
      - Actividad presente en una sola firma → termina en ese evento
      - Actividad terminal → termina directamente en el sink (sin dummies)
      - No se crean dummies por paralelismo (se dibujarán flechas múltiples)
    Resultado: eventos mínimos, dummies solo para precedencias.
    """
    warnings = []
    df, _ = parse_table(df, warnings)               # <-- GRUPO B: descarte explícito
    activities = df['Actividad'].tolist()
    durations = dict(zip(activities, df['Duración']))
    pred_dict = dict(zip(activities, df['Predecesoras']))

    # ---- Validación y limpieza de predecesoras ----
    for a in activities:
        valid = []
        for p in pred_dict[a]:
            if p == a:
                warnings.append(f"'{a}' se precede a sí misma — ignorado.")
            elif p in activities:
                valid.append(p)
            else:
                match = _fuzzy_match(p, activities)
                if match:
                    warnings.append(f"'{p}' no existe en {a}. Usando '{match}'.")
                    valid.append(match)
                else:
                    warnings.append(f"Predecesora '{p}' de '{a}' no existe.")
        seen = set()
        pred_dict[a] = [p for p in valid if not (p in seen or seen.add(p))]

    # ---- Grafo AON, ruptura de ciclos, reducción transitiva ----
    G_aon = nx.DiGraph()
    G_aon.add_nodes_from(activities)
    for a in activities:
        for p in pred_dict[a]:
            G_aon.add_edge(p, a)

    cycles_broken = 0
    while not nx.is_directed_acyclic_graph(G_aon) and cycles_broken < 100:
        try:
            cycle = nx.find_cycle(G_aon, orientation='original')
            u, v, _ = cycle[-1]
            G_aon.remove_edge(u, v)
            warnings.append(f"Ciclo: eliminada arista {u} → {v}")
            cycles_broken += 1
        except nx.NetworkXNoCycle:
            break

    # Reducción transitiva (solo aristas con camino alterno)
    for u, v in list(G_aon.edges()):
        G_temp = G_aon.copy()
        G_temp.remove_edge(u, v)
        if nx.has_path(G_temp, u, v):
            warnings.append(
                f"Precedencia '{u} → {v}' es redundante (ya implícita vía otro camino) — eliminada."
            )
            G_aon.remove_edge(u, v)

    pred_dict = {a: sorted(G_aon.predecessors(a)) for a in activities}
    succ_dict = {a: list(G_aon.successors(a)) for a in activities}

    # ---- PASO 1: firma de predecesoras → evento de inicio ----
    pred_set_to_event = {frozenset(): 1}   # source = evento 1
    next_event = 2
    for a in nx.topological_sort(G_aon):
        ps = frozenset(pred_dict[a])
        if ps not in pred_set_to_event:
            pred_set_to_event[ps] = next_event
            next_event += 1

    activity_start = {a: pred_set_to_event[frozenset(pred_dict[a])]
                      for a in activities}

    # ---- PASO 2: evento de fin de cada actividad ----
    pred_sets_with = defaultdict(list)
    for ps in pred_set_to_event:
        for member in ps:
            pred_sets_with[member].append(ps)

    activity_end = {}
    multi_dummies = []
    for a in activities:
        containing = pred_sets_with.get(a, [])
        if len(containing) == 0:
            # Terminal → al sink (sin eventos intermedios)
            activity_end[a] = None
        elif len(containing) == 1:
            # Caso común: termina exactamente en el evento donde
            # arrancan sus sucesoras (sin dummy)
            activity_end[a] = pred_set_to_event[containing[0]]
        else:
            # Multi-firma: escoger la firma MÁS PEQUEÑA como
            # primaria (el destino más específico de la actividad),
            # NO crear evento nuevo. Las otras se alcanzan con un
            # único dummy desde la primaria.
            primary = min(
                containing,
                key=lambda ps: (len(ps), sorted(ps))
            )
            activity_end[a] = pred_set_to_event[primary]
            for ps in containing:
                if ps == primary:
                    continue
                multi_dummies.append(
                    (activity_end[a], pred_set_to_event[ps])
                )

    # Sink único para todos los terminales
    sink_event = next_event
    next_event += 1
    for a in activities:
        if activity_end[a] is None:
            activity_end[a] = sink_event

    # ---- PASO 3: arcos reales ----
    arcs = []
    for a in activities:
        arcs.append({
            'from': activity_start[a],
            'to':   activity_end[a],
            'activity': a,
            'duration': durations[a],
            'is_dummy': False
        })

    # ---- PASO 4: dummies para actividades multi-firma ----
    seen_dummy = set()
    for f, t in multi_dummies:
        if (f, t) in seen_dummy or f == t:
            continue
        seen_dummy.add((f, t))
        arcs.append({'from': f, 'to': t, 'activity': '',
                     'duration': 0, 'is_dummy': True})

    # ---- PASO 4.5 (NUEVO): unicidad de (i,j) para actividades reales ----
    groups = defaultdict(list)
    real_indices = [i for i, a in enumerate(arcs) if not a['is_dummy']]
    for i in real_indices:
        a = arcs[i]
        groups[(a['from'], a['to'])].append(i)

    max_event = max(max(a['from'], a['to']) for a in arcs) if arcs else 1
    new_event_counter = max_event + 1
    auxiliary_events = set()
    for (f, t), idx_list in groups.items():
        if len(idx_list) > 1:
            for i in idx_list[1:]:
                original_to = arcs[i]['to']
                k = new_event_counter
                new_event_counter += 1
                auxiliary_events.add(k)
                arcs[i]['to'] = k
                arcs.append({
                    'from': k,
                    'to': original_to,
                    'activity': '',
                    'duration': 0.0,
                    'is_dummy': True
                })
    next_event = new_event_counter

    # ---- PASO 5: eliminar dummies redundantes ----
    arcs = _remove_redundant_dummies(arcs)

    # ---- PASO 6: renumerar eventos en orden topológico ----
    G_event = nx.DiGraph()
    for arc in arcs:
        G_event.add_node(arc['from'])
        G_event.add_node(arc['to'])
        G_event.add_edge(arc['from'], arc['to'])

    try:
        topo_events = list(nx.topological_sort(G_event))
    except nx.NetworkXUnfeasible:
        warnings.append("Error: topología inconsistente.")
        return arcs, {}, {}, warnings, set()

    renumber = {e: i + 1 for i, e in enumerate(topo_events)}
    for arc in arcs:
        arc['from'] = renumber[arc['from']]
        arc['to']   = renumber[arc['to']]

    auxiliary_events = {renumber[e] for e in auxiliary_events if e in renumber}

    # ---- PASO 7: CPM (forward + backward pass) ----
    G_cpm = nx.DiGraph()
    for arc in arcs:
        if G_cpm.has_edge(arc['from'], arc['to']):
            if arc['duration'] > G_cpm[arc['from']][arc['to']]['duration']:
                G_cpm[arc['from']][arc['to']]['duration'] = arc['duration']
        else:
            G_cpm.add_edge(arc['from'], arc['to'], duration=arc['duration'])

    if G_cpm.number_of_nodes() == 0:
        return arcs, {}, {}, warnings, auxiliary_events

    topo = list(nx.topological_sort(G_cpm))
    event_es = {n: 0 for n in G_cpm.nodes()}
    for n in topo:
        for p in G_cpm.predecessors(n):
            t = event_es[p] + G_cpm[p][n]['duration']
            if t > event_es[n]:
                event_es[n] = t
    project_dur = max(event_es.values()) if event_es else 0

    event_lf = {n: project_dur for n in G_cpm.nodes()}
    for n in reversed(topo):
        for s in G_cpm.successors(n):
            t = event_lf[s] - G_cpm[n][s]['duration']
            if t < event_lf[n]:
                event_lf[n] = t

    for arc in arcs:
        f, t = arc['from'], arc['to']
        arc['is_critical'] = (
            f in event_es and t in event_es and
            abs(event_es[f] - event_lf[f]) < 1e-6 and
            abs(event_es[t] - event_lf[t]) < 1e-6 and
            abs((event_es[t] - event_es[f]) - arc['duration']) < 1e-6
        )

    return arcs, event_es, event_lf, warnings, auxiliary_events


def _remove_redundant_dummies(arcs):
    """
    Un dummy u→v es redundante solo si existe otra ruta u→...→v formada
    exclusivamente por dummies (duración 0). Si hay alguna actividad real
    en el camino alternativo, el dummy se conserva.
    Además, deduplica dummies con el mismo (u, v); son intercambiables.
    """
    seen_dummy_pairs = set()
    deduped = []
    for a in arcs:
        if a['is_dummy']:
            key = (a['from'], a['to'])
            if key in seen_dummy_pairs:
                continue
            seen_dummy_pairs.add(key)
        deduped.append(a)
    arcs = deduped

    dummy_indices = [i for i, a in enumerate(arcs) if a['is_dummy']]
    G_dummy = nx.DiGraph()
    for i, a in enumerate(arcs):
        if a['is_dummy']:
            G_dummy.add_edge(a['from'], a['to'])
    to_remove = set()
    for i in dummy_indices:
        u, v = arcs[i]['from'], arcs[i]['to']
        G_dummy.remove_edge(u, v)
        if nx.has_path(G_dummy, u, v):
            to_remove.add(i)
        else:
            G_dummy.add_edge(u, v)
    return [a for i, a in enumerate(arcs) if i not in to_remove]


def _fuzzy_match(name, candidates):
    """Coincidencia aproximada para typos comunes (solo para nombres >=3 caracteres)."""
    nl = str(name).lower().strip()
    if not nl:
        return None
    for c in candidates:
        cl = str(c).lower().strip()
        if nl == cl:
            return c
        if len(nl) <= 2 or len(cl) <= 2:
            continue
        if abs(len(nl) - len(cl)) <= 1:
            diffs = sum(1 for a, b in zip(nl, cl) if a != b)
            diffs += abs(len(nl) - len(cl))
            if diffs <= 1:
                return c
    return None


# ================================================================
# 3. CÁLCULO DE TABLA CPM Y RUTA CRÍTICA
# ================================================================
def compute_activity_cpm(arcs, event_es, event_lf, include_dummies=True):
    """Genera la tabla CPM por actividad. Si include_dummies=True, incluye las ficticias."""
    timing = []
    for arc in arcs:
        if not include_dummies and arc['is_dummy']:
            continue
        if arc['is_dummy']:
            nombre = "(dummy)"
            dur = 0
        else:
            nombre = arc['activity']
            dur = arc['duration']
        es = event_es.get(arc['from'], 0)
        ef = es + dur
        lf = event_lf.get(arc['to'], 0)
        ls = lf - dur
        timing.append({
            'Actividad': nombre,
            'Duración': _fmt_num(dur),
            'Evento i': arc['from'],
            'Evento j': arc['to'],
            'ES': _fmt_num(es),
            'EF': _fmt_num(ef),
            'LS': _fmt_num(ls),
            'LF': _fmt_num(lf),
            'Holgura Total': _fmt_num(ls - es),
            'Crítica': '🔴 SÍ' if arc['is_critical'] else ''
        })
    timing.sort(key=lambda x: (x['Actividad'] != '(dummy)', x['Actividad']))
    return timing


def find_critical_path(arcs):
    """Devuelve todas las rutas críticas (puede haber más de una)."""
    crit_arcs = [a for a in arcs if a['is_critical']]
    if not crit_arcs:
        return []
    G = nx.MultiDiGraph()
    for a in crit_arcs:
        G.add_edge(a['from'], a['to'], activity=a['activity'],
                   duration=a['duration'], is_dummy=a['is_dummy'])
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    paths = []
    G_simple = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if not G_simple.has_edge(u, v):
            G_simple.add_edge(u, v, **data)
    for s in sources:
        for t in sinks:
            try:
                for path in nx.all_simple_paths(G_simple, s, t):
                    seq, total = [], 0
                    for i in range(len(path) - 1):
                        edge = G_simple[path[i]][path[i + 1]]
                        if not edge['is_dummy']:
                            seq.append(edge['activity'])
                        total += edge['duration']
                    paths.append((path, seq, total))
            except nx.NetworkXError:
                pass
    return paths


# ================================================================
# 4. VISUALIZACIÓN (Graphviz, estilo industrial)
# ================================================================
def draw_aoa_network(arcs, event_es, event_lf, auxiliary_events=None):
    """
    Dibuja la red AOA usando Graphviz (motor dot).
    Fallback a un mensaje si Graphviz no está disponible.
    """
    plt.close('all')   # liberar figuras previas

    if not GRAPHVIZ_AVAILABLE:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                "Graphviz no instalado en el entorno Python.\n"
                "Instalá con: pip install graphviz",
                ha='center', va='center', fontsize=12, color='red',
                transform=ax.transAxes)
        ax.axis('off')
        return fig

    if not arcs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center')
        ax.axis('off')
        return fig

    auxiliary_events = auxiliary_events or set()

    # Determinar eventos críticos
    crit_events = set()
    for arc in arcs:
        if arc.get('is_critical'):
            crit_events.add(arc['from'])
            crit_events.add(arc['to'])

    # Construir Digraph
    dot = graphviz.Digraph(
        name='PERT_AOA',
        strict=False,
        format='png'
    )
    # ------------------------------------------------
    # BUG #1: Etiquetas cortadas - añadir márgenes y evitar recortes
    dot.attr(
        rankdir='LR',
        splines='spline',
        nodesep='0.5',
        ranksep='0.8',
        margin='0.7',           # margen exterior del grafo (pulgadas)
        pad='0.4',              # padding adicional alrededor del canvas
        overlap='false' ,          # evita solapes residuales
        dpi='300'
    )

    dot.node_attr['style'] = 'filled'
    dot.node_attr['fontname'] = 'Helvetica'
    dot.node_attr['fontsize'] = '11'
    dot.node_attr['fontcolor'] = '#222'
    # Espacio interno extra en los nodos para que las etiquetas de aristas no queden pegadas al borde
    dot.node_attr['margin'] = '0.15,0.1'
    # ------------------------------------------------

    # Nodos
    all_events = set()
    for arc in arcs:
        all_events.add(arc['from'])
        all_events.add(arc['to'])

    for ev in sorted(all_events):
        is_aux = ev in auxiliary_events
        is_crit = ev in crit_events
        if is_aux:
            dot.node(
                str(ev),
                label=str(ev),
                shape='circle',
                width='0.3',
                height='0.3',
                fillcolor='white',
                color='#9E9E9E',
                style='dashed,filled',
                penwidth='1.0',
                fontsize='9'
            )
        elif is_crit:
            dot.node(
                str(ev),
                label=str(ev),
                shape='circle',
                fillcolor='#FFCCBC',
                color='#E53935',
                penwidth='2.5',
                fontsize='11'
            )
        else:
            dot.node(
                str(ev),
                label=str(ev),
                shape='circle',
                fillcolor='#FFE0B2',
                color='#FB8C00',
                penwidth='1.5',
                fontsize='11'
            )

    # Arcos
    for arc in arcs:
        f = str(arc['from'])
        t = str(arc['to'])
        is_dummy = arc['is_dummy']
        is_crit = arc['is_critical']

        if is_dummy:
            if is_crit:
                color = '#B71C1C'
                penwidth = '1.5'
                fontcolor = '#B71C1C'
            else:
                color = '#9E9E9E'
                penwidth = '1.2'
                fontcolor = '#9E9E9E'
            label = '<<I>ficticia</I>>'
            fontsize = '8'
        else:
            if is_crit:
                color = '#E53935'
                penwidth = '2.5'
            else:
                color = '#1976D2'
                penwidth = '1.8'
            dur_str = str(_fmt_num(arc['duration']))
            label = f"{arc['activity']}, {dur_str}"
            fontcolor = color
            fontsize = '10'

        dot.edge(
            f, t,
            label=label,
            fontcolor=fontcolor,
            fontname='Helvetica',
            fontsize=fontsize,
            color=color,
            penwidth=penwidth,
            arrowhead='vee',
            style='dashed' if is_dummy else 'solid'
        )

    # Renderizar a PNG en memoria
    png_bytes = None
    svg_bytes = None
    try:
        png_bytes = dot.pipe(format='png')
    except Exception as e:
        err_name = type(e).__name__
        if 'ExecutableNotFound' in err_name or isinstance(e, FileNotFoundError):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5,
                    "Graphviz no se encuentra en el sistema.\n"
                    "Instalá Graphviz:\n"
                    "  • Linux: sudo apt install graphviz\n"
                    "  • macOS: brew install graphviz\n"
                    "  • Windows: descargá el instalador de graphviz.org\n"
                    "  y asegurate de que esté en el PATH.",
                    ha='center', va='center', fontsize=11, color='red',
                    transform=ax.transAxes)
            ax.axis('off')
            return fig
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5,
                    f"Error al renderizar con Graphviz:\n{e}",
                    ha='center', va='center', fontsize=11, color='red',
                    transform=ax.transAxes)
            ax.axis('off')
            return fig

    # Crear figura matplotlib con el PNG original
    img = mpimg.imread(BytesIO(png_bytes))
    h, w = img.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout(pad=0)

    # Guardar PNG original para descarga sin pérdida
    fig._graphviz_png_bytes = png_bytes

    # Intentar obtener SVG para descarga vectorial
    try:
        svg_bytes = dot.pipe(format='svg')
        fig._graphviz_svg_bytes = svg_bytes
    except Exception:
        fig._graphviz_svg_bytes = None

    return fig


# ================================================================
# 5. EXPORTACIÓN A EXCEL (todo en un archivo)
# ================================================================
def export_to_excel(timing, event_es, event_lf, project_dur, paths, fig):
    """Genera un Excel con: tabla CPM, eventos, ruta crítica e imagen."""
    output = BytesIO()

    ev_data = []
    for ev in sorted(event_es.keys()):
        slack = event_lf[ev] - event_es[ev]
        ev_data.append({
            'Evento': ev,
            'Tiempo más temprano (ES)': _fmt_num(event_es[ev]),
            'Tiempo más tardío (LF)': _fmt_num(event_lf[ev]),
            'Holgura del evento': _fmt_num(slack),
            'Crítico': 'Sí' if abs(slack) < 1e-6 else 'No'
        })

    resumen = pd.DataFrame([
        {'Métrica': 'Duración total del proyecto', 'Valor': _fmt_num(project_dur)},
        {'Métrica': 'Número de actividades', 'Valor': len(timing)},
        {'Métrica': 'Actividades críticas',
         'Valor': len([t for t in timing if t['Crítica']])},
        {'Métrica': 'Cantidad de rutas críticas', 'Valor': len(paths)},
    ])

    rutas_data = []
    for i, (ev_path, act_seq, total) in enumerate(paths, 1):
        rutas_data.append({
            'Ruta #': i,
            'Eventos': ' → '.join(map(str, ev_path)),
            'Actividades': ' → '.join(act_seq) if act_seq else '(solo dummies)',
            'Duración': _fmt_num(total)
        })

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(timing).to_excel(writer, sheet_name='Tabla CPM', index=False)
        pd.DataFrame(ev_data).to_excel(writer, sheet_name='Eventos', index=False)
        if rutas_data:
            pd.DataFrame(rutas_data).to_excel(writer, sheet_name='Rutas críticas', index=False)
        resumen.to_excel(writer, sheet_name='Resumen', index=False)

    return output.getvalue()


# ================================================================
# 6. INTERFAZ STREAMLIT — OBSIDIAN ANALYTICS PREMIUM (v5)
# ================================================================
def main():
    st.set_page_config(
        page_title="Analítica PERT/CPM",
        layout="wide",
        page_icon="📊"
    )

    # =========================================================================
    # CSS — OBSIDIAN ANALYTICS PREMIUM
    # =========================================================================
    st.markdown("""
    <style>
    /* ── CONTENEDOR PRINCIPAL — quitar padding excesivo ────────────────── */
    .block-container {
        padding-top:    1rem    !important;
        padding-bottom: 1rem    !important;
        padding-left:   1.2rem  !important;
        padding-right:  1.2rem  !important;
        max-width:      100%    !important;
    }
    /* Quitar margen superior del header de Streamlit */
    [data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
    }
    /* ── PALETA OBSIDIAN ────────────────────────────────────────────────── */
    :root {
        --obs-indigo:        #6366F1;
        --obs-indigo-dk:     #4F46E5;
        --obs-indigo-soft:   rgba(99,102,241,0.12);
        --obs-indigo-glow:   rgba(99,102,241,0.30);
        --obs-cyan:          #06B6D4;
        --obs-cyan-soft:     rgba(6,182,212,0.10);
        --obs-cyan-glow:     rgba(6,182,212,0.24);
        --obs-purple:        #A855F7;
        --obs-purple-soft:   rgba(168,85,247,0.10);
        --obs-emerald:       #10B981;
        --obs-red:           #EF4444;
        --obs-red-soft:      rgba(239,68,68,0.12);
        --obs-amber:         #F59E0B;
        --obs-card:          rgba(255,255,255,0.03);
        --obs-card-border:   rgba(99,102,241,0.18);
        --obs-muted:         #94A3B8;
        --obs-dim:           #64748B;
    }

    /* ── TARJETAS PANEL ─────────────────────────────────────────────────── */
    .obs-card {
        background: var(--obs-card);
        border: 1px solid var(--obs-card-border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.7rem;
        transition: border-color .25s, box-shadow .25s;
    }
    .obs-card:hover {
        border-color: rgba(99,102,241,0.40);
        box-shadow: 0 0 22px rgba(99,102,241,0.08);
    }
    .obs-card-title {
        font-size: 0.67rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: var(--obs-indigo);
        margin-bottom: 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .obs-card-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(99,102,241,0.38), transparent);
        margin-left: 0.3rem;
    }
    .obs-divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(99,102,241,0.22), transparent);
        margin: 0.7rem 0;
    }

    /* ── MÉTRICAS ───────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--obs-card);
        border: 1px solid var(--obs-card-border);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        position: relative;
        overflow: hidden;
        transition: transform .2s, box-shadow .2s;
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--obs-indigo), var(--obs-cyan), var(--obs-purple));
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px var(--obs-indigo-glow);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.67rem !important;
        font-weight: 700 !important;
        opacity: 0.55;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        letter-spacing: -.02em;
    }

    /* ── BOTONES BASE ───────────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: .58rem 1.3rem;
        transition: all .2s cubic-bezier(.4,0,.2,1);
        font-size: .91rem;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--obs-indigo-dk), var(--obs-indigo)) !important;
        border: none !important;
        color: #fff !important;
        box-shadow: 0 4px 16px var(--obs-indigo-glow);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 26px var(--obs-indigo-glow);
    }
    .stButton > button:active { transform: translateY(0); }

    /* ── TABS ───────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: .4rem;
        border-bottom: 2px solid rgba(99,102,241,0.15);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: .62rem 1.25rem;
        font-weight: 500;
        transition: all .2s;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--obs-indigo-soft);
        color: var(--obs-indigo) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--obs-indigo) !important;
        border-bottom: 3px solid var(--obs-indigo) !important;
        font-weight: 700;
    }

    /* ── SIDEBAR OCULTO (usamos columnas propias) ───────────────────────── */
    [data-testid="stSidebar"] { display: none; }

    /* ── TABLAS ─────────────────────────────────────────────────────────── */
    [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(99,102,241,0.18) !important;
    }

    /* ── CÓDIGO INLINE ──────────────────────────────────────────────────── */
    code {
        background: rgba(99,102,241,0.10) !important;
        color: #A5B4FC !important;
        padding: .13rem .38rem;
        border-radius: 4px;
        font-size: .87em;
        font-weight: 500;
        border: 1px solid rgba(99,102,241,0.18);
    }

    /* ── ALERTAS ────────────────────────────────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: 10px;
        border-left-width: 4px;
    }

    /* ── BADGE DE MODO ──────────────────────────────────────────────────── */
    .mode-badge {
        display: inline-flex;
        align-items: center;
        gap: .3rem;
        padding: .22rem .75rem;
        border-radius: 99px;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .04em;
        vertical-align: middle;
    }
    .mode-badge.prob {
        background: rgba(6,182,212,0.12);
        border: 1px solid rgba(6,182,212,0.40);
        color: #67E8F9;
    }
    .mode-badge.det {
        background: rgba(99,102,241,0.10);
        border: 1px solid rgba(99,102,241,0.30);
        color: #A5B4FC;
    }

    /* ── BADGE ESTATUS ──────────────────────────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: .35rem;
        padding: .28rem .9rem;
        border-radius: 8px;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .03em;
    }
    .status-found {
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.35);
        color: #34D399;
    }
    .status-pending {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.22);
        color: #A5B4FC;
    }

    /* ── SUPRIMIR ERROR removeChild (bug React + ag-grid en Streamlit) ─── */
    /* El app se recupera solo; este CSS colapsa el cuadro antes de verlo  */
    [data-testid="stException"],
    [class*="ExceptionMessage"],
    div[data-baseweb="notification"][kind="negative"] {
        display:    none !important;
        height:     0    !important;
        overflow:   hidden !important;
        padding:    0    !important;
        margin:     0    !important;
        border:     none !important;
        opacity:    0    !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================================================================
    # HERO HEADER — OBSIDIAN ANALYTICS
    # =========================================================================
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(99,102,241,0.09) 0%,'
        'rgba(6,182,212,0.05) 55%,transparent 100%);'
        'border-radius:18px;padding:1.8rem 2.2rem 1.4rem;margin-bottom:1.4rem;'
        'border:1px solid rgba(99,102,241,0.22);position:relative;overflow:hidden;">'
        '<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        'background:linear-gradient(90deg,#6366F1 0%,#06B6D4 50%,#A855F7 100%);'
        'border-radius:18px 18px 0 0;"></div>'
        '<div style="display:flex;align-items:flex-end;justify-content:space-between;'
        'flex-wrap:wrap;gap:.8rem;">'
        '<div>'
        '<div style="display:inline-flex;align-items:center;gap:.5rem;padding:.25rem .85rem;'
        'background:rgba(99,102,241,0.14);border:1px solid rgba(99,102,241,0.35);color:#A5B4FC;'
        'border-radius:99px;font-size:.68rem;font-weight:700;letter-spacing:.10em;'
        'text-transform:uppercase;margin-bottom:.75rem;">📐 Investigación Operativa II</div>'
        '<h1 style="margin:0;font-size:2.2rem;font-weight:900;line-height:1.08;letter-spacing:-.03em;'
        'background:linear-gradient(135deg,#ffffff 25%,#A5B4FC 100%);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">'
        'Analítica '
        '<span style="background:linear-gradient(135deg,#6366F1 0%,#06B6D4 100%);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">'
        'PERT/CPM</span></h1>'
        '<p style="margin:.5rem 0 0;font-size:.88rem;color:rgba(255,255,255,0.50);line-height:1.5;">'
        'Notación Activity-On-Arrow &nbsp;·&nbsp; Construcción automática &nbsp;·&nbsp; Cálculo de ruta crítica'
        '</p></div>'
        '<div style="display:flex;gap:.55rem;flex-wrap:wrap;align-items:center;">'
        '<span style="padding:.25rem .8rem;background:rgba(16,185,129,0.10);'
        'border:1px solid rgba(16,185,129,0.28);border-radius:8px;font-size:.68rem;'
        'font-weight:600;color:#34D399;">● Tablero Principal</span>'
        '<span style="padding:.25rem .8rem;background:rgba(99,102,241,0.10);'
        'border:1px solid rgba(99,102,241,0.28);border-radius:8px;font-size:.68rem;'
        'font-weight:600;color:#A5B4FC;">AOA v5</span>'
        '</div></div></div>',
        unsafe_allow_html=True
    )

    # =========================================================================
    # SUPRESOR DE ERROR removeChild (bug conocido Streamlit + ag-grid)
    # Inyecta un MutationObserver que oculta el cuadro de error en <1 ms
    # =========================================================================
    components.html(
        """
        <script>
        (function () {
            var SELECTORS = [
                '[data-testid="stException"]',
                '[class*="ExceptionMessage"]',
                '[class*="stException"]',
                'div[data-baseweb="notification"]'
            ];
            function _hide(root) {
                SELECTORS.forEach(function (sel) {
                    root.querySelectorAll(sel).forEach(function (n) {
                        var txt = n.innerText || n.textContent || '';
                        if (txt.indexOf('removeChild') > -1) {
                            n.style.cssText = 'display:none!important;height:0!important;overflow:hidden!important;opacity:0!important;';
                        }
                    });
                });
            }
            try {
                new MutationObserver(function () { _hide(parent.document); })
                    .observe(parent.document.body, { childList: true, subtree: true });
                _hide(parent.document);
            } catch (e) {
                /* parent.document inaccesible — la CSS del bloque <style> actúa de respaldo */
            }
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )

    # =========================================================================
    # SESSION STATE
    # =========================================================================
    for _k, _v in [('example_loaded', False), ('results', None),
                    ('editor_version', 0), ('prob_mode', False)]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # =========================================================================
    # TRES COLUMNAS PRINCIPALES
    # =========================================================================
    col_izq, col_centro, col_der = st.columns([0.75, 4.2, 1.1], gap="medium")

    # ─────────────────────────────────────────────────────────────────────────
    # COLUMNA IZQUIERDA — Fuentes de Datos + Configuración del Algoritmo
    # ─────────────────────────────────────────────────────────────────────────
    with col_izq:

        # ── Fuentes de Datos ──────────────────────────────────────────────────
        st.markdown('**🗂 FUENTES DE DATOS**')
        input_mode = st.radio(
            "Fuente de datos",
            ["Tabla manual", "Subir archivo"],
            horizontal=False,
            label_visibility="collapsed"
        )

        st.divider()

        # ── Configuración del Algoritmo ───────────────────────────────────────
        st.markdown('**⚙ CONFIGURACIÓN DEL ALGORITMO**')
        example_loaded = st.session_state.get('example_loaded', False)
        n_rows = st.number_input(
            "Número de actividades",
            min_value=2, max_value=80, value=6,
            disabled=example_loaded,
            help="Filas iniciales. Tras cargar un ejemplo, usá +/− en la tabla."
        )
        if example_loaded:
            st.caption("Ejemplo cargado. Usá +/− en la tabla o **Limpiar todo** para reiniciar.")

        # Forzar recreación del editor si cambia n_rows
        prev_n_rows = st.session_state.get('prev_n_rows', n_rows)
        if prev_n_rows != n_rows and not st.session_state.get('example_loaded', False):
            st.session_state.editor_version += 1
            st.session_state.prev_n_rows = n_rows

        if input_mode == "Tabla manual":
            st.divider()
            _is_prob   = st.session_state.get('prob_mode', False)
            label_prob = "🔵 Modo Determinístico" if _is_prob else "📊 Modo Probabilístico"
            if st.button(label_prob, use_container_width=True, key="sidebar_prob_btn"):
                st.session_state.prob_mode = not _is_prob
                st.session_state.results   = None
                st.session_state.pop('prob_summary', None)
                st.session_state.editor_version += 1

        st.divider()

        # ── Formatos aceptados ────────────────────────────────────────────────
        st.markdown('**📋 FORMATOS ACEPTADOS**')
        st.markdown(
            '**Predecesoras:** `A, B, C`  \n'
            '**Sucesoras:** `A < B, C`  \n'
            'Sin predecesoras: vacío o `-`'
        )

        st.divider()

        # ── Info algoritmo ────────────────────────────────────────────────────
        st.caption("🔬 **Algoritmo:** Predecessor-signature · Reducción transitiva incluida")

        # ── Limpiar todo ──────────────────────────────────────────────────────
        if st.button("🗑 Limpiar todo", use_container_width=True, type="secondary"):
            st.session_state.example_loaded = False
            st.session_state.results        = None
            st.session_state.prob_mode      = False
            st.session_state.pop('prob_summary', None)
            st.session_state.editor_version += 1
            st.session_state.prev_n_rows    = n_rows
            st.session_state.pop('last_hash', None)

    # ─────────────────────────────────────────────────────────────────────────
    # COLUMNA CENTRAL — Tabla de Configuración de Actividades
    # ─────────────────────────────────────────────────────────────────────────
    with col_centro:

        edited_df = None

        if input_mode == "Subir archivo":
            st.markdown("**📂 Subir Archivo de Datos**")
            uploaded = st.file_uploader(
                "Sube CSV o Excel", type=['csv', 'xlsx', 'xls'],
                label_visibility="collapsed"
            )
            if uploaded:
                try:
                    if uploaded.name.endswith('.csv'):
                        edited_df = pd.read_csv(uploaded)
                    else:
                        edited_df = pd.read_excel(uploaded)
                    required_cols  = {'Actividad', 'Duración', 'Predecesoras'}
                    df_cols_lower  = {c.lower().strip(): c for c in edited_df.columns}
                    missing = []
                    for req in required_cols:
                        found = any(
                            v in df_cols_lower
                            for v in [req.lower(), req.lower().replace('ó', 'o'),
                                      'activity', 'duration', 'predecessors',
                                      'precedencia', 'predecesores']
                        )
                        if not found:
                            missing.append(req)
                    if missing:
                        st.error(f"Faltan columnas: {', '.join(missing)}. El archivo debe tener: Actividad, Duración, Predecesoras.")
                        return
                except Exception as e:
                    st.error(f"Error al leer el archivo: {e}")
                    return
            else:
                st.info("Sube un archivo CSV o Excel para comenzar.")
                return

        else:
            # ── Cabecera de la tabla ───────────────────────────────────────────
            if st.session_state.prob_mode:
                st.markdown("**📊 Tabla de Configuración de Actividades** — `Probabilístico`")
            else:
                st.markdown("**📋 Tabla de Configuración de Actividades** — `Determinístico`")

            if st.button("⚡ Cargar ejemplo", use_container_width=True):
                st.session_state.example_loaded = True
                st.session_state.editor_version += 1
                st.session_state.results = None

            if st.session_state.prob_mode:
                # ── MODO PROBABILÍSTICO ────────────────────────────────────────
                if st.session_state.example_loaded:
                    prob_default = {
                        'Actividad':    ['A',  'B',  'C',  'D',  'E',         'F',  'G',  'H',    'I',    'J',     'K'],
                        'Predecesoras': ['',   '',   '',   '',   'A, B, C, D', 'D',  '',   'E, F', 'E, F', 'G, H', 'G, H, I'],
                        'Optimista':    [1.0,  0.5,  1.5,  4.0,  1.0,          6.0,  1.0,  1.0,    3.0,    1.0,     4.0],
                        'Realista':     [1.5,  1.0,  2.5,  6.0,  1.5,          7.0,  1.5,  1.5,    5.0,    1.5,     4.0],
                        'Pesimista':    [3.0,  1.5,  4.0, 10.0,  2.0,          9.0,  2.5,  3.0,   13.0,    3.0,     4.0],
                    }
                else:
                    actividades = [chr(65 + i) for i in range(n_rows)]
                    prob_default = {
                        'Actividad':    actividades,
                        'Predecesoras': [''] * n_rows,
                        'Optimista':    [None] * n_rows,
                        'Realista':     [None] * n_rows,
                        'Pesimista':    [None] * n_rows,
                    }

                df_prob = pd.DataFrame(prob_default)
                st.latex(r"t_e = \dfrac{a + 4m + b}{6}")
                st.latex(r"\sigma^2 = \dfrac{(b - a)^2}{36}")

                _prob_h = min(40 * len(df_prob) + 62, 560)
                edited_prob = st.data_editor(
                    df_prob, num_rows="fixed", use_container_width=True,
                    key=f"editor_{st.session_state.editor_version}", height=_prob_h,
                    column_config={
                        'Actividad':    st.column_config.TextColumn('ACTIVIDAD', width='small'),
                        'Predecesoras': st.column_config.TextColumn(
                            'PREDECESORAS', width='medium',
                            help="Predecesoras: 'A, B'  |  Sucesoras: 'A < B, C'"
                        ),
                        'Optimista': st.column_config.NumberColumn('a (Optimista)', min_value=0, width='small'),
                        'Realista':  st.column_config.NumberColumn('m (Realista)',  min_value=0, width='small'),
                        'Pesimista': st.column_config.NumberColumn('b (Pesimista)', min_value=0, width='small'),
                    }
                )

                # Cálculos PERT por fila
                a_col   = pd.to_numeric(edited_prob['Optimista'], errors='coerce').fillna(0)
                m_col   = pd.to_numeric(edited_prob['Realista'],  errors='coerce').fillna(0)
                b_col   = pd.to_numeric(edited_prob['Pesimista'], errors='coerce').fillna(0)
                te_col  = (a_col + 4 * m_col + b_col) / 6
                var_col = ((b_col - a_col) ** 2) / 36

                acts_col  = edited_prob['Actividad'].astype(str).str.strip().str.upper()
                valid_idx = acts_col[acts_col != ''].index
                if len(valid_idx) > 0:
                    summary_df = pd.DataFrame({
                        'Actividad':     acts_col[valid_idx].values,
                        'a (Optimista)': a_col[valid_idx].round(4).values,
                        'm (Realista)':  m_col[valid_idx].round(4).values,
                        'b (Pesimista)': b_col[valid_idx].round(4).values,
                        'tₑ (Esperado)': te_col[valid_idx].round(4).values,
                        'σ² (Varianza)': var_col[valid_idx].round(4).values,
                    })
                    st.session_state.prob_summary = summary_df
                    with st.expander("📊 Tiempos esperados y varianzas", expanded=False):
                        _hl_cols = ['tₑ (Esperado)', 'σ² (Varianza)']
                        st.dataframe(
                            summary_df.style.set_properties(
                                subset=_hl_cols,
                                **{'background-color': '#0D1B2A',
                                   'color': '#93C5FD', 'font-weight': '600'}
                            ),
                            use_container_width=True, hide_index=True
                        )

                # Construir edited_df para build_aoa (Duración = tₑ)
                edited_df = pd.DataFrame({
                    'Actividad':    edited_prob['Actividad'],
                    'Duración':     te_col,
                    'Predecesoras': edited_prob['Predecesoras'],
                })

            else:
                # ── MODO DETERMINÍSTICO ────────────────────────────────────────
                if st.session_state.example_loaded:
                    default_data = {
                        'Actividad':    ['A', 'B', 'C', 'D', 'E', 'F'],
                        'Duración':     [8, 4, 2, 10, 5, 3],
                        'Predecesoras': ['', '', 'A', 'A', 'B', 'C, E']
                    }
                else:
                    actividades = [chr(65 + i) for i in range(n_rows)]
                    default_data = {
                        'Actividad':    actividades,
                        'Duración':     [None] * n_rows,
                        'Predecesoras': [''] * n_rows
                    }

                df = pd.DataFrame(default_data)
                _det_h = min(40 * len(df) + 62, 560)
                edited_df = st.data_editor(
                    df, num_rows="fixed", use_container_width=True,
                    key=f"editor_{st.session_state.editor_version}", height=_det_h,
                    column_config={
                        'Actividad':    st.column_config.TextColumn('ACTIVIDAD', width='small'),
                        'Duración':     st.column_config.NumberColumn(
                            'DURACIÓN (DÍAS)', min_value=0, width='small'
                        ),
                        'Predecesoras': st.column_config.TextColumn(
                            'PREDECESORAS', width='medium',
                            help="Predecesoras: 'A, B'  |  Sucesoras: 'A < B, C'"
                        )
                    }
                )

        # ── Validación básica ──────────────────────────────────────────────────
        if edited_df is None or edited_df.empty:
            st.warning("La tabla está vacía. Agregá al menos una actividad.")
            return
        actividades_validas = edited_df['Actividad'].astype(str).str.strip()
        actividades_validas = actividades_validas[actividades_validas != '']
        if len(actividades_validas) == 0:
            st.warning("No hay actividades válidas. Completá al menos el nombre y la duración de una actividad.")
            return

        # ── Invalidación de resultados anteriores ─────────────────────────────
        def _df_hash(df):
            if df is None or df.empty:
                return ""
            try:
                return hashlib.md5(
                    pd.util.hash_pandas_object(df, index=True).values.tobytes()
                ).hexdigest()
            except Exception:
                return str(df.to_dict())

        current_hash = _df_hash(edited_df)
        if (st.session_state.results is not None
                and st.session_state.get('last_hash') != current_hash):
            old_fig = st.session_state.results.get('fig')
            if old_fig is not None:
                plt.close(old_fig)
            st.session_state.results = None
            st.info("La tabla cambió. Presioná **Generar red** para recalcular.")

        # ── BOTÓN GENERAR ──────────────────────────────────────────────────────
        if st.button("🚀 Generar red PERT/CPM", type="primary", use_container_width=True):
            with st.spinner("Construyendo red de actividades..."):
                try:
                    arcs, event_es, event_lf, warnings, auxiliary_events = build_aoa(edited_df)
                    timing      = compute_activity_cpm(arcs, event_es, event_lf, include_dummies=True)
                    paths       = find_critical_path(arcs)
                    project_dur = max(event_es.values()) if event_es else 0
                    fig         = draw_aoa_network(arcs, event_es, event_lf, auxiliary_events=auxiliary_events)

                    st.session_state.results = {
                        'arcs': arcs, 'event_es': event_es, 'event_lf': event_lf,
                        'warnings': warnings, 'auxiliary_events': auxiliary_events,
                        'timing': timing, 'paths': paths,
                        'project_dur': project_dur, 'fig': fig
                    }
                    st.session_state.last_hash = current_hash
                except Exception as e:
                    st.session_state.results = None
                    st.error(f"Error al construir la red: {e}")
                    import traceback
                    with st.expander("Detalle técnico"):
                        st.code(traceback.format_exc())
                    return

        # ── Red AOA — mostrar en columna central (llena el espacio) ───────────
        if st.session_state.results:
            _rc = st.session_state.results
            st.divider()
            _dur_c  = _rc['project_dur']
            _dur_cs = (f"{int(_dur_c)}" if abs(_dur_c - round(_dur_c)) < 1e-6
                       else f"{_dur_c:.2f}")
            st.success(f"**✅ Duración total del proyecto: {_dur_cs} unidades de tiempo**")
            if _rc['paths']:
                _cp_lbl = "🔴 Ruta crítica" if len(_rc['paths']) == 1 else "🔴 Rutas críticas"
                st.markdown(f"**{_cp_lbl}:**")
                for _i, (_evp, _acs, _tot) in enumerate(_rc['paths'], 1):
                    _ts  = (f"{int(_tot)}" if abs(_tot - round(_tot)) < 1e-6
                            else f"{_tot:.2f}")
                    _evs = " → ".join(str(e) for e in _evp)
                    _acs_str = " → ".join(_acs) if _acs else "(solo dummies)"
                    if len(_rc['paths']) > 1:
                        st.markdown(f"**Ruta {_i}:** Eventos `{_evs}` — Actividades **{_acs_str}** ⏱ {_ts} u.t.")
                    else:
                        st.markdown(f"Eventos `{_evs}` — **{_acs_str}** ⏱ {_ts} u.t.")
            st.pyplot(_rc['fig'], use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # COLUMNA DERECHA — Resumen de Visualización
    # ─────────────────────────────────────────────────────────────────────────
    with col_der:

        if st.session_state.results:
            r             = st.session_state.results
            arcs_r        = r['arcs']
            event_es_r    = r['event_es']
            paths_r       = r['paths']
            project_dur_r = r['project_dur']

            dur_str = (f"{int(project_dur_r)}"
                       if abs(project_dur_r - round(project_dur_r)) < 1e-6
                       else f"{project_dur_r:.2f}")
            n_real  = len([a for a in arcs_r if not a['is_dummy']])
            n_crit  = len([a for a in arcs_r if a['is_critical'] and not a['is_dummy']])
            n_dummy = len([a for a in arcs_r if a['is_dummy']])
            n_ev    = len(event_es_r)
            conf    = 100 if not r['warnings'] else max(70, 100 - len(r['warnings']) * 5)

            st.markdown("**📊 Resumen de Visualización**")
            if paths_r:
                st.success("✔ Ruta Crítica Encontrada")
            else:
                st.warning("⚠ Sin ruta crítica")

            st.metric("Duración Total", dur_str)
            st.metric("Actividades", n_real)
            st.metric("Críticas", n_crit)
            st.metric("Eventos", n_ev)
            st.metric("Confianza del Análisis", f"{conf}%")

            st.divider()

            # ── Ruta Crítica ──────────────────────────────────────────────────
            if paths_r:
                st.markdown("**🔴 Ruta Crítica**")
                for i, (ev_path, act_seq, total) in enumerate(paths_r, 1):
                    total_str = (f"{int(total)}" if abs(total - round(total)) < 1e-6
                                 else f"{total:.2f}")
                    acts_str = " → ".join(act_seq) if act_seq else "(solo dummies)"
                    evs_str  = " → ".join(str(e) for e in ev_path)
                    if len(paths_r) > 1:
                        st.markdown(f"**Ruta {i}:** `{evs_str}`")
                    else:
                        st.markdown(f"**Eventos:** `{evs_str}`")
                    st.error(f"**{acts_str}** — ⏱ {total_str}")
                if n_dummy > 0:
                    st.caption(f"+ {n_dummy} ficticia(s)")

        else:
            st.info("🕸 **Vista Previa de la Red**\n\nConfigura las actividades y presiona **🚀 Generar red PERT/CPM**")
            st.markdown("**📊 Resumen de Visualización**")
            st.caption("⏳ Esperando datos...")

    # =========================================================================
    # RESULTADOS DETALLADOS (ancho completo debajo de las columnas)
    # =========================================================================
    if st.session_state.results:
        r           = st.session_state.results
        arcs        = r['arcs']
        event_es    = r['event_es']
        event_lf    = r['event_lf']
        warnings    = r['warnings']
        timing      = r['timing']
        paths       = r['paths']
        project_dur = r['project_dur']
        fig         = r['fig']

        if warnings:
            with st.expander(f"⚠️ {len(warnings)} advertencia(s)", expanded=False):
                for w in warnings:
                    st.warning(w)

        tab1, tab2, tab3 = st.tabs([
            "📋 Tabla CPM", "🔵 Eventos", "📊 Resumen"
        ])

        # TAB 1: Tabla CPM
        with tab1:
            st.subheader("📋 Tabla CPM — por actividad")
            st.dataframe(pd.DataFrame(timing), use_container_width=True, hide_index=True)
            with st.expander("📖 ¿Qué significa cada columna?"):
                st.markdown("""
| Columna | Significado |
|---------|-------------|
| **Evento i / j** | Eventos de inicio y fin del arco |
| **ES** | Early Start — inicio más temprano |
| **EF** | Early Finish — fin más temprano (ES + Duración) |
| **LS** | Late Start — inicio más tardío |
| **LF** | Late Finish — fin más tardío |
| **Holgura Total** | LS − ES (tiempo disponible sin retrasar el proyecto) |
| **Crítica** | 🔴 SÍ si holgura = 0 |
                """)

        # TAB 2: Eventos
        with tab2:
            st.subheader("🔵 Tabla de Eventos — Nodos de la red")
            ev_data = []
            for ev in sorted(event_es.keys()):
                slack = event_lf[ev] - event_es[ev]
                ev_data.append({
                    'Evento':                  ev,
                    'TPI (Tiempo temprano)':   _fmt_num(event_es[ev]),
                    'TLT (Tiempo tardío)':     _fmt_num(event_lf[ev]),
                    'Holgura':                 _fmt_num(slack),
                    'Crítico':                 '🔴 SÍ' if abs(slack) < 1e-6 else ''
                })
            st.dataframe(pd.DataFrame(ev_data), use_container_width=True, hide_index=True)

        # TAB 3: Resumen
        with tab3:
            n_real  = len([a for a in arcs if not a['is_dummy']])
            n_dummy = len([a for a in arcs if a['is_dummy']])
            n_crit  = len([a for a in arcs if a['is_critical'] and not a['is_dummy']])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Duración Total",
                        f"{int(project_dur)} u.t."
                        if abs(project_dur - round(project_dur)) < 1e-6
                        else f"{project_dur:.2f}")
            col2.metric("Actividades", n_real)
            col3.metric("Críticas",    n_crit)
            col4.metric("Eventos",     len(event_es))

            if n_dummy > 0:
                st.info(f"Se agregaron **{n_dummy} actividades ficticias (dummies)** para representar correctamente las dependencias.")

            # Varianzas PERT (solo en modo probabilístico)
            prob_sum = st.session_state.get('prob_summary')
            if st.session_state.get('prob_mode') and prob_sum is not None:
                st.divider()
                st.subheader("📐 Varianzas PERT por actividad")
                _hl_cols = ['tₑ (Esperado)', 'σ² (Varianza)']
                _styled2 = (
                    prob_sum.style
                    .set_properties(
                        subset=_hl_cols,
                        **{'background-color': '#0D1B2A',
                           'color': '#93C5FD', 'font-weight': '600'}
                    )
                )
                st.dataframe(_styled2, use_container_width=True, hide_index=True)
                if paths:
                    import math
                    var_dict = dict(zip(prob_sum['Actividad'], prob_sum['σ² (Varianza)']))
                    st.markdown("#### Varianza de la(s) ruta(s) crítica(s)")
                    crit_cols = st.columns(min(len(paths), 4))
                    for idx, (ev_path, act_seq, total) in enumerate(paths):
                        crit_var   = sum(var_dict.get(a, 0) for a in act_seq)
                        crit_sigma = math.sqrt(crit_var) if crit_var > 0 else 0
                        label = f"Ruta {idx + 1}" if len(paths) > 1 else "Ruta crítica"
                        with crit_cols[idx % len(crit_cols)]:
                            st.metric(f"σ² {label}", f"{round(crit_var, 4)}",
                                      help=f"σ = {round(crit_sigma, 4)}")

        # ── Descargas ──────────────────────────────────────────────────────────
        st.divider()
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)

        with col_d1:
            png_data = getattr(fig, '_graphviz_png_bytes', None)
            if png_data is None:
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
                png_data = buf.getvalue()
            st.download_button("⬇ Descargar PNG", data=png_data,
                               file_name="red_pert_aoa.png",
                               mime="image/png", use_container_width=True)

        with col_d2:
            svg_data = getattr(fig, '_graphviz_svg_bytes', None)
            if svg_data is not None:
                st.download_button("⬇ Descargar SVG", data=svg_data,
                                   file_name="red_pert_aoa.svg",
                                   mime="image/svg+xml", use_container_width=True)
            else:
                st.caption("SVG no disponible")

        with col_d3:
            csv_data = pd.DataFrame(timing).to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Descargar CSV", data=csv_data,
                               file_name="tabla_cpm.csv",
                               mime="text/csv", use_container_width=True)

        with col_d4:
            try:
                excel_data = export_to_excel(
                    timing, event_es, event_lf, project_dur, paths, fig
                )
                st.download_button(
                    "⬇ Reporte Excel", data=excel_data,
                    file_name="reporte_pert_cpm.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except ImportError:
                st.caption("Para Excel: `pip install openpyxl`")

    # =========================================================================
    # FOOTER
    # =========================================================================
    st.markdown("""
    <div style="
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(99,102,241,0.12);
        text-align: center;
        opacity: 0.38;
        font-size: 0.77rem;
    ">
        Analítica PERT/CPM &nbsp;·&nbsp; Algoritmo Predecessor-Signature con Graphviz (v5) &nbsp;·&nbsp; Obsidian Analytics
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
