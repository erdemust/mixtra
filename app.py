"""Self-contained Streamlit pathfinding and network-routing simulator.

Run with: streamlit run app.py
This file deliberately contains its algorithms, models, renderers, and UI so
Streamlit Cloud does not depend on local project-package imports.
"""
from collections import deque
from heapq import heappop, heappush
from math import sqrt
from random import Random
from time import perf_counter, sleep

import streamlit as st


# ---------------------------------------------------------------------------
# Grid model and search algorithms
# ---------------------------------------------------------------------------
def make_grid(size: int, density: int, seed: int) -> set[tuple[int, int]]:
    rng = Random(seed)
    walls = {(r, c) for r in range(size) for c in range(size)
             if rng.random() < density / 100}
    walls.discard((0, 0))
    walls.discard((size - 1, size - 1))
    return walls


def grid_neighbors(point, size, walls, diagonal=False):
    r, c = point
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    found = []
    for dr, dc in offsets:
        nxt = (r + dr, c + dc)
        if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
            continue
        if dr and dc and ((r + dr, c) in walls or (r, c + dc) in walls):
            continue
        found.append(nxt)
    return found


def grid_search(size, walls, start, goal, algorithm="BFS", diagonal=False):
    """Return path, visit order, max frontier size, and elapsed milliseconds."""
    began = perf_counter()
    parents = {start: None}
    distances = {start: 0.0}
    visited = set()
    visit_order = []
    max_frontier = 1
    serial = 0
    if algorithm == "BFS":
        frontier = deque([start])
    else:
        frontier = [(0.0, serial, start)]

    def heuristic(point):
        dr, dc = abs(point[0] - goal[0]), abs(point[1] - goal[1])
        if diagonal:
            return max(dr, dc) + (sqrt(2) - 1) * min(dr, dc)
        return dr + dc

    while frontier:
        if algorithm == "BFS":
            current = frontier.popleft()
        else:
            _, _, current = heappop(frontier)
            if current in visited:
                continue
        visited.add(current)
        visit_order.append(current)
        if current == goal:
            break

        for nxt in grid_neighbors(current, size, walls, diagonal):
            step_cost = sqrt(2) if nxt[0] != current[0] and nxt[1] != current[1] else 1.0
            tentative = distances[current] + step_cost
            if algorithm == "BFS":
                if nxt in parents:
                    continue
                parents[nxt] = current
                distances[nxt] = distances[current] + 1
                frontier.append(nxt)
            elif tentative < distances.get(nxt, float("inf")):
                parents[nxt] = current
                distances[nxt] = tentative
                serial += 1
                if algorithm == "A*":
                    score = tentative + heuristic(nxt)
                elif algorithm == "Greedy":
                    score = heuristic(nxt)
                else:  # Dijkstra
                    score = tentative
                heappush(frontier, (score, serial, nxt))
        max_frontier = max(max_frontier, len(frontier))

    path = []
    if goal in parents:
        point = goal
        while point is not None:
            path.append(point)
            point = parents[point]
        path.reverse()
    return {
        "path": path,
        "visited": visit_order,
        "visited_count": len(visit_order),
        "max_frontier": max_frontier,
        "elapsed_ms": (perf_counter() - began) * 1000,
    }


def bfs_search_frames(size, walls, start, goal, diagonal=False):
    """Return faithful BFS queue snapshots for the Streamlit walkthrough."""
    queue = deque([start])
    parents = {start: None}
    visited = []
    frames = []
    max_queue = 1
    while queue:
        current = queue.popleft()
        visited.append(current)
        if current == goal:
            path = []
            point = goal
            while point is not None:
                path.append(point)
                point = parents[point]
            path.reverse()
            frames.append({"current": current, "visited": list(visited),
                           "frontier": list(queue), "max_queue": max_queue,
                           "path": path})
            return frames
        for neighbor in grid_neighbors(current, size, walls, diagonal):
            if neighbor not in parents:
                parents[neighbor] = current
                queue.append(neighbor)
        max_queue = max(max_queue, len(queue))
        frames.append({"current": current, "visited": list(visited),
                       "frontier": list(queue), "max_queue": max_queue,
                       "path": []})
    return frames


def draw_grid(size, walls, result=None, start=(0, 0), goal=None,
              visited=None, frontier=None, current=None):
    goal = goal or (size - 1, size - 1)
    # 0 free, 1 wall, 2 visited, 3 route, 4 source, 5 goal, 6 frontier, 7 active
    cells = [[0] * size for _ in range(size)]
    for r, c in walls:
        cells[r][c] = 1
    active_visited = visited if visited is not None else (result or {}).get("visited", [])
    active_path = (result or {}).get("path", [])
    for r, c in active_visited:
        if cells[r][c] == 0:
            cells[r][c] = 2
    for r, c in (frontier or []):
        if cells[r][c] == 0:
            cells[r][c] = 6
    for r, c in active_path:
        if cells[r][c] in (0, 2, 6):
            cells[r][c] = 3
    if current is not None and current not in (start, goal):
        cells[current[0]][current[1]] = 7
    cells[start[0]][start[1]] = 4
    cells[goal[0]][goal[1]] = 5
    colors = ["#f1f5f9", "#202938", "#83b6e8", "#fbbf24",
              "#10b981", "#f43f5e", "#bfdbfe", "#00bceb"]
    cell, gap = 16, 1
    board = size * (cell + gap)
    rects = []
    for r, row in enumerate(cells):
        for c, value in enumerate(row):
            x, y = c * (cell + gap), r * (cell + gap)
            border = "#64748b" if value in (0, 1) else "#0f172a"
            rects.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                         f'rx="2" fill="{colors[value]}" stroke="{border}" stroke-width="0.6"/>')
            if value in (4, 5):
                label = "S" if value == 4 else "G"
                text_color = "#052e16" if value == 4 else "#4c0519"
                rects.append(f'<text x="{x + cell/2}" y="{y + cell*.69}" text-anchor="middle" '
                             f'font-family="sans-serif" font-size="11" font-weight="700" fill="{text_color}">{label}</text>')
    return (f'<div style="width:100%;overflow:auto;background:#080e18;padding:8px;border-radius:10px">'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {board} {board}" '
            f'role="img" aria-label="Pathfinding grid" style="width:100%;height:auto;max-height:75vh">'
            f'{"".join(rects)}</svg></div>')


# ---------------------------------------------------------------------------
# Cisco-style weighted topology
# ---------------------------------------------------------------------------
NETWORK = {
    "START": {"A": 2.0, "E": 6.0, "D": 8.0},
    "A": {"B": 3.0, "C": 6.75, "E": 5.0},
    "B": {"C": 5.0, "E": 2.67},
    "C": {"TARGET": 6.0, "G": 2.75},
    "E": {"TARGET": 3.0, "D": 1.0},
    "D": {"TARGET": 4.25, "G": 5.0},
    "TARGET": {"G": 9.0},
    "G": {},
}
NODE_POS = {
    "START": (.08, .48), "A": (.28, .78), "B": (.48, .78),
    "C": (.68, .78), "E": (.40, .30), "D": (.60, .30),
    "TARGET": (.84, .50), "G": (.94, .22),
}


def get_active_network(overrides):
    """Apply user-selected failures and congestion to the base topology."""
    active = {}
    for node, neighbors in NETWORK.items():
        active[node] = {}
        for neighbor, weight in neighbors.items():
            state = overrides.get(f"{node} → {neighbor}", "Normal")
            if state == "Failed":
                continue
            active[node][neighbor] = weight * (5 if state == "Congested (5x)" else 1)
    return active


def network_dijkstra(source, target, active_net):
    distances = {node: float("inf") for node in active_net}
    parents = {node: None for node in active_net}
    distances[source] = 0.0
    queue = [(0.0, source)]
    visited = []
    trace = []
    while queue:
        cost, node = heappop(queue)
        if node in visited or cost != distances.get(node, float("inf")):
            continue
        visited.append(node)
        trace.append({"Adım": len(trace) + 1, "Olay": "Yönlendiriciyi işle",
                      "Yönlendirici": node, "Maliyet": f"{cost:.2f}"})
        if node == target:
            break
        for neighbor, weight in active_net.get(node, {}).items():
            candidate = cost + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                parents[neighbor] = node
                heappush(queue, (candidate, neighbor))
                trace.append({"Adım": len(trace) + 1, "Olay": "Bağlantı maliyetini güncelle",
                              "Yönlendirici": f"{node} → {neighbor}",
                              "Maliyet": f"{weight:.2f} (toplam {candidate:.2f})"})
    path = []
    if distances[target] < float("inf"):
        node = target
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
    table = [{"Yönlendirici": node,
              "Mesafe": "∞" if distances.get(node, float("inf")) == float("inf") else f"{distances[node]:.2f}",
              "Sonraki durak": parents.get(node) or "—",
              "Durum": "Ziyaret edildi" if node in visited else "Ziyaret edilmedi"}
             for node in active_net]
    return {"source": source, "target": target, "distances": distances,
            "parents": parents, "visited": visited, "path": path,
            "cost": distances.get(target, float("inf")), "table": table, "trace": trace}


def network_bellman_ford(source, target, active_net):
    distances = {node: float("inf") for node in active_net}
    parents = {node: None for node in active_net}
    distances[source] = 0.0
    trace = []
    for _ in range(max(0, len(active_net) - 1)):
        changed = False
        for node, neighbors in active_net.items():
            if distances[node] == float("inf"):
                continue
            for neighbor, weight in neighbors.items():
                candidate = distances[node] + weight
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    parents[neighbor] = node
                    changed = True
                    trace.append({"Adım": len(trace) + 1, "Olay": "Bağlantı maliyetini güncelle",
                                  "Yönlendirici": f"{node} → {neighbor}",
                                  "Maliyet": f"{weight:.2f} (toplam {candidate:.2f})"})
        if not changed:
            break
    path = []
    if distances[target] < float("inf"):
        node = target
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
    reached = {node for node, cost in distances.items() if cost < float("inf")}
    table = [{"Yönlendirici": node,
              "Mesafe": "∞" if distances[node] == float("inf") else f"{distances[node]:.2f}",
              "Sonraki durak": parents[node] or "—",
              "Durum": "Ulaşılabilir" if node in reached else "Ulaşılamıyor"}
             for node in active_net]
    return {"source": source, "target": target, "distances": distances,
            "parents": parents, "visited": list(reached), "path": path,
            "cost": distances[target], "table": table, "trace": trace}


def draw_network(result, active_net):
    width, height = 1000, 570
    route_edges = set(zip(result["path"], result["path"][1:]))
    elements = ['<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
                '<path d="M0,0 L0,6 L7,3 z" fill="#94a3b8"/></marker></defs>']
    for node, neighbors in NETWORK.items():
        x1, y1 = NODE_POS[node]
        for neighbor, base_weight in neighbors.items():
            x2, y2 = NODE_POS[neighbor]
            hot = (node, neighbor) in route_edges
            is_active = neighbor in active_net.get(node, {})
            weight = active_net.get(node, {}).get(neighbor, base_weight)
            color = "#00bceb" if hot else ("#334155" if is_active else "#64748b")
            sx, sy = x1 * width, (1 - y1) * height
            ex, ey = x2 * width, (1 - y2) * height
            elements.append(f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" '
                            f'stroke="{color}" stroke-width="{3 if hot else 1.5}" '
                            f'stroke-dasharray="{"none" if is_active else "4 5"}" marker-end="url(#arrow)"/>')
            label = (f"{weight:g}" if weight == base_weight else f"{weight:g} (5×)") if is_active else "ARIZALI"
            label_color = "#fbbf24" if weight == base_weight else ("#f43f5e" if is_active else "#94a3b8")
            mx, my = (sx + ex) / 2, (sy + ey) / 2 - 5
            elements.append(f'<text x="{mx}" y="{my}" text-anchor="middle" font-size="12" '
                            f'font-family="sans-serif" fill="{label_color}" stroke="#080e18" '
                            f'stroke-width="4" paint-order="stroke">{label}</text>')
    for node, (x, y) in NODE_POS.items():
        color = "#10b981" if node == result["source"] else "#f43f5e" if node == result["target"] else "#00bceb"
        cx, cy = x * width, (1 - y) * height
        elements.append(f'<circle cx="{cx}" cy="{cy}" r="27" fill="#0f172a" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" font-size="12" '
                        f'font-family="sans-serif" font-weight="700" fill="white">{node}</text>')
    return (f'<div style="width:100%;overflow:auto;background:#080e18;padding:8px;border-radius:10px">'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="Weighted network topology" style="width:100%;height:auto">'
            f'{"".join(elements)}</svg></div>')


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Ağ Algoritmaları Laboratuvarı", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.stApp { background:#080e18; color:#e2e8f0; }
[data-testid="stSidebar"] { background:#0f172a; }
.block-container { padding-top:1.1rem; max-width:1500px; }
div[data-testid="stMetric"] { background:#0f172a; border:1px solid #1e293b;
padding:.75rem; border-radius:.7rem; }
</style>
""", unsafe_allow_html=True)
st.title("🧭 Ağ Algoritmaları Laboratuvarı")
st.caption("CS402 · YÖNLENDİRME PROTOKOLLERİ VE YOL BULMA")
grid_tab, network_tab = st.tabs(["▦  Izgarada Yol Bulma", "⌘  Cisco Topolojisi · Dijkstra"])

with st.sidebar:
    st.header("Izgara simülatörü")
    size = st.slider("Izgara boyutu", 10, 40, 20)
    density = st.slider("Engel yoğunluğu", 0, 50, 25, format="%d%%")
    seed = st.number_input("Harita tohumu", 0, 999_999, 7)
    algorithm = st.selectbox("Algoritma", ["A*", "BFS", "Dijkstra", "Açgözlü Arama"])
    diagonal = st.checkbox("Çapraz komşulara izin ver", False)
    edit_tool = st.selectbox("Harita düzenleme aracı", ["Duvar", "Başlangıç", "Hedef"])
    edit_cols = st.columns(2)
    with edit_cols[0]:
        edit_r = st.number_input("Satır", 0, size - 1, 0)
    with edit_cols[1]:
        edit_c = st.number_input("Sütun", 0, size - 1, 0)
    apply_edit = st.button("Hücreyi düzenle", use_container_width=True)
    randomize = st.button("🎲 Engelleri rastgele oluştur", use_container_width=True)
    clear_walls = st.button("▱ Tüm duvarları temizle", use_container_width=True)
    run_search = st.button("▶ Yol bul", type="primary", use_container_width=True)
    run_bfs_demo = st.button("BFS adımlarını göster", use_container_width=True)
    animate_delay = st.slider("Gösterim hızı", 0.0, 0.08, 0.02, 0.01)

signature = (size, density, int(seed))
if "signature" not in st.session_state:
    st.session_state.signature = None
if signature != st.session_state.signature or randomize:
    next_seed = int(seed) + (1 if randomize else 0)
    st.session_state.walls = make_grid(size, density, next_seed)
    st.session_state.start = (0, 0)
    st.session_state.goal = (size - 1, size - 1)
    st.session_state.signature = signature
    st.session_state.search_result = None
walls = st.session_state.walls
start, goal = st.session_state.start, st.session_state.goal
if apply_edit:
    point = (int(edit_r), int(edit_c))
    if edit_tool == "Başlangıç" and point != goal:
        start = point
        walls.discard(point)
        st.session_state.start = point
    elif edit_tool == "Başlangıç":
        st.warning("Başlangıç ve hedef farklı hücrelerde olmalıdır.")
    elif edit_tool == "Hedef" and point != start:
        goal = point
        walls.discard(point)
        st.session_state.goal = point
    elif edit_tool == "Hedef":
        st.warning("Başlangıç ve hedef farklı hücrelerde olmalıdır.")
    elif edit_tool == "Duvar":
        if point in (start, goal):
            st.warning("Başlangıç veya hedef hücresine duvar yerleştirilemez.")
        else:
            walls.add(point)
    st.session_state.search_result = None
    start, goal = st.session_state.start, st.session_state.goal
if clear_walls:
    walls.clear()
    st.session_state.search_result = None

if run_search:
    search_algorithm = "Greedy" if algorithm == "Açgözlü Arama" else algorithm
    st.session_state.search_result = grid_search(size, walls, start, goal, search_algorithm, diagonal)

with grid_tab:
    st.subheader("Izgara Yol Bulma Motoru")
    st.caption("Düzenlenebilir engelli bir ızgarada BFS, Dijkstra, A* ve Açgözlü Arama.")
    if run_bfs_demo:
        st.markdown("#### BFS kuyruğunun adım adım gösterimi")
        placeholder = st.empty()
        info = st.empty()
        frames = bfs_search_frames(size, walls, start, goal, diagonal)
        final_frame = frames[-1] if frames else {}
        for frame in frames:
            st.components.v1.html(
                draw_grid(size, walls, None, start, goal,
                          visited=frame["visited"], frontier=frame["frontier"],
                          current=frame["current"]),
                height=min(820, size * 19 + 55), scrolling=True,
            )
            info.caption(f"Ziyaret edilen hücre: {len(frame['visited'])} · Kuyruktaki hücre: {len(frame['frontier'])} · En büyük kuyruk: {frame['max_queue']}")
            if animate_delay:
                sleep(animate_delay)
        if final_frame.get("path"):
            st.components.v1.html(
                draw_grid(size, walls,
                          {"visited": final_frame["visited"], "path": final_frame["path"]},
                          start, goal),
                height=min(820, size * 19 + 55), scrolling=True,
            )
            st.success(f"BFS {len(final_frame['path']) - 1} adımda yol buldu. En büyük kuyruk boyutu: {final_frame['max_queue']}.")
        else:
            st.warning(f"Yol bulunamadı. BFS en büyük kuyruk boyutu: {final_frame.get('max_queue', 0)}.")

    visual, summary = st.columns([3, 1])
    with visual:
        st.components.v1.html(
            draw_grid(size, walls, st.session_state.search_result, start, goal),
            height=min(820, size * 19 + 55), scrolling=True,
        )
    with summary:
        st.markdown("#### Arama bilgileri")
        result = st.session_state.search_result
        st.markdown("🟩 Başlangıç · 🟥 Hedef · ⬛ Duvar\n\n🟦 Ziyaret edildi · 🟨 Yol · 🔹 Kuyruk")
        if result is None:
            st.info("Ayarları seçip **Yol bul** düğmesine basın.")
        elif result["path"]:
            st.success("Yol bulundu")
            st.metric("Yol uzunluğu", len(result["path"]) - 1, "adım")
            st.metric("Ziyaret edilen hücre", result["visited_count"])
            st.metric("Arama süresi", f"{result['elapsed_ms']:.2f} ms")
        else:
            st.error("Bu haritada yol bulunamadı.")
            st.metric("Ziyaret edilen hücre", result["visited_count"])

    with st.expander("Algoritmaların karşılaştırması"):
        st.markdown("""
- **BFS**: Katman katman ilerler ve ağırlıksız ızgarada en az adımlı yolu bulur.
- **Dijkstra**: Hareket maliyetlerini kullanarak en düşük maliyetli yolu bulur.
- **A***: Aramayı hedefe yönlendirmek için yol maliyeti ile sezgisel uzaklığı birlikte kullanır.
- **Açgözlü Arama**: Yalnızca sezgisel uzaklığa bakar. Daha hızlı olabilir, ancak en kısa yolu garanti etmez.

BFS ilk giren ilk çıkar (FIFO) kuyruğu kullanır. Kuyruğun sabit bir kapasitesi yoktur; gösterim, geçerli haritada ulaşılan en yüksek boyutu bildirir. Zaman ve bellek karmaşıklığı **O(V + E)** şeklindedir.
        """)

    with st.expander("Algoritmaları karşılaştır"):
        if st.button("Geçerli haritada karşılaştırmayı çalıştır", key="run_benchmark"):
            benchmark_rows = []
            for name in ("BFS", "Dijkstra", "A*", "Greedy"):
                data = grid_search(size, walls, start, goal, name, diagonal)
                display_name = "Açgözlü Arama" if name == "Greedy" else name
                benchmark_rows.append({"Algoritma": display_name, "Yol bulundu": bool(data["path"]),
                                       "Yol adımı": len(data["path"]) - 1 if data["path"] else None,
                                       "Ziyaret edilen hücre": data["visited_count"],
                                       "Süre (ms)": round(data["elapsed_ms"], 3)})
            st.dataframe(benchmark_rows, hide_index=True, use_container_width=True)
            st.bar_chart({"Algoritma": [row["Algoritma"] for row in benchmark_rows],
                          "Ziyaret edilen hücre": [row["Ziyaret edilen hücre"] for row in benchmark_rows]},
                         x="Algoritma", y="Ziyaret edilen hücre")

with network_tab:
    st.subheader("Cisco tarzı ağırlıklı topoloji")
    st.caption("Yönlendirici bağlantılarında arızaları ve yoğunluğu değiştirerek en kısa yol yönlendirmesini simüle edin.")
    nodes = list(NETWORK)
    with st.expander("🛠️ Bağlantı yoğunluğu ve arıza simülatörü", expanded=True):
        all_links = [f"{router} → {neighbor}"
                     for router, neighbors in NETWORK.items() for neighbor in neighbors]
        failure_col, congestion_col = st.columns(2)
        with failure_col:
            failed_links = st.multiselect(
                "Arızalı bağlantılar (sonsuz maliyet)", all_links, key="failed_links_ui"
            )
        with congestion_col:
            congested_links = st.multiselect(
                "Yoğun bağlantılar (5× maliyet)", all_links, key="congested_links_ui"
            )
        st.session_state.link_overrides = {
            link: ("Failed" if link in failed_links else
                   "Congested (5x)" if link in congested_links else "Normal")
            for link in all_links
        }

    active_net = get_active_network(st.session_state.link_overrides)
    control_cols = st.columns([1, 1, 1, 1])
    with control_cols[0]:
        source = st.selectbox("Kaynak yönlendirici", nodes, index=0, key="network_source")
    with control_cols[1]:
        destination = st.selectbox("Hedef yönlendirici", nodes, index=6, key="network_target")
    with control_cols[2]:
        route_algorithm = st.selectbox("Yönlendirme algoritması", ["Dijkstra (SPF)", "Bellman–Ford"])
    with control_cols[3]:
        st.write("")
        run_spf = st.button("▶ En kısa yolu hesapla", type="primary", use_container_width=True)
    calc_key = (source, destination, route_algorithm,
                tuple(sorted(st.session_state.link_overrides.items())))
    if run_spf or "network_result" not in st.session_state or st.session_state.get("network_calc_key") != calc_key:
        if route_algorithm == "Dijkstra (SPF)":
            st.session_state.network_result = network_dijkstra(source, destination, active_net)
        else:
            st.session_state.network_result = network_bellman_ford(source, destination, active_net)
        st.session_state.network_calc_key = calc_key
    network_result = st.session_state.network_result
    graph_col, table_col = st.columns([1.8, 1])
    with graph_col:
        st.components.v1.html(draw_network(network_result, active_net), height=600, scrolling=True)
    with table_col:
        st.markdown("#### Yönlendirme tablosu")
        st.dataframe(network_result["table"], hide_index=True, use_container_width=True)
        if network_result["path"]:
            st.success("En kısa yol: " + " → ".join(network_result["path"]))
            st.metric("Toplam bağlantı maliyeti", f"{network_result['cost']:.2f}")
        else:
            st.warning("Geçerli bağlantı koşullarında yol bulunamadı.")
        st.markdown("#### İşleme sırası")
        st.write(" → ".join(network_result["visited"]))
    with st.expander("Algoritma adım dökümü"):
        st.dataframe(network_result["trace"], hide_index=True, use_container_width=True)

st.caption("Ağ Algoritmaları Laboratuvarı · Etkileşimli Streamlit uygulaması")
