import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import heapq
import time
import json

st.set_page_config(page_title="Dijkstra Algorithm Interactive Sandbox", layout="wide")

st.title("🗺️ Dijkstra's Shortest Path Interactive Sandbox")
st.markdown("Adjacency matrix ağırlıklarını düzenleyin, en kısa yolu hesaplayın, adımları canlı simülasyonla izleyin ve verilerinizi **CSV/JSON** formatında dışa aktarın.")

# --- Session State Başlatma ---
if "node_labels" not in st.session_state:
    st.session_state.node_labels = ["A", "B", "C", "D", "E", "F"]

if "adj_matrix" not in st.session_state:
    st.session_state.adj_matrix = np.array([
        [0,  2,  0,  8,  0,  0],  # A
        [2,  0,  0,  5,  6,  0],  # B
        [0,  0,  0,  0,  9,  3],  # C
        [8,  5,  0,  0,  3,  2],  # D
        [0,  6,  9,  3,  0,  1],  # E
        [0,  0,  3,  2,  1,  0]   # F
    ], dtype=float)

if "last_step" not in st.session_state:
    st.session_state.last_step = 0

# --- Sidebar Ayarları ---
st.sidebar.header("⚙️ Graf Ayarları")

is_directed = st.sidebar.checkbox("Yönlü Graf (Directed Graph)", value=False)

labels_str = st.sidebar.text_input(
    "Düğüm Etiketleri (virgülle ayırın)", 
    value=",".join(st.session_state.node_labels)
)
new_labels = [l.strip() for l in labels_str.split(",") if l.strip()]

if new_labels != st.session_state.node_labels:
    old_labels = st.session_state.node_labels
    old_matrix = st.session_state.adj_matrix
    n_new = len(new_labels)
    new_matrix = np.zeros((n_new, n_new), dtype=float)
    
    for i, old_i in enumerate(old_labels):
        for j, old_j in enumerate(old_labels):
            if old_i in new_labels and old_j in new_labels:
                new_i = new_labels.index(old_i)
                new_j = new_labels.index(old_j)
                new_matrix[new_i][new_j] = old_matrix[i][j]
                
    st.session_state.node_labels = new_labels
    st.session_state.adj_matrix = new_matrix
    st.rerun()

n_nodes = len(st.session_state.node_labels)

st.sidebar.subheader("🎯 Rota Parametreleri")
start_node = st.sidebar.selectbox("Başlangıç Düğümü", st.session_state.node_labels, index=0)
target_node = st.sidebar.selectbox("Hedef Düğüm", st.session_state.node_labels, index=min(2, n_nodes-1))

# --- Ana Ekran: Matris Düzenleyici & Deterministik Çift Yönlü Simetri ---
st.subheader("📊 Komşuluk (Adjacency) Matrisi")
df_matrix = pd.DataFrame(
    st.session_state.adj_matrix, 
    index=st.session_state.node_labels, 
    columns=st.session_state.node_labels
)

edited_df = st.data_editor(df_matrix, num_rows="fixed", use_container_width=True)
new_matrix = edited_df.to_numpy(dtype=float).copy()

# Üst üçgen öncelikli deterministik çift yönlü simetri eşitleme
if not is_directed:
    old_matrix = st.session_state.adj_matrix
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if new_matrix[i, j] != old_matrix[i, j]:
                new_matrix[j, i] = new_matrix[i, j]
            elif new_matrix[j, i] != old_matrix[j, i]:
                new_matrix[i, j] = new_matrix[j, i]

st.session_state.adj_matrix = new_matrix

# Negatif Ağırlık Kontrolü
if (st.session_state.adj_matrix < 0).any():
    st.error("❌ Hata: Dijkstra algoritması negatif kenar ağırlıklarını desteklemez! Lütfen negatif değerleri düzeltin.")
    st.stop()

# --- Önbellekli (Cache) Adım Adım Dijkstra Motoru ---
@st.cache_data
def run_dijkstra_with_steps(matrix_bytes, node_labels, start_node, n):
    adj_matrix = np.frombuffer(matrix_bytes).reshape(n, n)
    graph = {node_labels[i]: [] for i in range(n)}
    
    for i in range(n):
        for j in range(n):
            w = adj_matrix[i][j]
            if w > 0:
                graph[node_labels[i]].append((node_labels[j], w))
                
    distances = {node: float('inf') for node in node_labels}
    distances[start_node] = 0
    previous_nodes = {node: None for node in node_labels}
    
    pq = [(0, start_node)]
    visited = set()
    steps = []
    
    # Başlangıç durumu (Adım 0)
    steps.append({
        "step_number": 0,
        "description": f"Başlangıç durumu. Kaynak düğüm '{start_node}' mesafesi 0 olarak atandı, diğerleri sonsuz (∞).",
        "current_node": None,
        "relaxing_edge": None,
        "distances": {k: (v if v != float('inf')) else None for k, v in distances.items()},
        "visited": list(visited),
        "previous": dict(previous_nodes),
        "queue": [[d, n] for d, n in pq if n not in visited]
    })
    
    step_count = 1
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        
        steps.append({
            "step_number": step_count,
            "description": f"En küçük mesafeye sahip unvisited düğüm olarak '{current_node}' (Mesafe: {current_distance}) seçildi ve ziyaret edildiye eklendi.",
            "current_node": current_node,
            "relaxing_edge": None,
            "distances": {k: (v if v != float('inf')) else None for k, v in distances.items()},
            "visited": list(visited),
            "previous": dict(previous_nodes),
            "queue": [[d, n] for d, n in pq if n not in visited]
        })
        step_count += 1
        
        for neighbor, weight in graph[current_node]:
            if neighbor in visited:
                continue
            tentative_distance = current_distance + weight
            if tentative_distance < distances[neighbor]:
                distances[neighbor] = tentative_distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (tentative_distance, neighbor))
                
                steps.append({
                    "step_number": step_count,
                    "description": f"Güncelleme (Relaxation): '{current_node}' üzerinden '{neighbor}' düğümüne daha kısa yol bulundu. Yeni mesafe: {tentative_distance}.",
                    "current_node": current_node,
                    "relaxing_edge": [current_node, neighbor],
                    "distances": {k: (v if v != float('inf')) else None for k, v in distances.items()},
                    "visited": list(visited),
                    "previous": dict(previous_nodes),
                    "queue": [[d, n] for d, n in pq if n not in visited]
                })
                step_count += 1
                
    return distances, previous_nodes, steps

def reconstruct_path(previous_nodes, start_node, target_node):
    path = []
    current = target_node
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()
    return path if path and path[0] == start_node else []

# Hesaplama ve Adımları Toplama
distances, previous_nodes, simulation_steps = run_dijkstra_with_steps(
    st.session_state.adj_matrix.tobytes(),
    tuple(st.session_state.node_labels),
    start_node,
    n_nodes
)
shortest_path = reconstruct_path(previous_nodes, start_node, target_node)

# --- Genel Sonuçlar Paneli ---
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📈 Rota Özeti")
    target_dist = distances[target_node]
    if start_node == target_node:
        st.info("ℹ️ Başlangıç ve hedef aynı düğüm. En kısa mesafe: 0.")
        st.metric(label=f"Mesafe ({start_node} ➔ {target_node})", value="0")
    elif target_dist == float('inf'):
        st.metric(label=f"Mesafe ({start_node} ➔ {target_node})", value="Ulaşılamıyor ❌")
    else:
        st.metric(label=f"En Kısa Mesafe", value=f"{target_dist}")
        st.success(f"**Bulunan Yol:** {' ➔ '.join(shortest_path)}")

with col2:
    with st.expander("📜 Tüm Algoritma Adımları Özeti"):
        for i, step in enumerate(simulation_steps):
            icon = "🔍" if step["current_node"] else "🎯"
            st.markdown(f"**{i+1}.** {icon} {step['description']}")

# --- 📥 DIŞA AKTARMA (EXPORT) BÖLÜMÜ ---
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Verileri İndir (Export)")

# 1. Matris CSV İndirme
csv_data = df_matrix.to_csv().encode('utf-8')
st.sidebar.download_button(
    label="📊 Matrisi İndir (CSV)",
    data=csv_data,
    file_name="dijkstra_adjacency_matrix.csv",
    mime="text/csv"
)

# 2. Simülasyon Logları JSON İndirme
simulation_export_data = {
    "graph_info": {
        "nodes": st.session_state.node_labels,
        "is_directed": is_directed,
        "start_node": start_node,
        "target_node": target_node
    },
    "final_results": {
        "shortest_path": shortest_path,
        "shortest_distance": target_dist if target_dist != float('inf') else None,
        "all_distances": {k: (v if v != float('inf')) else "Infinity" for k, v in distances.items()}
    },
    "simulation_steps": simulation_steps
}

json_data = json.dumps(simulation_export_data, ensure_ascii=False, indent=4).encode('utf-8')
st.sidebar.download_button(
    label="📜 Simülasyon Logunu İndir (JSON)",
    data=json_data,
    file_name="dijkstra_simulation_log.json",
    mime="application/json"
)


# --- İNTERAKTİF ADIM ADIM SİMÜLASYON OYNATICISI ---
st.markdown("---")
st.subheader("🎬 Adım Adım Algoritma Simülasyonu ve Canlı Graf")

control_col1, control_col2, control_col3 = st.columns([1, 1, 2])
with control_col1:
    play_button = st.button("▶️ Otomatik Oynat (Play)")
with control_col2:
    anim_speed = st.selectbox("Oynatma Hızı", ["Yavaş (1.2s)", "Normal (0.8s)", "Hızlı (0.4s)"], index=1)

speed_dict = {"Yavaş (1.2s)": 1.2, "Normal (0.8s)": 0.8, "Hızlı (0.4s)": 0.4}
selected_delay = speed_dict[anim_speed]

total_steps = len(simulation_steps)

# Ortak Render Fonksiyonu
def render_simulation_step(step_idx):
    is_final_step = (step_idx == total_steps - 1)
    current_step_data = simulation_steps[step_idx]
    st.info(f"**Adım {step_idx + 1} / {total_steps}:** {current_step_data['description']}")
    
    sim_col1, sim_col2 = st.columns([1.3, 1])
    
    with sim_col1:
        st.markdown("##### 📋 Anlık Düğüm Durumları")
        table_data = []
        for node in st.session_state.node_labels:
            dist_val = current_step_data["distances"][node]
            dist_str = "∞" if dist_val is None else str(dist_val)
            is_visited = node in current_step_data["visited"]
            prev = current_step_data["previous"][node]
            prev_str = "-" if prev is None else str(prev)
            
            if current_step_data["current_node"] is None and node == start_node:
                status = "Kaynak (Start) 🎯"
            elif current_step_data["current_node"] is None:
                status = "Bekliyor 💤"
            elif node == current_step_data["current_node"]:
                status = "Aktif Düğüm 🔍"
            elif is_visited:
                status = "Ziyaret Edildi ✅"
            elif dist_val is not None:
                status = "Keşfedildi ⏳"
            else:
                status = "Bekliyor 💤"
                
            table_data.append({
                "Düğüm": node,
                "Mesafe": dist_str,
                "Önceki": prev_str,
                "Durum": status
            })
        st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
        
        # Priority Queue Görünümü
        st.markdown("##### 🔄 Öncelik Kuyruğu (Min-Heap)")
        if current_step_data["queue"]:
            q_df = pd.DataFrame(current_step_data["queue"], columns=["Mesafe", "Düğüm"])
            q_df = q_df.sort_values("Mesafe").drop_duplicates("Düğüm", keep="first").reset_index(drop=True)
            st.dataframe(q_df, hide_index=True, use_container_width=True)
        else:
            st.success("✅ Kuyruk boşaldı — algoritma tamamlandı.")

    with sim_col2:
        st.markdown("##### 🎨 Adıma Göre Graf Durumu")
        G = nx.DiGraph() if is_directed else nx.Graph()
        for label in st.session_state.node_labels:
            G.add_node(label)
            
        for i in range(n_nodes):
            for j in range(n_nodes):
                w = st.session_state.adj_matrix[i][j]
                if w > 0:
                    if not is_directed and i > j:
                        continue
                    G.add_edge(st.session_state.node_labels[i], st.session_state.node_labels[j], weight=w)
                    
        fig, ax = plt.subplots(figsize=(5, 3.8))
        
        try:
            pos = nx.kamada_kawai_layout(G) if n_nodes > 2 else nx.spring_layout(G, seed=42)
        except Exception:
            pos = nx.spring_layout(G, seed=42)
        
        node_colors = []
        for node in G.nodes():
            if node == current_step_data["current_node"]:
                node_colors.append("#d62728")  # Kırmızı (Aktif)
            elif node in current_step_data["visited"]:
                node_colors.append("#2ca02c")  # Yeşil (Ziyaret Edildi)
            elif current_step_data["current_node"] is None and node == start_node:
                node_colors.append("#9467bd")  # Mor (Başlangıç)
            elif current_step_data["distances"][node] is not None:
                node_colors.append("#ff7f0e")  # Turuncu (Keşfedildi)
            else:
                node_colors.append("#1f77b4")  # Mavi (Bekliyor)
                
        nx.draw_networkx_nodes(G, pos, node_size=650, node_color=node_colors, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', font_color='white', ax=ax)
        
        if is_directed:
            for u, v in G.edges():
                rad = 0.1 if u < v else -0.1
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], width=2, edge_color='gray',
                    connectionstyle=f"arc3,rad={rad}", ax=ax, arrows=True, arrowsize=12
                )
        else:
            nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', ax=ax)
            
        relaxing = current_step_data.get("relaxing_edge")
        if relaxing:
            nx.draw_networkx_edges(G, pos, edgelist=[tuple(relaxing)], width=3.5, edge_color='orange', ax=ax)
            
        if is_final_step and shortest_path and len(shortest_path) > 1:
            path_edges = list(zip(shortest_path[:-1], shortest_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4, edge_color='gold', ax=ax)
            
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
        
        pos_below = {k: (v[0], v[1] - 0.14) for k, v in pos.items()}
        dist_labels = {
            n: f"d={current_step_data['distances'][n] if current_step_data['distances'][n] is not None else '∞'}"
            for n in G.nodes()
        }
        nx.draw_networkx_labels(G, pos_below, labels=dist_labels, font_size=8, font_weight='bold', font_color='black', ax=ax)
        
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)

# Otomatik oynatma veya Manuel kontrol
if play_button:
    placeholder = st.empty()
    for s_idx in range(total_steps):
        with placeholder.container():
            st.progress((s_idx + 1) / total_steps, text=f"Adım {s_idx + 1} / {total_steps} oynatılıyor...")
            render_simulation_step(s_idx)
        time.sleep(selected_delay)
    st.success("🎉 Simülasyon tamamlandı!")
    st.session_state.last_step = total_steps - 1
    placeholder.empty()
    render_simulation_step(total_steps - 1)
else:
    step_idx = st.slider(
        "Simülasyon Adımı (Manuel Kontrol)",
        min_value=0, 
        max_value=total_steps - 1,
        value=min(st.session_state.get("last_step", 0), total_steps - 1)
    )
    st.session_state.last_step = step_idx
    render_simulation_step(step_idx)
