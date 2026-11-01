# 🗺️ Dijkstra's Shortest Path Interactive Sandbox

An interactive, educational web application built with **Streamlit**, **NetworkX**, and **Matplotlib** that visualizes Dijkstra's Shortest Path Algorithm step by step.

![Streamlit App Badge](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

---

## ✨ Features

* **📊 Dynamic Adjacency Matrix Editor:** Easily edit edge weights directly in a table with automatic bidirectional symmetry enforcement for undirected graphs.
* **🎬 Step-by-Step Simulation & Auto-Play:** Step through algorithm iterations manually using a slider or use the **Auto-Play** feature with adjustable speed controls.
* **🔄 Priority Queue (Min-Heap) Inspector:** Observe how the algorithm tracks and updates tentative distances in real time, complete with lazy-deletion filtering.
* **🎨 Live Visual Graph Animation:** Nodes dynamically change colors based on their state (*Unvisited, Active, Visited, Tentative*), accompanied by relaxation edge highlights and final path reconstruction in gold.
* **📥 Data Export (CSV / JSON):** Download your custom adjacency matrix as a CSV file or export full simulation logs and results as a structured JSON file.
* **🛡️ Robust Error Handling:** Negative edge weight validation, automated `kamada_kawai_layout` / `spring_layout` fallbacks, and crash-proof session state management.

---

## 🚀 Local Installation & Running

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/your-repo-name.git](https://github.com/YOUR_USERNAME/your-repo-name.git)
   cd your-repo-name
