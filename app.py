import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Network Anomaly Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Warm Sand & Terracotta Minimalist Theme CSS
st.markdown("""
<style>
/* App Main Container - Warm Sand Background */
[data-testid="stAppViewContainer"] {
    background-color: #fbf9f4;
}

/* Sidebar background styling - Muted Linen Grey */
[data-testid="stSidebar"] {
    background-color: #f3efe6;
    border-right: 1px solid #e7dfd1;
}

/* Base Headings and Text Color Configuration (Dark Charcoal Slate) */
h1, h2, h3, h4, h5, h6, .stMarkdown p, label, .stMetricValue, [data-testid="stHeader"] {
    color: #1e293b !important;
    font-family: 'Segoe UI', system-ui, sans-serif;
}

/* Header container with clean Editorial styling */
.hero-banner {
    background-color: #ffffff;
    padding: 35px;
    border-radius: 16px;
    border: 1px solid #e7dfd1;
    border-bottom: 4px solid #c2410c;
    box-shadow: 0px 10px 30px rgba(194, 65, 12, 0.03);
    margin-bottom: 30px;
    text-align: center;
}

.hero-banner h1 {
    color: #7c2d12 !important;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 5px;
}

.hero-banner p {
    color: #c2410c !important;
    font-size: 15px;
    font-weight: 500;
}

/* Styled metric card panels - Clean Linen Slabs with warm terracotta borders */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #e7dfd1;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
}

/* Card panels for organizing inputs and statistics */
.input-card {
    background-color: #ffffff;
    border: 1px solid #e7dfd1;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
}

/* Main action buttons styled in Terracotta gradient */
.stButton > button {
    background: linear-gradient(135deg, #c2410c, #9a3412);
    color: white !important;
    font-weight: 700;
    border-radius: 8px;
    padding: 12px 30px;
    border: none;
    cursor: pointer;
    box-shadow: 0px 4px 15px rgba(194, 65, 12, 0.2);
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #9a3412, #7c2d12);
    box-shadow: 0px 6px 20px rgba(194, 65, 12, 0.3);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <h1>NETWORK ANOMALY DETECTOR</h1>
    <p>Unsupervised Isolation Forest Engine • Real-Time Packet Analysis & Threat Mitigation</p>
</div>
""", unsafe_allow_html=True)

# Load Model & Scaler safely
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(DIR_PATH, "models", "if_model.pkl")
scaler_path = os.path.join(DIR_PATH, "models", "scaler.pkl")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error("Model artifacts not found. Please run model_training.py first to generate the pkl files.")
    st.stop()

# Dynamic file resolver to handle spaces or capitalizations in "Network Traffic.csv"
file_path = None
possible_names = [
    os.path.join(DIR_PATH, "data", "Network Traffic.csv"),
    os.path.join(DIR_PATH, "data", "network_traffic.csv"),
    os.path.join(DIR_PATH, "data", "Network_Traffic.csv")
]
for name in possible_names:
    if os.path.exists(name):
        file_path = name
        break

if file_path is None and os.path.exists(os.path.join(DIR_PATH, "data")):
    for f in os.listdir(os.path.join(DIR_PATH, "data")):
        if "network" in f.lower() and f.endswith(".csv"):
            file_path = os.path.join(DIR_PATH, "data", f)
            break

if file_path is None:
    st.error("Could not locate 'Network Traffic.csv' inside the 'data/' directory.")
    st.stop()

df = pd.read_csv(file_path)
X = df.drop("label", axis=1)

# Pre-populate slider values using feature means
means = X.mean().to_dict()

# Sidebar summary infobox
with st.sidebar:
    st.markdown("### Engine Parameters")
    st.info(f"**Contamination:** {model.contamination:.4f}")
    st.info(f"**Base Estimators:** {model.n_estimators}")
    st.markdown("---")
    st.write("Unsupervised network anomaly detection system.")

# ========================================
# LAYOUT: SPLIT-SCREEN GRID (t-SNE Style)
# ========================================
col_control, col_display = st.columns([1, 2], gap="large")

# --- LEFT COLUMN: CONTROL & COMPRESSION PANEL ---
with col_control:
    st.markdown("### ⚙️ Packet Controller")
    st.write("Configure high-dimensional connection attributes to verify if they represent anomalous signatures.")
    
    # Nested Grouped Accordion Inputs (UX Upgrade)
    with st.expander("📡 Connection Profile", expanded=True):
        packet_size = st.slider("Packet Size", 0.0, 1.0, float(means["packet_size"]))
        inter_arrival = st.slider("Inter-Arrival Time (Seconds)", 0.0, 1.0, float(means["inter_arrival_time"]))
        src_port = st.number_input("Source Port", min_value=0, max_value=65535, value=int(means["src_port"]))
        dst_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=int(means["dst_port"]))
        packet_count_5s = st.slider("Packet Count (5s window)", 0.0, 1.0, float(means["packet_count_5s"]))
        mean_packet_size = st.slider("Mean Packet Size", 0.0, 1.0, float(means["mean_packet_size"]))

    with st.expander("📊 Spectral Signature", expanded=False):
        spectral_entropy = st.slider("Spectral Entropy", 0.0, 1.0, float(means["spectral_entropy"]))
        freq_band_energy = st.slider("Frequency Band Energy", 0.0, 1.0, float(means["frequency_band_energy"]))
        protocol_tcp = st.selectbox("Protocol Type: TCP?", ["False", "True"])
        protocol_udp = st.selectbox("Protocol Type: UDP?", ["False", "True"])

    with st.expander("🔒 IP & Flag Details", expanded=False):
        src_ip_2 = st.selectbox("Source IP: 192.168.1.2?", ["False", "True"])
        src_ip_3 = st.selectbox("Source IP: 192.168.1.3?", ["False", "True"])
        dst_ip_5 = st.selectbox("Destination IP: 192.168.1.5?", ["False", "True"])
        dst_ip_6 = st.selectbox("Destination IP: 192.168.1.6?", ["False", "True"])
        flags_fin = st.selectbox("TCP Flags: FIN?", ["False", "True"])
        flags_syn = st.selectbox("TCP Flags: SYN?", ["False", "True"])
        flags_syn_ack = st.selectbox("TCP Flags: SYN-ACK?", ["False", "True"])

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("RUN PACKET ANOMALY DETECTION", use_container_width=True):
        # Convert string mappings to boolean states
        protocol_tcp_val = True if protocol_tcp == "True" else False
        protocol_udp_val = True if protocol_udp == "True" else False
        src_ip_2_val = True if src_ip_2 == "True" else False
        src_ip_3_val = True if src_ip_3 == "True" else False
        dst_ip_5_val = True if dst_ip_5 == "True" else False
        dst_ip_6_val = True if dst_ip_6 == "True" else False
        flags_fin_val = True if flags_fin == "True" else False
        flags_syn_val = True if flags_syn == "True" else False
        flags_syn_ack_val = True if flags_syn_ack == "True" else False

        # Align columns exactly with expected 17-feature scaled input vector
        raw_row = np.array([[
            packet_size, inter_arrival, src_port, dst_port, packet_count_5s,
            mean_packet_size, spectral_entropy, freq_band_energy,
            protocol_tcp_val, protocol_udp_val, src_ip_2_val, src_ip_3_val,
            dst_ip_5_val, dst_ip_6_val, flags_fin_val, flags_syn_val, flags_syn_ack_val
        ]])

        scaled_row = scaler.transform(raw_row)
        prediction = model.predict(scaled_row)[0]

        if prediction == -1:
            color_theme = "linear-gradient(90deg, #991b1b, #dc2626)"
            status_tag = "ANOMALOUS TRAFFIC SIGNAL (THREAT WARNING)"
        else:
            color_theme = "linear-gradient(90deg, #065f46, #10b981)"
            status_tag = "STABLE CONNECTION PROFILE"

        st.markdown(f"""
        <div style="
            background: {color_theme};
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            text-align: center;
            margin-top: 15px;
            color: white !important;">
            <h4 style="color: white !important; margin-bottom: 5px;">Security Check Verdict</h4>
            <p style="font-size: 20px; font-weight: 700; color: white !important;">{status_tag}</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

# --- RIGHT COLUMN: INTERACTIVE VISUALIZATION & KPIS ---
with col_display:
    st.markdown("### 📊 Live Connection Map & Stats")
    
    # Run predictions on dataset for stats and visualization
    X_scaled_all = scaler.transform(X)
    raw_preds_all = model.predict(X_scaled_all)
    anomalies_count = np.sum(raw_preds_all == -1)
    normal_count = np.sum(raw_preds_all == 1)

    # Metrics row inside the display column
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Monitored Connections", f"{df.shape[0]:,}")
    kpi_col2.metric("Normal Connections", f"{normal_count:,}")
    kpi_col3.metric("Anomalous Flags", f"{anomalies_count:,}", delta=f"{(anomalies_count/df.shape[0])*100:.2f}% Rate", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # Prepare DataFrame for Plotly Express
    plot_df = df.copy()
    plot_df["Security Status"] = np.where(raw_preds_all == -1, "Anomaly (Threat Flagged)", "Normal Traffic")

    # Render interactive Plotly scatter map
    fig_proj = px.scatter(
        plot_df, 
        x="packet_size", 
        y="inter_arrival_time", 
        color="Security Status",
        title="Interactive Outlier Mapping Space (Packet Size vs Inter-Arrival Time)",
        color_discrete_map={"Anomaly (Threat Flagged)": "#dc2626", "Normal Traffic": "#0d9488"},
        template="plotly_white"
    )
    
    fig_proj.update_layout(
        paper_bgcolor="#fbf9f4",
        plot_bgcolor="#ffffff",
        font_color="#1e293b",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig_proj, use_container_width=True)

    # Contextual Explanation Container
    st.markdown("""
    <div class="input-card">
        <h4 style="color: #c2410c !important;">💡 Unsupervised Threat Profiling</h4>
        <p style="color: #475569 !important; font-size: 14px;">
            The Isolation Forest functions by isolating anomalies instead of profiling normal data points. Because malicious packets or unexpected connection patterns are far less frequent and possess highly distinct values, they are partitioned much quicker than normal traffic in the decision trees. This allows the system to identify potential network threats without requiring manual categorical labeling.
        </p>
    </div>
    """, unsafe_allow_html=True)