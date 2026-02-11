import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Maritime Anomaly Detection System",
    page_icon="⚓",
    layout="wide"
)

# --- CSS CUSTOM UNTUK TAMPILAN ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 1px 1px 5px #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚓ Sistem Deteksi Anomali Pergerakan Kapal")
st.markdown("### Studi Kasus: Pelabuhan Tanjung Priok (Isolation Forest vs One Class SVM)")
st.markdown("---")

# --- FUNGSI-FUNGSI LOGIKA (BACKEND) ---

@st.cache_data
def load_raw_data(file):
    df = pd.read_csv(file, sep=';')
    return df

def clean_coordinates(df):
    # 1. Cleaning Lat/Lon Function
    def clean_lat(val):
        try:
            return float(val) / 10.0
        except:
            return np.nan

    def clean_lon(val):
        if pd.isna(val): return np.nan
        if isinstance(val, str):
            clean_str = val.replace('.', '')
            try:
                val_float = float(clean_str)
                if val_float > 1000000: return val_float / 10000
                elif val_float > 100000: return val_float / 1000
                else: return val_float
            except: return np.nan
        return float(val)

    df['latitude'] = df['lat'].apply(clean_lat)
    df['longitude'] = df['lon'].apply(clean_lon)
    df['logtime'] = pd.to_datetime(df['logtime'])
    
    # Sort
    df = df.sort_values(['mmsi', 'logtime'])
    return df

def filter_roi_and_movement(df):
    # 2. Filter Stationary
    movement = df.groupby('mmsi').agg({'latitude': ['min', 'max'], 'longitude': ['min', 'max']})
    moving_mmsis = movement[
        (movement['latitude']['min'] != movement['latitude']['max']) |
        (movement['longitude']['min'] != movement['longitude']['max'])
    ].index
    df_moving = df[df['mmsi'].isin(moving_mmsis)].copy()
    
    # 3. Filter ROI (Priok)
    lat_min, lat_max = -6.15, -5.90
    lon_min, lon_max = 106.75, 107.00
    df_roi = df_moving[
        (df_moving['latitude'] >= lat_min) & (df_moving['latitude'] <= lat_max) &
        (df_moving['longitude'] >= lon_min) & (df_moving['longitude'] <= lon_max)
    ].copy()
    
    return df_roi

def segment_trajectories(df):
    # 4. Segmentation (Gap > 30 min)
    df['time_diff'] = df.groupby('mmsi')['logtime'].diff().dt.total_seconds() / 60.0
    df['is_new_segment'] = (df['mmsi'] != df['mmsi'].shift(1)) | (df['time_diff'] > 30) | (df['time_diff'].isna())
    df['segment_id'] = df['is_new_segment'].cumsum()
    return df

def feature_engineering(df):
    # Haversine
    def haversine_vectorize(lat1, lon1, lat2, lon2):
        R = 6371000 
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    # Calculate Deltas
    df['prev_lat'] = df.groupby(['mmsi', 'segment_id'])['latitude'].shift(1)
    df['prev_lon'] = df.groupby(['mmsi', 'segment_id'])['longitude'].shift(1)
    df['prev_time'] = df.groupby(['mmsi', 'segment_id'])['logtime'].shift(1)
    df['prev_speed'] = df.groupby(['mmsi', 'segment_id'])['speed'].shift(1)
    df['prev_course'] = df.groupby(['mmsi', 'segment_id'])['course'].shift(1)

    df['time_diff_sec'] = (df['logtime'] - df['prev_time']).dt.total_seconds()
    df['dist_move_m'] = haversine_vectorize(df['prev_lat'], df['prev_lon'], df['latitude'], df['longitude'])
    
    # Kinematics
    df['calc_speed_knots'] = (df['dist_move_m'] / df['time_diff_sec']) * 1.94384
    df['calc_speed_knots'] = df['calc_speed_knots'].fillna(0)
    
    df['acceleration'] = (df['speed'] - df['prev_speed']) / df['time_diff_sec']
    
    def calc_angle_diff(a, b):
        diff = b - a
        diff = (diff + 180) % 360 - 180
        return diff
    
    df['course_diff'] = calc_angle_diff(df['prev_course'], df['course'])
    df['rot_calc'] = df['course_diff'] / df['time_diff_sec']
    
    # Drop artifacts
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['acceleration', 'rot_calc'])
    return df

def physics_cleaning(df, max_speed=35.0):
    noise_mask = (df['calc_speed_knots'] > max_speed)
    df_clean = df[~noise_mask].copy()
    noise_count = noise_mask.sum()
    return df_clean, noise_count

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload AIS CSV", type=['csv'])
    
    st.header("2. Pengaturan Model")
    st.subheader("Isolation Forest")
    conta = st.slider("Contamination", 0.001, 0.05, 0.01, step=0.001)
    
    st.subheader("One Class SVM")
    nu_val = st.slider("Nu Parameter", 0.001, 0.05, 0.01, step=0.001)
    gamma_val = st.selectbox("Gamma", ["scale", "auto", 0.1])

# --- MAIN FLOW ---

if uploaded_file is not None:
    # --- TAB 1: DATA PREPARATION (Transparan untuk User) ---
    tab1, tab2, tab3 = st.tabs(["🛠️ Pemrosesan Data", "🧠 Deteksi Anomali", "📈 Analisis Sensitivitas"])
    
    # Placeholder global variables
    final_df = None
    
    with tab1:
        st.subheader("Langkah 1: Pembersihan & Engineering")
        
        with st.status("Sedang memproses data...", expanded=True) as status:
            # 1. Load
            st.write("📂 Membaca file CSV...")
            raw_df = load_raw_data(uploaded_file)
            st.write(f"   - Data Mentah: {len(raw_df)} baris")
            
            # 2. Clean Coords
            st.write("🧹 Membersihkan format koordinat & Filter ROI Tanjung Priok...")
            cleaned_coords = clean_coordinates(raw_df)
            roi_df = filter_roi_and_movement(cleaned_coords)
            st.write(f"   - Sisa data di ROI & Bergerak: {len(roi_df)} baris")
            
            # 3. Segmentation
            st.write("✂️ Segmentasi Trajektori (Gap > 30 menit)...")
            seg_df = segment_trajectories(roi_df)
            
            # 4. Engineering
            st.write("⚙️ Feature Engineering (Calc Speed, Accel, ROT)...")
            eng_df = feature_engineering(seg_df)
            
            # 5. Physics Filter
            st.write("🛡️ Physics-Based Filtering (Second-Pass Cleaning)...")
            final_df, noise_dropped = physics_cleaning(eng_df, max_speed=35.0)
            
            status.update(label="Pemrosesan Data Selesai!", state="complete", expanded=False)
        
        # Display Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Kapal Unik", final_df['mmsi'].nunique())
        col2.metric("Total Data Bersih", len(final_df))
        col3.metric("Noise Dibuang", noise_dropped, help="Data dengan speed > 35 knots (Mustahil)")
        col4.metric("Segmen Rute", final_df['segment_id'].nunique())
        
        st.markdown("#### Preview Data Hasil Engineering")
        st.dataframe(final_df[['mmsi', 'logtime', 'speed', 'calc_speed_knots', 'acceleration', 'rot_calc']].head())
        
        # Simpan di session state
        st.session_state['data'] = final_df

    # --- TAB 2: MODELLING & VISUALISASI ---
    with tab2:
        if 'data' in st.session_state:
            df = st.session_state['data']
            
            st.subheader("Eksekusi Model Deteksi Anomali")
            
            if st.button("Jalankan Model (iForest & OCSVM)"):
                # Prepare Features
                features = ['speed', 'calc_speed_knots', 'acceleration', 'rot_calc']
                X = df[features]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Run iForest
                iso = IsolationForest(contamination=conta, random_state=42)
                df['iforest_label'] = iso.fit_predict(X_scaled)
                df['iforest_score'] = iso.decision_function(X_scaled)
                
                # Run OCSVM
                ocsvm = OneClassSVM(kernel='rbf', nu=nu_val, gamma=gamma_val)
                df['ocsvm_label'] = ocsvm.fit_predict(X_scaled)
                
                # Save results
                st.session_state['results'] = df
                st.success("Model Selesai Dijalankan!")
            
            if 'results' in st.session_state:
                res_df = st.session_state['results']
                
                # Stats Anomali
                n_iforest = len(res_df[res_df['iforest_label'] == -1])
                n_ocsvm = len(res_df[res_df['ocsvm_label'] == -1])
                n_both = len(res_df[(res_df['iforest_label'] == -1) & (res_df['ocsvm_label'] == -1)])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Anomali iForest", n_iforest)
                c2.metric("Anomali OCSVM", n_ocsvm)
                c3.metric("Irisan (Strong Anomaly)", n_both)
                
                # Visualisasi Peta
                st.subheader("Peta Sebaran Anomali")
                mode_view = st.radio("Pilih Mode Visualisasi:", ["Isolation Forest", "One Class SVM", "Perbandingan"])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Background
                ax.scatter(res_df['longitude'], res_df['latitude'], c='lightgray', s=1, alpha=0.3, label='Normal')
                
                if mode_view == "Isolation Forest":
                    anom = res_df[res_df['iforest_label'] == -1]
                    ax.scatter(anom['longitude'], anom['latitude'], c='red', s=15, marker='x', label='iForest')
                    st.caption(f"Menampilkan {len(anom)} titik anomali iForest.")
                    
                elif mode_view == "One Class SVM":
                    anom = res_df[res_df['ocsvm_label'] == -1]
                    ax.scatter(anom['longitude'], anom['latitude'], c='blue', s=15, marker='x', label='OCSVM')
                    st.caption(f"Menampilkan {len(anom)} titik anomali OCSVM.")
                    
                elif mode_view == "Perbandingan":
                    anom_if = res_df[res_df['iforest_label'] == -1]
                    anom_oc = res_df[res_df['ocsvm_label'] == -1]
                    ax.scatter(anom_oc['longitude'], anom_oc['latitude'], c='blue', s=15, marker='o', alpha=0.3, label='OCSVM')
                    ax.scatter(anom_if['longitude'], anom_if['latitude'], c='red', s=15, marker='x', alpha=0.8, label='iForest')
                
                ax.set_title(f"Visualisasi Anomali ({mode_view})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(106.75, 107.00)
                ax.set_ylim(-6.15, -5.90)
                st.pyplot(fig)
                
                # Tabel Top 5
                st.subheader("🏆 Top 5 Anomali (iForest Score Terendah)")
                top5 = res_df[res_df['iforest_label'] == -1].sort_values('iforest_score').head(5)
                st.table(top5[['mmsi', 'speed', 'calc_speed_knots', 'acceleration', 'iforest_score']])

    # --- TAB 3: SENSITIVITY ANALYSIS ---
    with tab3:
        st.subheader("Analisis Sensitivitas Hyperparameter")
        st.info("Proses ini akan menjalankan model berulang kali dengan parameter berbeda. Harap bersabar.")
        
        if 'data' in st.session_state and st.button("Mulai Analisis Sensitivitas"):
            df_sens = st.session_state['data']
            features = ['speed', 'calc_speed_knots', 'acceleration', 'rot_calc']
            X_sens = StandardScaler().fit_transform(df_sens[features])
            
            # 1. iForest Sensitivity
            cont_range = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
            if_counts = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, c in enumerate(cont_range):
                status_text.text(f"Running iForest Contamination={c}...")
                model = IsolationForest(contamination=c, random_state=42)
                preds = model.fit_predict(X_sens)
                if_counts.append(np.sum(preds == -1))
                progress_bar.progress((i + 1) / len(cont_range))
                
            # Plotting
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(cont_range, if_counts, marker='o', color='red')
            ax.set_title("Sensitivitas Isolation Forest")
            ax.set_xlabel("Contamination")
            ax.set_ylabel("Jumlah Anomali")
            ax.grid(True)
            st.pyplot(fig)
            
            st.success("Analisis Selesai! Grafik di atas membuktikan stabilitas model.")

else:
    st.info("👋 Silakan upload file `ais1.csv` Anda di menu sebelah kiri untuk memulai sistem.")