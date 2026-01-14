import streamlit as st
import pandas as pd
import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from datetime import timedelta
import warnings
from dotenv import load_dotenv
import io 
from fpdf import FPDF # Library untuk PDF

# --- TAMBAHAN LIBRARY BARU ---
import firebase_admin
from firebase_admin import credentials, firestore, auth
from openai import OpenAI
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- KONFIGURASI AWAL (WAJIB DI ATAS) ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Oil & Gas AI Dashboard", layout="wide", page_icon="🛢️")

# Load environment variables
load_dotenv()

# --- MODIFIKASI: INISIALISASI FIREBASE (ROBUST & AUTO-FIX) ---
if not firebase_admin._apps:
    try:
        firebase_creds = None
        
        # Skenario 1: Cek di Streamlit Secrets dengan header [firebase] (Recommended)
        if "firebase" in st.secrets:
            firebase_creds = dict(st.secrets["firebase"])
        
        # Skenario 2: Cek di Root Streamlit Secrets (Jika user lupa header [firebase])
        elif "private_key" in st.secrets and "project_id" in st.secrets:
            firebase_creds = dict(st.secrets)

        # Proses Login jika kredensial ditemukan di Cloud
        if firebase_creds:
            # FIX PENTING: Koreksi format private_key (mengubah \\n menjadi \n asli)
            if "private_key" in firebase_creds:
                firebase_creds["private_key"] = firebase_creds["private_key"].replace("\\n", "\n")
            
            cred = credentials.Certificate(firebase_creds)
            firebase_admin.initialize_app(cred)
            
        # Skenario 3: Cek file JSON di Local Computer
        elif os.path.exists('keyfirebase.json'):
            cred = credentials.Certificate('keyfirebase.json')
            firebase_admin.initialize_app(cred)
            
        else:
            st.warning("⚠️ Konfigurasi Firebase tidak ditemukan di Secrets maupun file lokal.")
            
    except Exception as e:
        st.error(f"Gagal inisialisasi Firebase: {e}")

# ==========================================
# BAGIAN 1: SISTEM LOGIN & USER MANAGEMENT
# ==========================================

def init_user_db():
    """Inisialisasi tabel user lokal dengan migrasi otomatis"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # 1. Cek apakah tabel users sudah ada
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = c.fetchone()

    if table_exists:
        # 2. Cek apakah kolom 'provider' sudah ada
        try:
            c.execute("SELECT provider FROM users LIMIT 1")
        except sqlite3.OperationalError:
            # Jika error (kolom tidak ada), tambahkan kolom baru
            st.warning("Mengupdate struktur database lama...")
            try:
                c.execute("ALTER TABLE users ADD COLUMN email TEXT")
                c.execute("ALTER TABLE users ADD COLUMN provider TEXT")
                # Set default value untuk user lama
                c.execute("UPDATE users SET provider = 'local', email = '' WHERE provider IS NULL")
                conn.commit()
                st.success("Database berhasil diupdate! Silakan refresh halaman.")
            except Exception as e:
                # Jika gagal alter, reset tabel (opsi terakhir)
                c.execute("DROP TABLE users")
                conn.commit()

    # 3. Pastikan tabel dibuat dengan struktur lengkap
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, status TEXT, email TEXT, provider TEXT)''')
    
    # 4. Buat Admin Default
    env_user = os.getenv("APP_USERNAME", "admin")
    env_pass = os.getenv("APP_PASSWORD", "admin123")
    
    c.execute("SELECT * FROM users WHERE username=?", (env_user,))
    if not c.fetchone():
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", 
                  (env_user, env_pass, 'approved', 'admin@system.local', 'local'))
        conn.commit()
    
    conn.close()

def register_user(username, password, email="", provider="local"):
    """Mendaftarkan user baru dengan status pending"""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        # Cek duplikat username
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        if c.fetchone():
            conn.close()
            return False, "Username sudah terdaftar."
        
        # Jika login via Google, password bisa kosong/dummy
        if provider == 'google': 
            password = 'GOOGLE_AUTH_USER'

        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (username, password, 'pending', email, provider))
        conn.commit()
        conn.close()
        return True, "Registrasi berhasil! Menunggu persetujuan Admin."
    except Exception as e:
        return False, str(e)

def login_user(username, password):
    """Verifikasi login user manual"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT status FROM users WHERE username=? AND password=? AND provider='local'", (username, password))
    result = c.fetchone()
    conn.close()
    
    if result:
        status = result[0]
        if status == 'approved':
            return True, "Login Berhasil"
        else:
            return False, "Akun Anda sedang menunggu persetujuan Admin."
    return False, "Username atau Password salah."

def login_with_google(email):
    """Simulasi Login Google (Cek apakah email terdaftar & approved)"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT status, username FROM users WHERE email=? AND provider='google'", (email,))
    result = c.fetchone()
    conn.close()
    
    if result:
        status, username = result
        if status == 'approved':
            return True, "Login Berhasil", username
        else:
            return False, "Akun Google Anda belum disetujui Admin.", username
    else:
        # Jika belum ada, otomatis register
        username = email.split('@')[0]
        register_user(username, "", email, "google")
        return False, "Akun Google baru didaftarkan. Menunggu persetujuan Admin.", username

def get_pending_users():
    conn = sqlite3.connect('users.db')
    df = pd.read_sql("SELECT username, email, provider, status FROM users WHERE status='pending'", conn)
    conn.close()
    return df

def approve_user(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("UPDATE users SET status='approved' WHERE username=?", (username,))
    conn.commit()
    conn.close()

def reject_user(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()

def page_admin_panel():
    # --- SECURITY GATE: HANYA ADMIN YANG BISA AKSES ---
    # Ini memastikan dalaman konten tidak akan dirender jika user bukan admin
    if not st.session_state.get('is_admin', False):
        st.error("⛔ AKSES DITOLAK: Halaman ini khusus untuk Administrator.")
        return

    st.title("🔒 Admin Panel - User Management")
    
    # List Pending Users
    st.subheader("⏳ Permintaan Akses Baru")
    df_pending = get_pending_users()
    
    if not df_pending.empty:
        st.info(f"Ada {len(df_pending)} user menunggu persetujuan.")
        for idx, row in df_pending.iterrows():
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"**{row['username']}**")
                    st.caption(f"Email: {row['email']} | Via: {row['provider']}")
                with c2:
                    if st.button("✅ Approve", key=f"app_{row['username']}"):
                        approve_user(row['username'])
                        st.success(f"User {row['username']} disetujui!")
                        st.rerun()
                with c3:
                    if st.button("❌ Reject", key=f"rej_{row['username']}"):
                        reject_user(row['username'])
                        st.error(f"User {row['username']} ditolak.")
                        st.rerun()
    else:
        st.success("Tidak ada permintaan registrasi baru.")

    st.markdown("---")
    # List All Users
    with st.expander("👥 Lihat Semua User Terdaftar"):
        conn = sqlite3.connect('users.db')
        df_all = pd.read_sql("SELECT username, email, status, provider FROM users", conn)
        st.dataframe(df_all, use_container_width=True)
        conn.close()

def check_login_page():
    init_user_db()
    
    st.markdown(
        """
        <style>
        .login-container {
            max-width: 450px; margin: auto; padding: 40px;
            border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            background-color: #ffffff;
        }
        .google-btn {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: white;
            color: #333;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            margin-top: 10px;
        }
        .google-btn:hover { background-color: #f1f1f1; }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn-icons-png.freepik.com/256/16812/16812635.png?semt=ais_white_label", width=80)
        st.title("Login Dashboard")
        st.caption("Oil & Gas Data Analytics Platform")
        
        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])
        
        # --- TAB LOGIN ---
        with tab_login:
            # Login Manual
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Sign In")

                if submit_button:
                    if username and password:
                        success, msg = login_user(username, password)
                        if success:
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = username
                            
                            # Cek apakah user ini adalah Admin (berdasarkan .env)
                            env_user = os.getenv("APP_USERNAME", "admin")
                            st.session_state['is_admin'] = (username == env_user)
                            
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Harap isi username dan password.")

            st.markdown("---")
            st.markdown("**Atau masuk dengan Google:**")
            
            # Simulasi Google Login (Input Email)
            google_email = st.text_input("Masukkan Gmail Anda", key="g_email_login")
            if st.button("Gunakan Akun Google", key="btn_google_login"):
                if "@gmail.com" in google_email:
                    success, msg, username_g = login_with_google(google_email)
                    if success:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username_g
                        st.session_state['is_admin'] = False # User Google dianggap user biasa (kecuali diatur lain)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.warning(msg) # Msg: Belum disetujui / Baru daftar
                else:
                    st.error("Harap gunakan email @gmail.com yang valid.")

        # --- TAB REGISTER ---
        with tab_register:
            st.info("Pendaftaran akun baru memerlukan persetujuan Admin.")
            with st.form("register_form"):
                new_user = st.text_input("Buat Username")
                new_email = st.text_input("Email (Opsional)")
                new_pass = st.text_input("Buat Password", type="password")
                
                reg_button = st.form_submit_button("Daftar Sekarang")
                
                if reg_button:
                    if new_user and new_pass:
                        success, msg = register_user(new_user, new_pass, new_email, "local")
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("Mohon lengkapi data.")

# ==========================================
# BAGIAN 2: LOGIC UTAMA (COMMON)
# ==========================================

def ensure_folder_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def get_database_files(folder_name):
    if not os.path.exists(folder_name):
        return []
    return [f for f in os.listdir(folder_name) if f.endswith('.db')]

def load_data_from_db(db_path, table_name="production_data"):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        if cursor.fetchone() is None:
            conn.close()
            return None
            
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

def clean_data(df):
    df.columns = df.columns.str.strip().str.title()
    rename_map = {
        'Oil Volume': 'Oil volume', 'Gas Volume': 'Gas volume',
        'Water Volume': 'Water volume', 'Volume Of Liquid': 'Volume of liquid',
        'Water Cut': 'Water cut', 'Working Hours': 'Working hours',
        'Dynamic Level': 'Dynamic level', 'Reservoir Pressure': 'Reservoir pressure'
    }
    df = df.rename(columns=rename_map)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    numeric_cols = ['Oil volume', 'Volume of liquid', 'Gas volume', 'Water volume', 
                    'Water cut', 'Working hours', 'Dynamic level', 'Reservoir pressure']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def download_excel_button(df, filename="data_download.xlsx", key_unique=None):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        processed_data = output.getvalue()
        return st.download_button(
            label="📥 Download Data (.xlsx)",
            data=processed_data,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key_unique
        )
    except Exception as e:
        st.error(f"Gagal menyiapkan download: {e}")

# --- FUNGSI GENERATE PDF REPORT ---
def create_pdf_report(title, content_dict, filename="report.pdf"):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Oil & Gas AI Dashboard Report', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def clean_text(text):
        return str(text).encode('latin-1', 'replace').decode('latin-1')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text(title), 0, 1, 'L')
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12)
    for key, value in content_dict.items():
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{clean_text(key)}:", 0, 1)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, clean_text(value))
        pdf.ln(2)
        
    return pdf.output(dest='S').encode('latin-1')

# --- FUNGSI OPENAI CHAT AGENT (PRODUCTION) ---
def ask_openai_about_data(df, user_question):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ Error: OPENAI_API_KEY tidak ditemukan di file .env"

    try:
        client = OpenAI(api_key=api_key)
        
        data_summary = df.describe().to_string()
        data_head = df.head(5).to_string()
        
        system_prompt = f"""
        Kamu adalah Asisten AI ahli Teknik Perminyakan (Petroleum Engineer).
        Tugasmu adalah menjawab pertanyaan user berdasarkan data sumur minyak yang diberikan.
        
        Konteks Data:
        1. Statistik Data (Describe):
        {data_summary}
        
        2. Sampel Data (5 Baris Pertama):
        {data_head}
        
        Panduan:
        - Jawab dengan bahasa Indonesia yang profesional namun mudah dimengerti.
        - Fokus pada insight teknis (tren produksi, masalah pompa, water cut, dll).
        - Jika data tidak cukup untuk menjawab, katakan jujur.
        """

        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"Terjadi kesalahan pada OpenAI: {str(e)}"

# --- FUNGSI OPENAI CHAT AGENT (WELL LOG) ---
def ask_openai_about_well_log(df, user_question, ml_context=""):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ Error: OPENAI_API_KEY tidak ditemukan di file .env"

    try:
        client = OpenAI(api_key=api_key)
        
        data_summary = df.describe().to_string()
        data_head = df.head(5).to_string()
        
        facies_info = ""
        if 'Facies' in df.columns:
             facies_counts = df['Facies'].value_counts().to_string()
             facies_info = f"\nDistribusi Facies (Jenis Batuan):\n{facies_counts}"

        system_prompt = f"""
        Kamu adalah Senior Petrophysicist dan Geologist AI.
        Tugasmu adalah menganalisis data Well Log (rekaman sumur) dan menjawab pertanyaan user.

        Konteks Data Log:
        1. Statistik Data Log (GR, Resistivity, Porosity, dll):
        {data_summary}

        2. Sampel Data:
        {data_head}
        {facies_info}
        
        3. Konteks Analisis ML Terakhir (Jika ada):
        {ml_context}

        Panduan Jawaban:
        - Jawab dalam bahasa Indonesia yang teknis (domain Petrophysics).
        - Jika ada 'Konteks Analisis ML Terakhir', bahas hasil training model atau prediksi manual user tersebut.
        - Gunakan istilah seperti Gamma Ray (GR) untuk indikasi Shale/Sand, Resistivity untuk indikasi hidrokarbon/air, dan Porosity (PHIND/NPHI).
        - Analisis potensi reservoir berdasarkan crossover log (jika ditanya).
        """

        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"Terjadi kesalahan pada OpenAI: {str(e)}"

# ==========================================
# BAGIAN 2: FUNGSI CLOUD (FIRESTORE)
# ==========================================

def get_firestore_collections():
    try:
        db = firestore.client()
        collections = db.collections()
        return [col.id for col in collections]
    except Exception as e:
        return []

def upload_to_firestore(df, collection_name):
    try:
        db = firestore.client()
        df_clean = df.where(pd.notnull(df), None)
        if 'Date' in df_clean.columns:
            df_clean['Date'] = df_clean['Date'].astype(str)

        records = df_clean.to_dict(orient='records')
        batch = db.batch()
        count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_records = len(records)
        
        for i, record in enumerate(records):
            doc_ref = db.collection(collection_name).document()
            batch.set(doc_ref, record)
            count += 1
            if count >= 400:
                batch.commit()
                batch = db.batch()
                count = 0
                progress_bar.progress(min(i / total_records, 1.0))
                status_text.text(f"Mengupload {i}/{total_records} data...")

        if count > 0: batch.commit()
        progress_bar.progress(1.0)
        status_text.text("Selesai!")
        return True, f"Berhasil mengupload {total_records} baris ke '{collection_name}'"
    except Exception as e:
        return False, str(e)

def get_data_from_firestore(collection_name, limit=500):
    try:
        db = firestore.client()
        docs = db.collection(collection_name).limit(limit).stream()
        data = []
        for doc in docs:
            data.append(doc.to_dict())
        if not data: return None, "Koleksi kosong/tidak ditemukan."
        df = pd.DataFrame(data)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date')
        return df, None
    except Exception as e:
        return None, str(e)

# ==========================================
# BAGIAN 3: FUNGSI ANALISIS & VISUALISASI
# ==========================================

def plot_annual_production(df):
    df['year'] = df['Date'].dt.year
    grp1 = df.groupby('year')[['Oil volume', 'Water volume', 'Volume of liquid']].sum().reset_index()
    grp1['Water cut (%)'] = (grp1['Water volume'] / grp1['Volume of liquid']) * 100
    
    fig = plt.figure(figsize=(15, 7))
    bottom_val = np.zeros(grp1["year"].unique().shape[0], dtype=float)
    cols = ["Water volume", "Oil volume"]
    colors = ["#1f77b4", "#2ca02c"]
    
    for idx, col in enumerate(cols):
        plt.bar(grp1["year"], grp1[col], bottom=bottom_val, color=colors[idx], label=f"{col} (m3)", alpha=0.8)
        bottom_val += grp1[col].to_numpy()

    plt.plot(grp1["year"], grp1["Volume of liquid"], marker='o', c="grey", linestyle="dotted", label="Total Liquid")
    
    max_y = grp1["Volume of liquid"].max()
    offset = max_y * 0.02 
    for i, year in enumerate(grp1["year"]):
        val_wc = grp1["Water cut (%)"].iloc[i]
        if pd.notnull(val_wc) and grp1["Volume of liquid"].iloc[i] > 0:
            plt.text(year, grp1["Volume of liquid"].iloc[i] + offset, f"{val_wc:.1f}%", ha='center', fontsize=10, fontweight='bold')

    plt.ylabel("Volume (m3)")
    plt.xlabel("Tahun")
    plt.title("Annual Production Summary", fontsize="14")
    plt.xticks(grp1["year"].unique())
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    return fig, grp1

def ai_forecast_production(df, days_ahead=90):
    data = df[['Date', 'Oil volume']].dropna()
    if len(data) < 10: return None, None, "Data kurang dari 10 baris."
    data['Date_Ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
    X = data[['Date_Ordinal']]
    y = data['Oil volume']
    model = LinearRegression()
    model.fit(X, y)
    
    last_date = data['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_pred = model.predict(future_ordinals)
    
    df_future = pd.DataFrame({'Date': future_dates, 'Predicted Oil Volume': future_pred})
    
    future_val = future_pred[-1]
    slope = model.coef_[0]
    
    if future_val < 5: insight = "⚠️ **KRITIS:** Prediksi < 5 m3. Butuh Workover."
    elif slope < 0: insight = "📉 **Tren Menurun.** Cek Artificial Lift."
    else: insight = "✅ **Stabil.** Lanjutkan monitoring."
    return df_future, model, insight

def ai_detect_anomalies(df):
    features = ['Oil volume', 'Water cut', 'Working hours']
    available = [f for f in features if f in df.columns]
    if len(available) < 2: return df, False, "Kolom tidak lengkap."
    data_ml = df[available].dropna()
    if len(data_ml) < 10: return df, False, "Data terlalu sedikit."

    iso = IsolationForest(contamination=0.05, random_state=42)
    df.loc[data_ml.index, 'Anomaly_Score'] = iso.fit_predict(data_ml)
    df['Is_Anomaly'] = df['Anomaly_Score'] == -1
    
    anomalies = df[df['Is_Anomaly'] == True]
    insight = "Tidak ada anomali signifikan."
    if not anomalies.empty:
        insight = f"⚠️ Ditemukan {len(anomalies)} data anomali (Pola tidak wajar)."
    return df, True, insight

def ai_calculate_rul(df, limit=2.0):
    data = df[['Date', 'Oil volume']].dropna()
    data = data[data['Oil volume'] > 0]
    if len(data) < 30: return None, "Data history < 30 hari."
    recent = data.tail(180).copy()
    start_date = recent['Date'].min()
    recent['Days'] = (recent['Date'] - start_date).dt.days
    
    X = recent[['Days']]
    y_log = np.log(recent['Oil volume'])
    model = LinearRegression()
    model.fit(X, y_log)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    if slope >= 0: return None, "Produksi NAIK/STABIL. Tidak bisa hitung RUL."
    t_end = (np.log(limit) - intercept) / slope
    date_end = start_date + timedelta(days=int(t_end))
    rem_days = (date_end - data['Date'].max()).days
    
    if rem_days < 0: insight = "⚠️ Sumur sudah melewati batas ekonomis."
    else: insight = f"📉 **Prediksi RUL:** Batas ({limit} m3) pada **{date_end.strftime('%d %B %Y')}** (sisa {rem_days} hari)."

    return {'slope': slope, 'intercept': intercept, 'end_date': date_end, 'insight': insight}, None

# ==========================================
# BAGIAN 4: HALAMAN-HALAMAN UI
# ==========================================

def page_upload():
    st.title("📥 Upload Data Center")
    st.markdown("Pilih jenis data yang ingin Anda upload ke **Database Lokal**.")

    tab1, tab2 = st.tabs(["🛢️ Production Data", "🪨 Well Log Data"])

    # --- TAB 1: PRODUCTION DATA ---
    with tab1:
        st.subheader("Upload Data Produksi Harian")
        
        # --- KETERANGAN FORMAT KOLOM ---
        st.info("ℹ️ **Format Kolom Wajib:** Date, Oil Volume, Volume of liquid, Gas volume, Water volume, Water cut, Working hours, Dynamic level, Reservoir pressure")
        
        uploaded_file = st.file_uploader("Pilih file CSV/Excel", type=["csv", "xlsx", "xls"], key="prod_uploader")
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
                else: df = pd.read_excel(uploaded_file)
                
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Simpan Data Produksi", key="save_prod"):
                    with st.spinner("Menyimpan ke database/production..."):
                        folder_name = "database/production" # Subfolder
                        ensure_folder_exists(folder_name)
                        
                        file_name_clean = os.path.splitext(uploaded_file.name)[0]
                        db_name = f"{file_name_clean}.db"
                        db_path = os.path.join(folder_name, db_name)
                        
                        df_clean = clean_data(df)
                        conn = sqlite3.connect(db_path)
                        df_clean.to_sql('production_data', conn, if_exists='replace', index=False)
                        conn.close()
                        st.success(f"Database tersimpan di: `{db_path}`")
            except Exception as e: st.error(f"Error: {e}")

    # --- TAB 2: WELL LOG DATA ---
    with tab2:
        st.subheader("Upload Data Well Log (Petrophysics)")
        
        # --- KETERANGAN FORMAT KOLOM ---
        st.info("ℹ️ **Format Kolom Wajib:** Facies, Formation, Well Name, Depth, GR, ILD_log10, DeltaPHI, PHIND, PE, NM_M, RELPOS")
        
        uploaded_log = st.file_uploader("Pilih file Log (.csv)", type=["csv"], key="log_uploader")
        
        if uploaded_log is not None:
            try:
                df_log = pd.read_csv(uploaded_log)
                st.dataframe(df_log.head(), use_container_width=True)
                
                if st.button("Simpan Data Well Log", key="save_log"):
                    with st.spinner("Menyimpan ke database/well_logs..."):
                        folder_name = "database/well_logs" # Subfolder beda
                        ensure_folder_exists(folder_name)
                        
                        file_name_clean = os.path.splitext(uploaded_log.name)[0]
                        db_name = f"{file_name_clean}.db"
                        db_path = os.path.join(folder_name, db_name)
                        
                        conn = sqlite3.connect(db_path)
                        # Simpan ke tabel 'well_log_data'
                        df_log.to_sql('well_log_data', conn, if_exists='replace', index=False)
                        conn.close()
                        st.success(f"Database tersimpan di: `{db_path}`")
            except Exception as e: st.error(f"Error: {e}")

def page_dashboard():
    st.title("📊 Dashboard Analisis Sumur (Lokal)")
    # Ambil dari folder production
    folder_name = "database/production"
    db_files = get_database_files(folder_name)
    
    if not db_files:
        st.warning("Belum ada database produksi. Silakan upload di menu **Upload Data > Production Data**.")
        return
        
    selected_db = st.selectbox("Pilih Sumur (Database Lokal):", db_files)
    if selected_db:
        db_path = os.path.join(folder_name, selected_db)
        df = load_data_from_db(db_path, "production_data") # Table name spesifik
        
        if df is not None:
            if 'Date' in df.columns:
                df = df.sort_values('Date')
                df_chart = df.set_index('Date')
            else:
                st.error("Kolom 'Date' tidak ditemukan.")
                return

            # --- VISUALISASI UTAMA ---
            st.subheader("Ringkasan Produksi")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Oil Volume", f"{df['Oil volume'].sum():,.0f}" if 'Oil volume' in df.columns else 0)
            with col2: st.metric("Rata-rata Water Cut", f"{df['Water cut'].mean():.2f} %" if 'Water cut' in df.columns else 0)
            with col3: st.metric("Avg Pressure", f"{df['Reservoir pressure'].mean():.2f}" if 'Reservoir pressure' in df.columns else 0)
            with col4: st.metric("Total Data Points", f"{len(df)} Hari")

            st.markdown("---")
            # --- PDF REPORT BUTTON (GENERAL STATS) ---
            report_content_prod = {
                "Analysis Type": "Well Production Summary",
                "Well Name": selected_db.replace('.db', ''),
                "Total Oil Production": f"{df['Oil volume'].sum():,.0f} m3",
                "Average Water Cut": f"{df['Water cut'].mean():.2f} %",
                "Average Pressure": f"{df['Reservoir pressure'].mean():.2f} atm",
                "Data Points": f"{len(df)} days"
            }
            pdf_data_prod = create_pdf_report("Production Analysis Report", report_content_prod)
            st.download_button("📄 Download Summary PDF", pdf_data_prod, file_name="production_summary.pdf", mime="application/pdf")

            # --- CHART 1: PRODUKSI HARIAN ---
            st.subheader("📈 Tren Produksi Harian")
            potential_cols = ['Oil volume', 'Water volume', 'Volume of liquid', 'Gas volume']
            available_cols = [c for c in potential_cols if c in df.columns]
            default_selection = [c for c in ['Oil volume', 'Water volume'] if c in available_cols]
            
            if available_cols:
                selected_cols = st.multiselect("Pilih Parameter Produksi:", options=available_cols, default=default_selection)
                if selected_cols:
                    st.line_chart(df_chart[selected_cols])
                    download_excel_button(df[['Date'] + selected_cols], "daily_production_trend.xlsx", "btn_trend")

            # --- CHART 2: PRESSURE & DYNAMIC ---
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if 'Reservoir pressure' in df.columns:
                    st.markdown("**Reservoir Pressure**")
                    st.line_chart(df_chart['Reservoir pressure'], color="#FF5733")
                    download_excel_button(df[['Date', 'Reservoir pressure']], "reservoir_pressure.xlsx", "btn_press")
            with col_p2:
                if 'Dynamic level' in df.columns:
                    st.markdown("**Dynamic Level**")
                    st.line_chart(df_chart['Dynamic level'], color="#33FF57")
                    download_excel_button(df[['Date', 'Dynamic level']], "dynamic_level.xlsx", "btn_dyn")

            # --- CHART 3: WATER CUT & HOURS ---
            st.subheader("⏱️ Water Cut & Working Hours")
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                if 'Water cut' in df.columns:
                    st.markdown("**Water Cut (%)**")
                    st.area_chart(df_chart['Water cut'])
                    download_excel_button(df[['Date', 'Water cut']], "water_cut.xlsx", "btn_wc")
            with col_w2:
                if 'Working hours' in df.columns:
                    st.markdown("**Working Hours**")
                    st.bar_chart(df_chart['Working hours'])
                    download_excel_button(df[['Date', 'Working hours']], "working_hours.xlsx", "btn_wh")

            st.markdown("---")
            # --- CHART 4: ANNUAL PRODUCTION ---
            st.subheader("📆 Analisis Produksi Tahunan (Annual Production)")
            fig_annual, df_annual_data = plot_annual_production(df)
            st.pyplot(fig_annual)
            download_excel_button(df_annual_data, "annual_production_summary.xlsx", "btn_annual")

            with st.expander("Lihat Data Mentah Lengkap"):
                st.dataframe(df, use_container_width=True)
                download_excel_button(df, f"raw_data_{selected_db}.xlsx", "btn_raw_local")

            # --- CHATBOX AGENT (OPENAI) ---
            st.markdown("---")
            st.subheader("💬 AI Data Analyst Assistant")
            st.info("Tanyakan apa saja mengenai data sumur ini. AI akan menganalisis statistik data di atas.")
            
            with st.form("chat_form"):
                user_question = st.text_input("Pertanyaan Anda:", placeholder="Contoh: Bagaimana tren produksi minyak bulan ini? Apakah ada anomali?")
                submitted = st.form_submit_button("Tanya AI")
                
                if submitted and user_question:
                    with st.spinner("AI sedang menganalisis data..."):
                        # Panggil fungsi OpenAI dengan Dataframe saat ini
                        response = ask_openai_about_data(df, user_question)
                        st.markdown(f"**Jawaban AI:**")
                        st.write(response)

def page_ai_analysis():
    st.title("🤖 AI Engineering Assistant")
    folder_name = "database/production"
    db_files = get_database_files(folder_name)
    if not db_files:
        st.warning("Belum ada database produksi.")
        return
    selected_db = st.selectbox("Pilih Sumur untuk Analisis AI:", db_files)
    if selected_db:
        db_path = os.path.join(folder_name, selected_db)
        df = load_data_from_db(db_path, "production_data")
        if df is not None and 'Date' in df.columns:
            df = df.sort_values('Date')
            tab1, tab2, tab3 = st.tabs(["🔮 Forecasting", "⚠️ Anomaly", "📉 Decline Curve (RUL)"])
            
            with tab1:
                st.subheader("Peramalan Produksi")
                days = st.slider("Jumlah Hari Prediksi", 30, 90, 60)
                if st.button("Jalankan Forecast"):
                    with st.spinner("Menghitung model..."):
                        df_future, model, insight = ai_forecast_production(df, days)
                        if df_future is not None:
                            st.info(insight)
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.plot(df['Date'], df['Oil volume'], label='Historis', color='blue')
                            ax.plot(df_future['Date'], df_future['Predicted Oil Volume'], label='Forecast', color='orange', linestyle='--')
                            ax.legend()
                            st.pyplot(fig)
                            download_excel_button(df_future, "forecast_result.xlsx", "btn_forecast")
                            
                            # --- PDF REPORT BUTTON ---
                            report_content = {
                                "Analysis Type": "Production Forecasting",
                                "Forecast Days": f"{days} Days",
                                "Insight": insight,
                                "Last Historical Date": df['Date'].max().strftime('%Y-%m-%d'),
                                "Last Production": f"{df['Oil volume'].iloc[-1]} m3"
                            }
                            pdf_data = create_pdf_report("Production Forecast Report", report_content)
                            st.download_button("📄 Download Report PDF", pdf_data, file_name="forecast_report.pdf", mime="application/pdf")
                            
                        else: st.error(insight)

            with tab2:
                st.subheader("Deteksi Anomali")
                if st.button("Deteksi Masalah"):
                    with st.spinner("Scanning data..."):
                        df_anom, success, msg = ai_detect_anomalies(df)
                        if success:
                            st.info(f"💡 {msg}")
                            anomalies = df_anom[df_anom['Is_Anomaly'] == True]
                            
                            fig, ax = plt.subplots(figsize=(10, 5))
                            normal = df_anom[df_anom['Is_Anomaly'] == False]
                            ax.scatter(normal['Date'], normal['Oil volume'], c='blue', alpha=0.5, s=15, label='Normal')
                            ax.scatter(anomalies['Date'], anomalies['Oil volume'], c='red', s=60, marker='x', label='Anomaly')
                            ax.legend()
                            st.pyplot(fig)
                            
                            if not anomalies.empty:
                                st.write("Data Anomali:")
                                st.dataframe(anomalies[['Date', 'Oil volume', 'Working hours', 'Water cut']], use_container_width=True)
                                download_excel_button(anomalies, "anomaly_data.xlsx", "btn_anomaly")
                                
                                # --- PDF REPORT BUTTON ---
                                report_content = {
                                    "Analysis Type": "Anomaly Detection",
                                    "Total Anomalies Found": len(anomalies),
                                    "Insight": msg,
                                    "Note": "Please check the pump efficiency for anomaly dates."
                                }
                                pdf_data = create_pdf_report("Anomaly Detection Report", report_content)
                                st.download_button("📄 Download Report PDF", pdf_data, file_name="anomaly_report.pdf", mime="application/pdf")
                                
                        else: st.warning(msg)

            with tab3:
                st.subheader("Remaining Useful Life (RUL)")
                limit = st.number_input("Batas Ekonomis (m3)", value=2.0)
                if st.button("Hitung RUL"):
                    with st.spinner("Menghitung..."):
                        rul, err = ai_calculate_rul(df, limit)
                        if rul:
                            st.info(rul['insight'])
                            fig, ax = plt.subplots(figsize=(10, 5))
                            recent = df.tail(180)
                            ax.scatter(recent['Date'], recent['Oil volume'], color='gray', alpha=0.5)
                            days_diff = (rul['end_date'] - recent['Date'].min()).days
                            future = np.arange(0, days_diff + 30)
                            pred = np.exp(rul['intercept']) * np.exp(rul['slope'] * future)
                            dates = [recent['Date'].min() + timedelta(days=int(x)) for x in future]
                            ax.plot(dates, pred, color='red', linestyle='--')
                            ax.axhline(y=limit, color='green', linestyle=':')
                            st.pyplot(fig)
                            
                            # --- PDF REPORT BUTTON ---
                            report_content = {
                                "Analysis Type": "Remaining Useful Life (RUL)",
                                "Economic Limit": f"{limit} m3",
                                "Predicted End Date": rul['end_date'].strftime('%d %B %Y'),
                                "Insight": rul['insight']
                            }
                            pdf_data = create_pdf_report("RUL Analysis Report", report_content)
                            st.download_button("📄 Download Report PDF", pdf_data, file_name="rul_report.pdf", mime="application/pdf")
                            
                        else: st.warning(err)

def page_well_log_analysis():
    st.title("🪨 Well Log Analysis (Petrophysics)")
    
    # Ambil data dari folder well_logs (bukan upload ulang terus)
    folder_name = "database/well_logs"
    db_files = get_database_files(folder_name)
    
    if not db_files:
        st.warning("Belum ada data Well Log. Silakan upload di menu **Upload Data > Well Log Data**.")
        return

    # Dropdown pilih DB Well Log
    selected_db = st.selectbox("Pilih Database Log:", db_files)
    
    if selected_db:
        db_path = os.path.join(folder_name, selected_db)
        # Load tabel 'well_log_data'
        df = load_data_from_db(db_path, "well_log_data")
        
        if df is not None:
            st.write("### Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # 2. Visualisasi Log
            st.subheader("📊 Visualisasi Well Log")
            
            available_wells = df['Well Name'].unique() if 'Well Name' in df.columns else ['Unknown']
            selected_well = st.selectbox("Pilih Sumur:", available_wells)
            
            if 'Well Name' in df.columns:
                plot_df = df[df['Well Name'] == selected_well].sort_values('Depth')
            else:
                plot_df = df.sort_values('Depth')

            if st.checkbox("Tampilkan Plot Log"):
                fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 10), sharey=True)
                fig.suptitle(f"Well Log Plot: {selected_well}", fontsize=16)
                
                ax[0].plot(plot_df['GR'], plot_df['Depth'], color='green'); ax[0].set_xlabel("GR (API)"); ax[0].set_xlim(0, 200); ax[0].grid(True)
                ax[1].plot(plot_df['ILD_log10'], plot_df['Depth'], color='red'); ax[1].set_xlabel("Resistivity (log10)"); ax[1].set_xlim(-1, 3); ax[1].grid(True)
                ax[2].plot(plot_df['PHIND'], plot_df['Depth'], color='blue'); ax[2].set_xlabel("Porosity PHIND (%)"); ax[2].set_xlim(0, 50); ax[2].invert_xaxis(); ax[2].grid(True)
                ax[3].scatter(plot_df['Facies'], plot_df['Depth'], c=plot_df['Facies'], cmap='Set1', s=20); ax[3].set_xlabel("Facies"); ax[3].set_xlim(0, 10); ax[3].grid(True)

                plt.gca().invert_yaxis()
                st.pyplot(fig)

            # --- CHATBOT AI PETROPHYSICIST (ADDED HERE) ---
            st.markdown("---")
            st.subheader("💬 AI Petrophysicist Assistant")
            st.info("Tanyakan apa saja tentang data log sumur ini. AI akan menganalisis parameter GR, Resistivity, Porosity, dan litologi.")

            with st.form("chat_log_form"):
                user_q_log = st.text_input("Pertanyaan Analisis Log:", placeholder="Misal: Apakah ada indikasi zona hidrokarbon di sumur ini?")
                submit_log = st.form_submit_button("Tanya AI Log Analyst")

                if submit_log and user_q_log:
                    with st.spinner("AI Petrophysicist sedang menganalisis data log..."):
                        # Filter data spesifik sumur yang dipilih agar AI lebih akurat
                        context_df = plot_df if 'plot_df' in locals() else df
                        
                        # --- Cek apakah ada context ML sebelumnya ---
                        ml_context = st.session_state.get('log_ai_context', "")
                        
                        response_log = ask_openai_about_well_log(context_df, user_q_log, ml_context)
                        st.markdown(f"**Jawaban AI Petrophysicist:**")
                        st.write(response_log)

            st.markdown("---")

            # 3. Machine Learning
            st.subheader("🤖 Automated Lithology Classification")
            st.markdown("Latih model **Random Forest** untuk memprediksi jenis batuan.")

            FACIES_LABELS = {
                1: "Sandstone", 2: "Cinstone", 3: "Shale", 4: "Marl", 
                5: "Dolomite", 6: "Limestone", 7: "Chert", 8: "Tuff", 9: "Anhydrite"
            }
            
            # --- INITIALIZE SESSION STATE UNTUK KONTEKS AI (WELL LOG) ---
            if 'log_ai_context' not in st.session_state:
                st.session_state['log_ai_context'] = ""
            
            if 'rf_model' not in st.session_state:
                st.session_state['rf_model'] = None

            if st.button("🚀 Train Model"):
                with st.spinner("Melatih model AI..."):
                    df_clean = df.dropna()
                    features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
                    target = 'Facies'
                    
                    missing = [c for c in features + [target] if c not in df_clean.columns]
                    if missing:
                        st.error(f"Kolom hilang: {missing}")
                    else:
                        X, y = df_clean[features], df_clean[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        clf = RandomForestClassifier(n_estimators=100, random_state=42)
                        clf.fit(X_train, y_train)
                        st.session_state['rf_model'] = clf
                        
                        acc = accuracy_score(y_test, clf.predict(X_test))
                        
                        # --- UPDATE CONTEXT AI ---
                        train_msg = f"User baru saja melatih model Random Forest. Akurasi model: {acc*100:.2f}%. Model siap digunakan untuk prediksi."
                        st.session_state['log_ai_context'] = train_msg
                        
                        st.success(f"Model Dilatih! Akurasi: **{acc*100:.2f}%**")
                        
                        st.write("#### Feature Importance:")
                        imp = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_}).sort_values('Importance', ascending=False)
                        st.bar_chart(imp.set_index('Feature'))
                        
                        # --- PDF REPORT BUTTON ---
                        report_content = {
                            "Analysis Type": "Lithology Classification Training",
                            "Model Used": "Random Forest",
                            "Training Accuracy": f"{acc*100:.2f}%",
                            "Top Feature": imp.iloc[0]['Feature'],
                            "Dataset Size": f"{len(df_clean)} samples"
                        }
                        pdf_data = create_pdf_report("Machine Learning Training Report", report_content)
                        st.download_button("📄 Download Training Report PDF", pdf_data, file_name="ml_training_report.pdf", mime="application/pdf")

            # D. Prediksi Demo
            if st.session_state['rf_model'] is not None:
                st.markdown("---")
                st.write("#### 🕵️ Uji Coba Prediksi Manual")
                
                pred_mode = st.radio("Metode Input:", ["Manual Input", "Ambil dari Data Sumur"])
                input_data = None
                selected_info = ""
                
                if pred_mode == "Manual Input":
                    c1, c2, c3 = st.columns(3)
                    with c1: val_gr = st.number_input("Gamma Ray (GR)", value=60.0)
                    with c2: val_ild = st.number_input("Resistivity (log10)", value=0.6)
                    with c3: val_phi = st.number_input("Porosity (PHIND)", value=12.0)
                    
                    c4, c5, c6 = st.columns(3)
                    with c4: val_pe = st.number_input("Photoelectric (PE)", value=3.5)
                    with c5: val_deltaphi = st.number_input("DeltaPHI", value=3.0)
                    with c6: val_relpos = st.number_input("Relative Position", value=0.5)
                    
                    if st.button("🔍 Tebak Jenis Batuan"):
                        input_data = pd.DataFrame({
                            'GR': [val_gr], 'ILD_log10': [val_ild], 'DeltaPHI': [val_deltaphi], 
                            'PHIND': [val_phi], 'PE': [val_pe], 'NM_M': [1], 'RELPOS': [val_relpos]
                        })
                        selected_info = "Manual Input Data"

                else: # Ambil dari Data Sumur
                    if 'Depth' in plot_df.columns:
                        selected_depth = st.selectbox(f"Pilih Kedalaman di Sumur {selected_well}:", plot_df['Depth'].unique())
                        row_data = plot_df[plot_df['Depth'] == selected_depth].iloc[0]
                        
                        st.info(f"Data Log pada kedalaman {selected_depth}: GR={row_data['GR']}, PHI={row_data['PHIND']}, ILD={row_data['ILD_log10']}")
                        
                        if st.button("🔍 Tebak Batuan di Kedalaman Ini"):
                            input_data = pd.DataFrame([row_data[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']]])
                            selected_info = f"Well: {selected_well}, Depth: {selected_depth}"

                if input_data is not None:
                    model = st.session_state['rf_model']
                    pred_facies = model.predict(input_data)[0]
                    label_name = FACIES_LABELS.get(pred_facies, "Unknown")
                    
                    # --- UPDATE CONTEXT AI ---
                    pred_msg = f"User baru saja melakukan prediksi litologi. Input Source: {selected_info}. Hasil Prediksi: Facies {pred_facies} ({label_name})."
                    st.session_state['log_ai_context'] = pred_msg
                    
                    st.success(f"### Hasil Prediksi: **Facies {pred_facies} - {label_name}**")
                    
                    # --- PDF REPORT BUTTON ---
                    report_content = {
                        "Analysis Type": "Lithology Prediction",
                        "Source": selected_info,
                        "Predicted Facies": f"{pred_facies} ({label_name})",
                        "Input Values": str(input_data.to_dict(orient='records')[0])
                    }
                    pdf_data = create_pdf_report("Lithology Prediction Report", report_content)
                    st.download_button("📄 Download Prediction Report PDF", pdf_data, file_name="prediction_report.pdf", mime="application/pdf")

def page_sync_cloud():
    st.title("☁️ Sync to Firestore")
    if not firebase_admin._apps:
        st.error("Firebase belum terinisialisasi.")
        return
        
    # --- PILIH TIPE DATA YANG MAU DISYNC ---
    data_type = st.radio("Pilih Tipe Data untuk Diupload:", ["Production Data", "Well Log Data"])
    
    # Set folder & collection default berdasarkan pilihan
    if data_type == "Production Data":
        folder_name = "database/production"
        table_name = "production_data"
        default_prefix = "prod_"
    else:
        folder_name = "database/well_logs"
        table_name = "well_log_data"
        default_prefix = "log_"

    db_files = get_database_files(folder_name)
    
    if not db_files: 
        st.warning(f"Belum ada database di folder `{folder_name}`.")
        return

    col1, col2 = st.columns(2)
    with col1: selected_db = st.selectbox("Pilih Database Lokal:", db_files)
    with col2:
        # Auto name suggestion
        clean_name = os.path.splitext(selected_db)[0]
        collection_name = st.text_input("Nama Koleksi di Firestore:", value=clean_name)

    if st.button(f"🚀 Upload {data_type} ke Cloud"):
        db_path = os.path.join(folder_name, selected_db)
        # Load dengan nama tabel yang sesuai
        df = load_data_from_db(db_path, table_name)
        
        if df is not None:
            st.write(f"Membaca {len(df)} baris data...")
            success, msg = upload_to_firestore(df, collection_name)
            if success: st.success(msg); st.balloons()
            else: st.error(f"Gagal: {msg}")

def page_cloud_dashboard():
    st.title("🔥 Cloud Live Dashboard")
    if not firebase_admin._apps:
        st.error("Firebase belum terinisialisasi.")
        return

    with st.spinner("Mengambil daftar koleksi dari Firestore..."):
        available_collections = get_firestore_collections()

    if not available_collections:
        st.warning("Belum ada koleksi ditemukan di Firestore atau koneksi bermasalah.")
        collection_name = st.text_input("Masukkan Nama Koleksi Manual:", value="weather_data")
    else:
        collection_name = st.selectbox("Pilih Sumur/Koleksi (Cloud):", available_collections)
    
    limit_data = st.slider("Jumlah Data Terbaru", 100, 5000, 1000)
    
    if st.button("Load Data from Cloud"):
        with st.spinner("Mengambil data..."):
            df_cloud, error = get_data_from_firestore(collection_name, limit_data)
            
            if error:
                st.warning(f"Gagal mengambil data: {error}")
            elif df_cloud is not None:
                st.success(f"Berhasil memuat {len(df_cloud)} data terbaru dari koleksi: {collection_name}")
                
                st.subheader("Ringkasan Data Cloud")
                st.dataframe(df_cloud.head()) # Preview singkat
                st.metric("Total Records", len(df_cloud))
                
                # Cek tipe data untuk menentukan visualisasi yang cocok
                # Jika ada 'Oil volume', anggap Production Data
                if 'Oil volume' in df_cloud.columns and 'Date' in df_cloud.columns:
                    st.subheader("Grafik Produksi (Cloud)")
                    df_chart = df_cloud.set_index('Date')
                    st.line_chart(df_chart[['Oil volume', 'Water volume']])
                
                # Jika ada 'GR', anggap Well Log Data
                elif 'GR' in df_cloud.columns and 'Depth' in df_cloud.columns:
                    st.subheader("Grafik Well Log (Cloud)")
                    st.line_chart(df_cloud.set_index('Depth')[['GR']])
                
                with st.expander("Lihat Data Mentah Firestore"):
                    st.dataframe(df_cloud, use_container_width=True)
                    download_excel_button(df_cloud, "cloud_raw_data.xlsx", "btn_cloud_raw")

# ==========================================
# MENU UTAMA (SIDEBAR & ROUTING)
# ==========================================

def main():
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    
    # Menu routing logic
    if not st.session_state['logged_in']:
        check_login_page()
    else:
        st.sidebar.title("Oil and Gas Dashboard Analysis")
        st.sidebar.success(f"Welcome, {st.session_state.get('username', 'User')}")
        
        # Opsi Admin
        if st.session_state.get('is_admin', False):
            st.sidebar.info("🔧 Admin Access Active")
            
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state.pop('username', None)
            st.session_state.pop('is_admin', None)
            st.rerun()

        # Daftar Menu
        menu_options = [
            "📥 Upload Data", 
            "📊 Well Production Analysis", 
            "🤖 AI Well Production Analysis", 
            "🪨 Well Log Analysis (Petrophysics)",
            "☁️ Sync to Cloud", 
            "🔥 Get Data from Cloud"
        ]
        
        # Tambahkan menu Admin HANYA jika login sebagai admin
        if st.session_state.get('is_admin', False):
            menu_options.append("🔒 Admin Panel")

        menu = st.sidebar.radio("Pilih Menu:", menu_options)

        if menu == "📥 Upload Data": page_upload()
        elif menu == "📊 Well Production Analysis": page_dashboard()
        elif menu == "🤖 AI Well Production Analysis": page_ai_analysis()
        elif menu == "🪨 Well Log Analysis (Petrophysics)": page_well_log_analysis()
        elif menu == "☁️ Sync to Cloud": page_sync_cloud()
        elif menu == "🔥 Get Data from Cloud": page_cloud_dashboard()
        elif menu == "🔒 Admin Panel": page_admin_panel()

if __name__ == "__main__":
    main()
