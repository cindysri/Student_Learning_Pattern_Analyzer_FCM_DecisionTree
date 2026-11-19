import os
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "mahasiswa.csv")
MODEL_PATH = os.path.join(BASE, "models", "fcm_decision_tree.joblib")

FEATURES = [
    "jam_belajar_per_minggu",
    "kehadiran_persen",
    "rata_nilai_tugas",
    "rata_nilai_kuis",
]
NAMES = [
    "Jam belajar/minggu",
    "Kehadiran (%)",
    "Nilai tugas",
    "Nilai kuis",
]
COLUMN_GUIDE = {
    "jam_belajar_per_minggu": "Jam belajar per minggu (angka 0–80)",
    "kehadiran_persen": "Persentase kehadiran (0–100)",
    "rata_nilai_tugas": "Rata-rata nilai tugas (0–100)",
    "rata_nilai_kuis": "Rata-rata nilai kuis (0–100)",
}

st.set_page_config(
    page_title="Student Learning Pattern Analyzer",
    page_icon="📚",
    layout="wide",
)


@st.cache_data
def load_default_data():
    df = pd.read_csv(DATA_PATH)
    return validate_dataset(df)


def validate_dataset(raw_df):
    """Validasi dan rapikan dataset yang diberikan pengguna."""
    df = raw_df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(
            "Kolom wajib belum lengkap: "
            + ", ".join(missing)
            + ". Nama kolom harus persis seperti panduan di sidebar."
        )

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Hanya buang baris yang tidak memiliki nilai valid pada fitur wajib.
    before = len(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    removed = before - len(df)

    if len(df) < 6:
        raise ValueError(
            f"Dataset hanya memiliki {len(df)} baris valid. "
            "Unggah minimal 6 baris dengan angka valid pada semua fitur."
        )

    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Dataset mengandung nilai tak hingga atau tidak valid.")

    bounds = {
        "jam_belajar_per_minggu": (0, 80),
        "kehadiran_persen": (0, 100),
        "rata_nilai_tugas": (0, 100),
        "rata_nilai_kuis": (0, 100),
    }
    for column, (low, high) in bounds.items():
        invalid = ~df[column].between(low, high)
        if invalid.any():
            raise ValueError(
                f"Kolom '{column}' memiliki nilai di luar rentang "
                f"{low}–{high}. Periksa data yang diunggah."
            )

    # FCM bermasalah bila seluruh fitur konstan.
    if (df[FEATURES].nunique() <= 1).all():
        raise ValueError(
            "Semua fitur bernilai sama. Unggah data yang memiliki variasi."
        )

    return df, removed


def make_artifact(df, n_clusters, fuzzifier, max_iter, error):
    X = df[FEATURES].to_numpy(dtype=float)

    if len(X) <= n_clusters:
        raise ValueError(
            f"Jumlah data ({len(X)}) harus lebih besar dari jumlah cluster "
            f"({n_clusters})."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    centers, U, U0, distances, objective_history, iterations, fpc = (
        fuzz.cluster.cmeans(
            X_scaled.T,
            c=n_clusters,
            m=fuzzifier,
            error=error,
            maxiter=max_iter,
            seed=42,
        )
    )

    dominant = np.argmax(U, axis=0)

    # Split tanpa stratify supaya dataset kecil/tidak seimbang tidak gagal.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        dominant,
        test_size=0.25,
        random_state=42,
    )

    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=2,
        random_state=42,
    )
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    agreement = accuracy_score(y_test, y_pred)

    try:
        silhouette = (
            silhouette_score(X_scaled, dominant)
            if len(np.unique(dominant)) > 1 and len(np.unique(dominant)) < len(X)
            else None
        )
    except Exception:
        silhouette = None

    artifact = {
        "scaler": scaler,
        "centers": centers,
        "U": U,
        "dominant": dominant,
        "tree": tree,
        "features": FEATURES,
        "n_clusters": n_clusters,
        "m": fuzzifier,
        "error": error,
        "max_iter": max_iter,
        "iterations": iterations,
        "fpc": fpc,
        "objective_history": objective_history,
        "agreement": agreement,
        "silhouette": silhouette,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    return artifact


def membership_for_new(x_scaled, centers, m):
    distances = np.linalg.norm(centers - x_scaled.reshape(1, -1), axis=1)

    if np.any(distances < 1e-12):
        membership = np.zeros(len(centers))
        membership[np.argmin(distances)] = 1.0
        return membership

    power = 2.0 / (m - 1.0)
    inverse_distances = distances ** (-power)
    return inverse_distances / inverse_distances.sum()


def describe_centers(centers_original):
    result = []
    for center in centers_original:
        # Deskripsi sederhana, bukan kategori resmi kemampuan akademik.
        score = np.mean([
            center[0] / 20.0,
            center[1] / 100.0,
            center[2] / 100.0,
            center[3] / 100.0,
        ])
        if score >= 0.72:
            name = "Profil relatif tinggi"
        elif score >= 0.48:
            name = "Profil relatif sedang"
        else:
            name = "Profil relatif perlu perhatian"
        result.append(name)
    return result


st.title("Student Learning Pattern Analyzer")
st.caption(
    "Analisis pola belajar mahasiswa dengan Fuzzy C-Means dan Decision Tree"
)

with st.sidebar:
    st.header("Dataset")
    st.write("Kamu bisa memakai data simulasi bawaan atau mengunggah file CSV sendiri.")
    uploaded_file = st.file_uploader(
        "Unggah dataset (.csv)",
        type=["csv"],
        help="CSV harus memiliki empat kolom wajib yang ditampilkan di bawah.",
    )

    with st.expander("Format kolom CSV wajib", expanded=False):
        for key, description in COLUMN_GUIDE.items():
            st.code(key, language="text")
            st.caption(description)
        st.write("Contoh: setiap baris adalah satu mahasiswa; setiap kolom berisi angka.")

    if uploaded_file is None:
        try:
            df, removed_rows = load_default_data()
            data_source = "Data simulasi bawaan"
        except Exception as exc:
            st.error(f"Gagal membaca dataset bawaan: {exc}")
            st.stop()
    else:
        try:
            uploaded_bytes = uploaded_file.getvalue()
            file_signature = hashlib.md5(uploaded_bytes).hexdigest()
            raw_upload = pd.read_csv(pd.io.common.BytesIO(uploaded_bytes))
            df, removed_rows = validate_dataset(raw_upload)
            data_source = f"Dataset unggahan: {uploaded_file.name}"
            st.success(f"CSV berhasil dibaca. Baris valid: {len(df)}.")
            if removed_rows:
                st.warning(
                    f"{removed_rows} baris diabaikan karena nilai fitur wajib kosong/tidak valid."
                )
        except Exception as exc:
            st.error(f"Dataset tidak dapat digunakan: {exc}")
            st.stop()

    st.caption(f"Sumber aktif: {data_source}")
    st.caption(f"Jumlah data valid: {len(df)}")

    st.divider()
    st.header("Parameter FCM")
    max_clusters = min(5, len(df) - 1)
    cluster_options = list(range(2, max_clusters + 1))
    if not cluster_options:
        st.error("Dataset belum cukup untuk menjalankan FCM.")
        st.stop()

    c = st.selectbox(
        "Jumlah cluster (c)",
        cluster_options,
        index=min(1, len(cluster_options) - 1),
    )
    m = st.slider("Fuzzifier (m)", 1.5, 3.0, 2.0, 0.1)
    max_iter = st.slider("Maksimum iterasi", 100, 2000, 1000, 100)
    error = st.select_slider(
        "Toleransi berhenti",
        options=[0.0001, 0.001, 0.005, 0.01],
        value=0.005,
    )
    run = st.button(
        "Jalankan / latih ulang model",
        type="primary",
        use_container_width=True,
    )

# Include data content in the key so uploading a different CSV retrains the models.
data_hash = hashlib.md5(df[FEATURES].to_csv(index=False).encode("utf-8")).hexdigest()
model_key = (data_hash, c, m, max_iter, error)

if (
    run
    or "artifact" not in st.session_state
    or st.session_state.get("model_key") != model_key
):
    try:
        with st.spinner("Menghitung FCM dan melatih Decision Tree..."):
            st.session_state.artifact = make_artifact(df, c, m, max_iter, error)
            st.session_state.model_key = model_key
    except Exception as exc:
        st.error(f"Model gagal dijalankan: {exc}")
        st.stop()

artifact = st.session_state.artifact
centers_original = artifact["scaler"].inverse_transform(artifact["centers"])
profiles = describe_centers(centers_original)

if uploaded_file is None:
    st.warning(
        "Dataset aktif adalah DATA SIMULASI untuk prototipe, bukan data mahasiswa nyata."
    )
else:
    st.info(
        "Dataset unggahan sedang digunakan. Pastikan kamu memiliki izin untuk "
        "mengolah data tersebut dan jangan unggah data pribadi yang tidak diperlukan."
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["Analisis mahasiswa", "Dashboard FCM", "Decision Tree", "Metode & evaluasi"]
)

with tab1:
    st.subheader("Input aktivitas belajar")
    st.caption("Masukkan data satu mahasiswa untuk melihat keanggotaan FCM dan prediksi Decision Tree.")

    with st.form("student_input"):
        col1, col2 = st.columns(2)
        with col1:
            study = st.number_input("Jam belajar per minggu", 0.0, 80.0, 8.0, 0.5)
            attend = st.number_input("Kehadiran (%)", 0.0, 100.0, 80.0, 1.0)
        with col2:
            task = st.number_input("Rata-rata nilai tugas", 0.0, 100.0, 78.0, 1.0)
            quiz = st.number_input("Rata-rata nilai kuis", 0.0, 100.0, 75.0, 1.0)
        submit = st.form_submit_button("Analisis pola belajar", type="primary", use_container_width=True)

    if submit:
        row = np.array([[study, attend, task, quiz]], dtype=float)
        x_scaled = artifact["scaler"].transform(row)[0]
        memberships = membership_for_new(x_scaled, artifact["centers"], artifact["m"])
        dominant_cluster = int(np.argmax(memberships))
        tree_label = int(artifact["tree"].predict(row)[0])

        left, right = st.columns(2)
        with left:
            st.metric("Cluster dominan FCM", f"Cluster {dominant_cluster + 1}")
            st.write(f"**Interpretasi centroid:** {profiles[dominant_cluster]}")
        with right:
            st.metric("Prediksi Decision Tree", f"Cluster {tree_label + 1}")
            st.write("Decision Tree memprediksi label dominan FCM; hasilnya dapat berbeda dari FCM langsung.")

        membership_df = pd.DataFrame({
            "Cluster": [f"Cluster {i + 1}" for i in range(c)],
            "Derajat keanggotaan": memberships,
            "Persentase": [f"{v * 100:.2f}%" for v in memberships],
            "Profil centroid": profiles,
        })
        st.write("**Derajat keanggotaan FCM**")
        st.dataframe(membership_df, hide_index=True, use_container_width=True)
        st.plotly_chart(
            px.bar(membership_df, x="Cluster", y="Derajat keanggotaan",
                   title="Derajat keanggotaan terhadap setiap cluster"),
            use_container_width=True,
        )
        st.caption(
            "Derajat keanggotaan bukan probabilitas kelulusan atau diagnosis kemampuan. "
            "Interpretasi profil bersifat deskriptif berdasarkan centroid."
        )

with tab2:
    st.subheader("Ringkasan hasil Fuzzy C-Means")
    result = df.copy()
    for j in range(c):
        result[f"keanggotaan_cluster_{j + 1}"] = artifact["U"][j]
    result["cluster_dominan"] = artifact["dominant"] + 1

    k1, k2, k3 = st.columns(3)
    k1.metric("Jumlah data valid", len(df))
    k2.metric("Iterasi FCM", artifact["iterations"])
    k3.metric("Fuzzy Partition Coefficient", f"{artifact['fpc']:.4f}")

    if artifact["silhouette"] is not None:
        st.metric("Silhouette Score (label dominan)", f"{artifact['silhouette']:.4f}")
    else:
        st.info("Silhouette Score tidak dapat dihitung untuk pembagian cluster saat ini.")

    counts = result["cluster_dominan"].value_counts().sort_index().reindex(range(1, c + 1), fill_value=0)
    count_df = pd.DataFrame({"Cluster": [f"Cluster {i}" for i in counts.index], "Jumlah data": counts.values})
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(count_df, x="Cluster", y="Jumlah data", title="Distribusi cluster dominan"), use_container_width=True)
    with col2:
        plot_df = result.copy()
        plot_df["Cluster"] = plot_df["cluster_dominan"].astype(str)
        hover_columns = ["Cluster"]
        if "id_mahasiswa" in plot_df.columns:
            hover_columns.append("id_mahasiswa")
        st.plotly_chart(
            px.scatter(plot_df, x=FEATURES[0], y=FEATURES[2], color="Cluster",
                       hover_data=hover_columns,
                       labels={FEATURES[0]: "Jam belajar/minggu", FEATURES[2]: "Nilai tugas"},
                       title="Jam belajar vs nilai tugas"),
            use_container_width=True,
        )

    st.write("**Pusat cluster (skala asli)**")
    st.dataframe(
        pd.DataFrame(centers_original, columns=NAMES, index=[f"Cluster {i + 1}" for i in range(c)]).round(2),
        use_container_width=True,
    )
    st.write("**Tabel data dan derajat keanggotaan**")
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.download_button(
        "Unduh hasil FCM (CSV)",
        result.to_csv(index=False).encode("utf-8"),
        "hasil_fcm_mahasiswa.csv",
        "text/csv",
    )

with tab3:
    st.subheader("Aturan yang dipelajari Decision Tree")
    st.write(
        "Pohon keputusan memprediksi label cluster dominan yang dihasilkan FCM. "
        "Aturan menggunakan skala fitur asli."
    )
    st.metric("Kesesuaian terhadap label FCM pada data uji", f"{artifact['agreement']:.2%}")
    st.caption(
        "Angka ini mengukur kesesuaian Decision Tree dengan label FCM, "
        "bukan akurasi terhadap label kebenaran dari dosen atau ahli."
    )
    st.code(export_text(artifact["tree"], feature_names=NAMES, decimals=2), language="text")

    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(
        artifact["tree"],
        feature_names=NAMES,
        class_names=[f"Cluster {i + 1}" for i in range(c)],
        filled=False,
        rounded=True,
        precision=2,
        ax=ax,
    )
    ax.set_title("Decision Tree untuk menjelaskan label dominan FCM")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    st.subheader("Penjelasan metode")
    st.markdown(
        """
**1. Preprocessing**
- Dataset dapat berasal dari CSV bawaan atau file CSV yang diunggah.
- Sistem memeriksa kolom wajib, mengubah nilai fitur menjadi angka,
  menghapus baris dengan nilai fitur wajib kosong/tidak valid, dan memeriksa rentang nilai.
- `StandardScaler` digunakan untuk FCM agar perbedaan skala fitur tidak mendominasi jarak.
- Decision Tree menggunakan fitur pada skala asli agar aturan mudah dibaca.

**2. Fuzzy C-Means (FCM)**
- `c`: jumlah cluster.
- `m`: fuzzifier dan harus lebih besar dari 1.
- `error`: toleransi berhenti.
- `max_iter`: batas maksimum iterasi.
- FCM memperbarui pusat cluster dan matriks keanggotaan secara berulang.
- Setiap baris mempunyai derajat keanggotaan untuk setiap cluster; totalnya mendekati 1.

**3. Integrasi Decision Tree**
- Label target berasal dari cluster dengan derajat keanggotaan terbesar.
- Decision Tree mempelajari hubungan fitur dengan label dominan FCM.
- Pohon merupakan model penjelas/pengaproksimasi label FCM, bukan label kebenaran kemampuan mahasiswa.

**4. Evaluasi dan batasan**
- FPC membantu melihat ketegasan partisi fuzzy.
- Silhouette dihitung dari label dominan sebagai ukuran tambahan pemisahan cluster.
- Kesesuaian Decision Tree dihitung terhadap label FCM pada data uji, bukan label ground-truth.
- Hasil bergantung pada kualitas dataset dan tidak membuktikan sebab-akibat.
- Data simulasi tidak cukup untuk menyimpulkan pola belajar mahasiswa nyata.
"""
    )
    st.write("**Riwayat fungsi objektif FCM**")
    history_df = pd.DataFrame({
        "Iterasi": np.arange(1, len(artifact["objective_history"]) + 1),
        "Objective function": artifact["objective_history"],
    })
    st.plotly_chart(
        px.line(history_df, x="Iterasi", y="Objective function",
                title="Perubahan fungsi objektif selama iterasi FCM"),
        use_container_width=True,
    )
