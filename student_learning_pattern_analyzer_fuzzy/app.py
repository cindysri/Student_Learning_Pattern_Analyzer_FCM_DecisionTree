import os
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

st.set_page_config(
    page_title="Student Learning Pattern Analyzer",
    page_icon="📚",
    layout="wide",
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    if len(df) < 6:
        raise ValueError("Dataset perlu minimal 6 baris data valid.")

    return df


@st.cache_resource
def build_pipeline(data_signature, n_clusters, fuzzifier, max_iter, error):
    df = load_data()
    X = df[FEATURES].to_numpy(dtype=float)

    # Normalisasi hanya digunakan untuk proses Fuzzy C-Means.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # scikit-fuzzy mengharapkan bentuk data (fitur, sampel).
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

    # Label dominan dibuat dari derajat keanggotaan FCM terbesar.
    dominant = np.argmax(U, axis=0)

    # Decision Tree menggunakan fitur pada skala asli agar aturan
    # mudah dibaca, misalnya jam belajar/minggu dan nilai 0-100.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        dominant,
        test_size=0.25,
        random_state=42,
        stratify=dominant,
    )

    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=3,
        random_state=42,
    )
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    agreement = accuracy_score(y_test, y_pred)

    try:
        silhouette = (
            silhouette_score(X_scaled, dominant)
            if len(np.unique(dominant)) > 1
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
    """Hitung derajat keanggotaan data baru menggunakan pusat FCM."""
    distances = np.linalg.norm(centers - x_scaled.reshape(1, -1), axis=1)

    # Tangani kondisi data tepat sama dengan salah satu pusat cluster.
    if np.any(distances < 1e-12):
        membership = np.zeros(len(centers))
        membership[np.argmin(distances)] = 1.0
        return membership

    power = 2.0 / (m - 1.0)
    inverse_distances = distances ** (-power)

    return inverse_distances / inverse_distances.sum()


def describe_centers(centers_original):
    """Beri deskripsi relatif berdasarkan rata-rata fitur centroid."""
    result = []

    for center in centers_original:
        normalized_profile_score = np.mean(
            [
                center[0] / 20.0,
                center[1] / 100.0,
                center[2] / 100.0,
                center[3] / 100.0,
            ]
        )

        if normalized_profile_score >= 0.72:
            name = "Profil relatif tinggi"
        elif normalized_profile_score >= 0.48:
            name = "Profil relatif sedang"
        else:
            name = "Profil perlu perhatian"

        result.append(name)

    return result


try:
    df = load_data()
except Exception as exc:
    st.error(f"Gagal membaca dataset: {exc}")
    st.stop()


with st.sidebar:
    st.header("Parameter FCM")
    c = st.selectbox("Jumlah cluster (c)", [2, 3, 4, 5], index=1)
    m = st.slider(
        "Fuzzifier (m)",
        min_value=1.5,
        max_value=3.0,
        value=2.0,
        step=0.1,
    )
    max_iter = st.slider(
        "Maksimum iterasi",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
    )
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


model_key = (c, m, max_iter, error)

if (
    run
    or "artifact" not in st.session_state
    or st.session_state.get("model_key") != model_key
):
    with st.spinner("Menghitung FCM dan melatih Decision Tree..."):
        st.session_state.artifact = build_pipeline(
            len(df), c, m, max_iter, error
        )
        st.session_state.model_key = model_key

artifact = st.session_state.artifact
centers_original = artifact["scaler"].inverse_transform(artifact["centers"])
profiles = describe_centers(centers_original)

st.title("Student Learning Pattern Analyzer")
st.caption(
    "Analisis pola belajar mahasiswa dengan Fuzzy C-Means dan Decision Tree"
)
st.warning(
    "Dataset bawaan adalah DATA SIMULASI untuk prototipe, "
    "bukan data mahasiswa nyata."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Analisis mahasiswa",
        "Dashboard FCM",
        "Decision Tree",
        "Metode & evaluasi",
    ]
)


with tab1:
    st.subheader("Input aktivitas belajar")

    with st.form("student_input"):
        col1, col2 = st.columns(2)

        with col1:
            study = st.number_input(
                "Jam belajar per minggu", 0.0, 80.0, 8.0, 0.5
            )
            attend = st.number_input(
                "Kehadiran (%)", 0.0, 100.0, 80.0, 1.0
            )

        with col2:
            task = st.number_input(
                "Rata-rata nilai tugas", 0.0, 100.0, 78.0, 1.0
            )
            quiz = st.number_input(
                "Rata-rata nilai kuis", 0.0, 100.0, 75.0, 1.0
            )

        submit = st.form_submit_button(
            "Analisis pola belajar",
            type="primary",
            use_container_width=True,
        )

    if submit:
        row = np.array([[study, attend, task, quiz]], dtype=float)

        # FCM membutuhkan input yang dinormalisasi.
        x_scaled = artifact["scaler"].transform(row)[0]
        memberships = membership_for_new(
            x_scaled,
            artifact["centers"],
            artifact["m"],
        )
        dominant_cluster = int(np.argmax(memberships))

        # Decision Tree dilatih menggunakan skala asli,
        # sehingga prediksi input baru juga harus menggunakan row asli.
        tree_label = int(artifact["tree"].predict(row)[0])

        left, right = st.columns(2)

        with left:
            st.metric(
                "Cluster dominan FCM",
                f"Cluster {dominant_cluster + 1}",
            )
            st.write(
                f"**Interpretasi centroid:** {profiles[dominant_cluster]}"
            )

        with right:
            st.metric("Prediksi Decision Tree", f"Cluster {tree_label + 1}")
            st.write(
                "Decision Tree memprediksi label dominan FCM. "
                "Hasilnya dapat berbeda dari hasil FCM langsung."
            )

        st.write("**Derajat keanggotaan FCM**")

        membership_df = pd.DataFrame(
            {
                "Cluster": [f"Cluster {i + 1}" for i in range(c)],
                "Derajat keanggotaan": memberships,
                "Persentase": [
                    f"{value * 100:.2f}%" for value in memberships
                ],
                "Profil centroid": profiles,
            }
        )

        st.dataframe(
            membership_df,
            hide_index=True,
            use_container_width=True,
        )

        st.plotly_chart(
            px.bar(
                membership_df,
                x="Cluster",
                y="Derajat keanggotaan",
                title="Derajat keanggotaan terhadap setiap cluster",
                hover_data=["Persentase", "Profil centroid"],
            ),
            use_container_width=True,
        )

        st.caption(
            "Derajat keanggotaan bukan probabilitas kelulusan atau diagnosis "
            "kemampuan. Interpretasi profil bersifat deskriptif berdasarkan "
            "centroid."
        )


with tab2:
    st.subheader("Ringkasan hasil Fuzzy C-Means")

    X_scaled = artifact["scaler"].transform(
        df[FEATURES].to_numpy(dtype=float)
    )
    result = df.copy()

    for j in range(c):
        result[f"keanggotaan_cluster_{j + 1}"] = artifact["U"][j]

    result["cluster_dominan"] = artifact["dominant"] + 1

    k1, k2, k3 = st.columns(3)
    k1.metric("Jumlah data valid", len(df))
    k2.metric("Iterasi FCM", artifact["iterations"])
    k3.metric(
        "Fuzzy Partition Coefficient",
        f"{artifact['fpc']:.4f}",
    )

    if artifact["silhouette"] is not None:
        st.metric(
            "Silhouette Score (label dominan)",
            f"{artifact['silhouette']:.4f}",
        )

    counts = (
        result["cluster_dominan"]
        .value_counts()
        .sort_index()
        .reindex(range(1, c + 1), fill_value=0)
    )
    count_df = pd.DataFrame(
        {
            "Cluster": [f"Cluster {i}" for i in counts.index],
            "Jumlah mahasiswa": counts.values,
        }
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.bar(
                count_df,
                x="Cluster",
                y="Jumlah mahasiswa",
                title="Distribusi cluster dominan",
            ),
            use_container_width=True,
        )

    with col2:
        plot_df = result.copy()
        plot_df["Cluster"] = plot_df["cluster_dominan"].astype(str)

        hover_columns = ["Cluster"]
        if "id_mahasiswa" in plot_df.columns:
            hover_columns.append("id_mahasiswa")

        st.plotly_chart(
            px.scatter(
                plot_df,
                x=FEATURES[0],
                y=FEATURES[2],
                color="Cluster",
                hover_data=hover_columns,
                labels={
                    FEATURES[0]: "Jam belajar/minggu",
                    FEATURES[2]: "Nilai tugas",
                },
                title="Jam belajar vs nilai tugas",
            ),
            use_container_width=True,
        )

    st.write("**Pusat cluster (dikembalikan ke skala asli)**")
    st.dataframe(
        pd.DataFrame(
            centers_original,
            columns=NAMES,
            index=[f"Cluster {i + 1}" for i in range(c)],
        ).round(2),
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
        "Pohon keputusan dilatih untuk memprediksi label cluster dominan "
        "yang dihasilkan FCM. Batas aturan menggunakan skala fitur asli."
    )

    st.metric(
        "Akurasi kesesuaian terhadap label FCM pada data uji",
        f"{artifact['agreement']:.2%}",
    )
    st.caption(
        "Angka ini mengukur kesesuaian Decision Tree dengan label FCM, "
        "bukan akurasi terhadap label kebenaran dari dosen atau ahli."
    )

    rules = export_text(
        artifact["tree"],
        feature_names=NAMES,
        decimals=2,
    )
    st.code(rules, language="text")

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
- Memeriksa kolom dan menghapus baris yang tidak memiliki nilai valid
  pada empat fitur.
- Menormalisasi fitur dengan `StandardScaler` untuk FCM agar perbedaan
  skala jam dan nilai tidak mendominasi perhitungan jarak.
- Decision Tree menggunakan fitur pada skala asli supaya aturan lebih
  mudah dibaca.

**2. Fuzzy C-Means**
- `c`: jumlah cluster.
- `m`: fuzzifier, harus lebih besar dari 1.
- `error`: toleransi perubahan matriks keanggotaan.
- `max_iter`: batas maksimum iterasi.
- FCM menghitung pusat cluster dengan pembobotan derajat keanggotaan
  dan memperbarui matriks keanggotaan sampai konvergen atau batas
  iterasi tercapai.
- Setiap data memiliki derajat keanggotaan pada setiap cluster;
  jumlah derajat keanggotaannya mendekati 1.

**3. Integrasi Decision Tree**
- Label target dibuat dari cluster dengan derajat keanggotaan terbesar.
- Decision Tree mempelajari hubungan fitur dengan label dominan FCM.
- Decision Tree menggunakan fitur skala asli agar aturan keputusan
  lebih mudah dijelaskan.
- Decision Tree merupakan model penjelas/pengaproksimasi label FCM,
  bukan label kebenaran kemampuan mahasiswa.

**4. Evaluasi dan keterbatasan**
- FPC membantu melihat ketegasan partisi fuzzy.
- Silhouette dihitung dari label dominan sebagai ukuran tambahan
  pemisahan cluster.
- Nilai kesesuaian Decision Tree dihitung terhadap label FCM pada
  data uji, bukan label ground-truth.
- Hasil bergantung pada kualitas dan keterwakilan dataset.
  Data simulasi tidak cukup untuk menyimpulkan pola belajar
  mahasiswa nyata.
"""
    )

    st.write("**Riwayat fungsi objektif FCM**")

    history_df = pd.DataFrame(
        {
            "Iterasi": np.arange(
                1, len(artifact["objective_history"]) + 1
            ),
            "Objective function": artifact["objective_history"],
        }
    )

    st.plotly_chart(
        px.line(
            history_df,
            x="Iterasi",
            y="Objective function",
            title="Perubahan fungsi objektif selama iterasi FCM",
        ),
        use_container_width=True,
    )
