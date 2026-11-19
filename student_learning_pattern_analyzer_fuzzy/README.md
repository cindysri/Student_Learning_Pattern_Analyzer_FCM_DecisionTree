# Student Learning Pattern Analyzer — FCM + Decision Tree

Prototype aplikasi GUI Streamlit untuk tugas Teknologi Logika Fuzzy.

## Menjalankan di Windows
1. Ekstrak ZIP.
2. Buka folder proyek di VS Code.
3. Buka Terminal:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Buka URL lokal yang muncul di terminal.

## Metode
- StandardScaler untuk preprocessing.
- Fuzzy C-Means (`scikit-fuzzy`) untuk menghasilkan pusat cluster dan derajat keanggotaan fuzzy.
- Decision Tree untuk mempelajari label dominan FCM dan menghasilkan aturan yang mudah dijelaskan.
- FPC, Silhouette Score berbasis label dominan, serta akurasi kesesuaian Decision Tree dengan label FCM pada data uji sebagai evaluasi pelengkap.

## Fitur GUI
- Input manual empat variabel.
- Tabel dan grafik derajat keanggotaan.
- Dashboard distribusi cluster dan pusat cluster.
- Visualisasi Decision Tree dan aturan tekstual.
- Riwayat fungsi objektif FCM.
- Unduh hasil clustering CSV.
- Pengaturan parameter FCM.

## Dataset
`data/mahasiswa.csv` memuat 180 baris DATA SIMULASI. Ini bukan data nyata. Ganti dengan dataset yang sah dan sesuai sebelum menarik kesimpulan akademik.

## Catatan akademik penting
Decision Tree dilatih menggunakan label dominan hasil FCM. Jadi akurasi Decision Tree pada data uji mengukur kesesuaian terhadap label FCM, bukan kebenaran cluster berdasarkan label ahli. Interpretasi profil cluster juga perlu divalidasi berdasarkan konteks dataset. Jelaskan parameter, rumus, alur iterasi, dan contoh hitung manual di laporan karena penggunaan library tidak menghilangkan kewajiban memahami algoritma.
