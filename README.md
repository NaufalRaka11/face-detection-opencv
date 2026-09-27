# Deteksi Wajah dengan OpenCV Haar Cascade

Program deteksi wajah manusia menggunakan **OpenCV** dan model
**`haarcascade_frontalface_default.xml`**. Setiap wajah yang terdeteksi
diberi *bounding box*, dan jumlah wajah ditampilkan pada frame.

Program mendukung dua mode input:

| Mode | Input | Keterangan |
|------|-------|------------|
| **Gambar** | File `.jpg` / `.png` | Mengambil satu frame dari foto, lalu menampilkan hasilnya |
| **Kamera** | Webcam | Real-time, mendeteksi wajah terus-menerus sampai pengguna keluar |

Kedua mode memakai fungsi deteksi yang sama (`proses_frame`), jadi
perilaku dan hasilnya identik.

---

## Struktur Folder

```text
.
├── main.py                              # program utama + parser argumen
├── face_detector.py                     # logika deteksi wajah (dapat dipakai ulang)
├── gambar_wajah.jpg                     # gambar contoh (2 wajah)
├── models/
│   └── haarcascade_frontalface_default.xml   # model Haar Cascade
├── tests/
│   ├── test_face_detector.py            # pengujian logika deteksi
│   └── test_main.py                     # pengujian mode gambar & kamera
├── requirements.txt
└── pytest.ini
```

---

## Persiapan

Butuh Python 3.8 atau lebih baru.

```bash
pip install -r requirements.txt
```

### Penting: versi OpenCV

Task ini mewajibkan Haar Cascade (`cv2.CascadeClassifier`). **API tersebut
dihapus pada OpenCV 5.x**, jadi versinya harus di bawah 5:

```text
opencv-python>=4.8,<5
```

Kalau OpenCV 5 terpasang, program akan berhenti dengan pesan:

```text
[error] OpenCV 5.0.0 tidak menyediakan CascadeClassifier (Haar Cascade dihapus pada OpenCV 5).
Pasang OpenCV 4.x dengan: pip install 'opencv-python>=4.8,<5'
```

Perbaikannya:

```bash
pip install "opencv-python>=4.8,<5"
```

---

## Cara Pakai

### Mode gambar (statis)

```bash
# memakai gambar_wajah.jpg
python3 main.py

# memakai berkas sendiri
python3 main.py --gambar foto_klasek.jpg
```

### Mode kamera (real-time)

```bash
# kamera pertama
python3 main.py --kamera

# kamera kedua, webcam eksternal
python3 main.py --kamera --camera-index 1
```

Pada mode kamera: tekan **`q`** atau **ESC** untuk keluar, **`s`** untuk
menyimpan *snapshot* frame saat itu.

### Opsi lain

```text
--simpan BERKAS        Simpan hasil akhir ke berkas gambar
--tanpa-jendela         Jangan buka preview (untuk server / tanpa monitor)
--frame-maks N          Batas frame pada mode kamera tanpa jendela (default 300)
--scale-factor FLOAT    Langkah penskalaan detectMultiScale (default 1.2)
--min-neighbors INT     Jumlah kandidat yang harus cocok (default 5)
--min-size WxH          Ukuran wajah terkecil yang dicari (default 30)
--tanpa-label           Jangan tulis nomor urut di atas bounding box
```

Contoh menjalankan tanpa jendela sekaligus menyimpan hasil:

```bash
python3 main.py --gambar gambar_wajah.jpg --tanpa-jendela --simpan hasil.jpg
```

---

## Alur Program

```text
mode gambar                          mode kamera
-----------                          -----------
cv2.imread(berkas)                   kamera.read()  ->  frame
        \                                   /
         \_______________  _______________/
                         \/
              proses_frame(frame, model)
                         |
      +------------------+------------------+
      |                  |                  |
ke_grayscale      equalizeHist      detectMultiScale
      |                  |                  |
 grayscale            kontras        daftar (x, y, w, h)
                         |
                 gambar_bounding_box  ->  hitung_wajah
                         |
              tampilkan (imshow/waitKey) + simpan bila diminta
```

Penjelasan singkat tiap langkah:

1. **Grayscale** — Haar Cascade dilatih memakai citra grayscale, jadi
   `cvtColor(frame, COLOR_BGR2GRAY)` wajib dilakukan sebelum deteksi.
2. **Histogram equalization** — meratakan pencahayaan supaya kontras fitur
   wajah lebih stabil, terutama pada video kamera.
3. **`detectMultiScale`** — mencari wajah pada beberapa skala. Hasilnya
   adalah daftar tupel `(x, y, w, h)`.
4. **Bounding box** — setiap tupel digambar dengan `cv2.rectangle`, diberi
   nomor urut, lalu jumlah wajah ditulis di pojok kiri atas.
5. **Model dimuat sekali** — `muat_model()` dipanggil sebelum loop, bukan
   di dalam loop, supaya file XML tidak dibaca ulang tiap frame.

---

## Menyesuaikan Deteksi

Kalau wajah kecil tidak terdeteksi, turunkan `--min-size`:

```bash
python3 main.py --kamera --min-size 20
```

Kalau banyak false positive (kotak muncul pada bukan wajah), naikkan
`--min-neighbors`:

```bash
python3 main.py --kamera --min-neighbors 8
```

Kalau wajah masih ada yang terlewat, turunkan `--scale-factor` (lebih
akurat, tapi lebih lambat):

```bash
python3 main.py --kamera --scale-factor 1.1
```

---

## Dari Mana Model Datang

`face_detector.cari_path_model()` mencari
`haarcascade_frontalface_default.xml` dengan urutan:

1. folder `models/` di samping `main.py` (model lokal)
2. folder cascades bawaan OpenCV (`cv2.data.haarcascades`)

Model lokal sudah disertakan di folder `models/` supaya program tetap jalan
walaupun OpenCV yang terpasang tidak menyertakan berkas cascade. Model ini
milik OpenCV (Rainer Lienhart) dan lisensinya tercantum di dalam berkas XML.

---

## Menjalankan Pengujian

```bash
pip install -r requirements.txt
python3 -m pytest
```

Pengujian tidak membutuhkan webcam maupun monitor: loop kamera diuji dengan
kamera dan jendela palsu, sementara deteksi wajahnya memakai model Haar
sungguhan pada gambar contoh.

---

## Credit

* Model Haar Cascade: OpenCV, contributed by Rainer Lienhart.
* `gambar_wajah.jpg`: contoh dari repositori
  [`ageitgey/face_recognition`](https://github.com/ageitgey/face_recognition)
  (lisensi MIT).
