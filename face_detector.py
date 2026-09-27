"""
Modul inti deteksi wajah manusia dengan OpenCV Haar Cascade.

Pisahkan bagian "logika deteksi" dari `main.py` supaya alur yang sama
dapat dipakai ulang oleh dua mode input:

    kamera real-time  ->  main.py --kamera
    gambar statis     ->  main.py --gambar foto.jpg

Semua fungsi di modul ini menerima objek detector lewat argumen
`model`, bukan membuatnya sendiri di dalam modul. Dengan begitu modul
ini bisa diuji tanpa kamera dan tanpa display.
"""

import os

import cv2

NAMA_MODEL = "haarcascade_frontalface_default.xml"

# Parameter default untuk CascadeClassifier.detectMultiScale.
#
#   scale_factor  : besar langkah penskalaan tiap iterasi. Makin kecil
#                   -> makin akurat, tapi makin lambat.
#   min_neighbors : berapa banyak area kandidat yang harus saling cocok
#                   sebelum sebuah kotak dianggap wajah. Makin besar
#                   -> makin sedikit false positive.
#   min_size      : ukuran wajah terkecil (w, h) yang dicari, dalam piksel.
SCALE_FACTOR_DEFAULT = 1.2
MIN_NEIGHBORS_DEFAULT = 5
MIN_SIZE_DEFAULT = (30, 30)

# Warna dalam format BGR (OpenCV memakai BGR, bukan RGB).
WARNA_KOTAK = (0, 255, 0)
WARNA_TEKS = (0, 255, 0)


def parse_min_size(nilai):
    """
    Ubah teks Ukuran minimum jadi tuple (width, height).

    Menerima `"40"` (diartikan 40x40) atau `"40x60"`. Nilai yang tidak
    valid akan dilempar sebagai `ValueError` supaya pesan errornya jelas
    daripada dibiarkan gagal di dalam OpenCV.
    """
    if isinstance(nilai, (tuple, list)):
        lebar, tinggi = nilai
        return int(lebar), int(tinggi)

    teks = str(nilai).lower().replace(" ", "")
    pemisah = "x" if "x" in teks else ("*" if "*" in teks else None)

    try:
        if pemisah:
            bagian = teks.split(pemisah)
            if len(bagian) != 2:
                raise ValueError
            lebar, tinggi = int(bagian[0]), int(bagian[1])
        else:
            lebar = tinggi = int(teks)
    except ValueError:
        raise ValueError(
            f"Min size tidak valid: {nilai!r}. Gunakan format '40' atau '40x60'."
        ) from None

    if lebar <= 0 or tinggi <= 0:
        raise ValueError(f"Min size harus lebih besar dari 0: {nilai!r}.")

    return lebar, tinggi


def cari_path_model(direktori_skrip=None, folder_cascades=None):
    """
    Cari file `haarcascade_frontalface_default.xml` di disk.

    Urutan pengecekan:
      1. folder `models/` di samping file ini   -> model lokal
      2. folder cascades bawaan OpenCV          -> model internal

    `direktori_skrip` dan `folder_cascades` bisa dioverride agar pencarian
    dapat diarahkan ke lokasi tertentu, termasuk ke lokasi yang memang
    tidak berisi model (dipakai pengujian).

    Mengembalikan path file XML, atau `None` bila keduanya tidak ada.
    """
    if direktori_skrip is None:
        direktori_skrip = os.path.dirname(os.path.abspath(__file__))

    kandidat_lokal = os.path.join(direktori_skrip, "models", NAMA_MODEL)
    if os.path.isfile(kandidat_lokal):
        return kandidat_lokal

    if folder_cascades is None:
        folder_cascades = getattr(cv2.data, "haarcascades", "") or ""

    kandidat_bawaan = os.path.join(folder_cascades, NAMA_MODEL)
    if folder_cascades and os.path.isfile(kandidat_bawaan):
        return kandidat_bawaan

    return None


def cek_kompatibilitas_opencv():
    """
    Pastikan OpenCV yang terpasang masih menyediakan Haar Cascade.

    OpenCV 5.x menghapus `CascadeClassifier` (API Haar Cascade digantikan
    FaceDetectorYN berbasis DNN). Task ini mewajibkan Haar Cascade, jadi
    versinya harus dibatasi ke 4.x.
    """
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            f"OpenCV {cv2.__version__} tidak menyediakan CascadeClassifier "
            "(Haar Cascade dihapus pada OpenCV 5). Pasang OpenCV 4.x dengan: "
            "pip install 'opencv-python>=4.8,<5'"
        )


def muat_model(direktori_skrip=None, folder_cascades=None):
    """
    Muat model Haar Cascade dan kembalikan `cv2.CascadeClassifier`.

    Dipanggil sekali di awal program, bukan di dalam loop video, supaya
    model tidak dibaca ulang dari disk pada setiap frame.
    """
    cek_kompatibilitas_opencv()

    path_model = cari_path_model(direktori_skrip, folder_cascades)
    if path_model is None:
        raise FileNotFoundError(
            f"Model {NAMA_MODEL} tidak ditemukan. Letakkan file XML di folder "
            "'models/' atau pasang ulang OpenCV: pip install opencv-python"
        )

    model = cv2.CascadeClassifier(path_model)
    if model.empty():
        raise ValueError(f"Model {NAMA_MODEL} gagal dimuat dari: {path_model}")

    print(f"[info] Model dimuat: {path_model}")
    return model


def ke_grayscale(gambar):
    """
    Ubah gambar BGR menjadi grayscale.

    Haar Cascade dilatih pada citra grayscale, jadi konversi ini wajib
    dilakukan sebelum memanggil `detectMultiScale`. Input yang sudah
    grayscale dikembalikan apa adanya.
    """
    if gambar is None:
        raise ValueError("Gambar/frame kosong (None).")
    if gambar.ndim == 2:
        return gambar
    return cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)


def deteksi_wajah(
    frame,
    model,
    scale_factor=SCALE_FACTOR_DEFAULT,
    min_neighbors=MIN_NEIGHBORS_DEFAULT,
    min_size=MIN_SIZE_DEFAULT,
):
    """
    Deteksi wajah pada satu frame BGR.

    Mengembalikan list of (x, y, w, h), satu elemen per wajah yang
    ditemukan. List kosong berarti tidak ada wajah terdeteksi.
    """
    gray = ke_grayscale(frame)

    # Histogram equalization meratakan pencahayaan sehingga kontras fitur
    # wajah lebih stabil, terutama untuk video kamera.
    gray = cv2.equalizeHist(gray)

    wajah = model.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=tuple(int(nilai) for nilai in min_size),
    )

    # OpenCV bisa mengembalikan array kosong (bukan tuple) saat tidak ada
    # deteksi, jadi normalisasi bentuk hasilnya di sini.
    return [tuple(int(v) for v in kotak) for kotak in wajah]


def gambar_bounding_box(frame, wajah, label_urutan=True, warna=WARNA_KOTAK):
    """
    Gambar kotak penanda pada setiap wajah yang terdeteksi.

    `wajah` adalah list (x, y, w, h) dari `deteksi_wajah`. Frame diubah
    secara in-place dan dikembalikan lagi agar bisa di-chain.
    """
    for urutan, (x, y, w, h) in enumerate(wajah, start=1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), warna, 2)

        if label_urutan:
            cv2.putText(
                frame,
                f"Wajah {urutan}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                WARNA_TEKS,
                2,
            )

    return frame


def hitung_wajah(frame, wajah, teks="Wajah terdeteksi"):
    """Tulis jumlah wajah ke pojok kiri atas frame."""
    cv2.putText(
        frame,
        f"{teks}: {len(wajah)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        WARNA_TEKS,
        2,
    )
    return frame


def proses_frame(
    frame,
    model,
    scale_factor=SCALE_FACTOR_DEFAULT,
    min_neighbors=MIN_NEIGHBORS_DEFAULT,
    min_size=MIN_SIZE_DEFAULT,
    label_urutan=True,
):
    """
    Satu frame masuk, satu frame berannotasi keluar.

    Inilah fungsi yang dipakai oleh kedua mode input, sehingga perilakunya
    identik: video kamera dan foto statis hanya berbeda pada cara
    memperoleh frame-nya.

    Mengembalikan (frame_ber_kotak, jumlah_wajah).
    """
    wajah = deteksi_wajah(
        frame,
        model,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
    )
    gambar_bounding_box(frame, wajah, label_urutan=label_urutan)
    hitung_wajah(frame, wajah)
    return frame, len(wajah)
