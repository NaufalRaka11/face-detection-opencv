#!/usr/bin/env python3
"""
Program deteksi wajah manusia dengan OpenCV Haar Cascade.

Dua mode input yang tersedia (pilih salah satu saat dijalankan):

  1. Mode gambar (statis)
        python3 main.py
        python3 main.py --gambar foto_wajah.jpg

  2. Mode kamera (real-time)
        python3 main.py --kamera
        python3 main.py --kamera --camera-index 1

Pada kedua mode, setiap wajah yang terdeteksi diberi bounding box, dan
jumlah wajah ditampilkan di pojok kiri atas frame.
"""

import argparse
import sys
import time

import cv2

import face_detector as fd

# Nama berkas gambar yang dipakai kalau mode gambar dijalankan tanpa argumen.
GAMBAR_DEFAULT = "gambar_wajah.jpg"

# Izinkan ekstensi gambar umum, bukan hanya .jpg dan .png.
EKSTENSI_GAMBAR = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def tampilkan_window(nama_jendela, frame, delay_ms):
    """
    Tampilkan frame di sebuah jendela lalu tunggu tombol.

    Fungsi ini disatukan dalam satu pemanggilan agar mudah diganti dengan
    stub saat pengujian, sehingga test tidak butuh monitor sungguhan.

    Mengembalikan kode tombol yang ditekan (0 kalau tidak ada).
    """
    cv2.imshow(nama_jendela, frame)
    return cv2.waitKey(delay_ms) & 0xFF


def tutup_jendela():
    """Tutup semua jendela OpenCV."""
    cv2.destroyAllWindows()


def buat_parser():
    """Susun argumen baris perintah."""
    parser = argparse.ArgumentParser(
        description="Deteksi wajah manusia dengan OpenCV Haar Cascade.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    grup_input = parser.add_mutually_exclusive_group()
    grup_input.add_argument(
        "--kamera",
        action="store_true",
        help="Mode kamera real-time (webcam).",
    )
    grup_input.add_argument(
        "--gambar",
        nargs="?",
        const=GAMBAR_DEFAULT,
        metavar="BERKAS",
        help=f"Mode gambar statis. Tanpa nilai akan memakai '{GAMBAR_DEFAULT}'.",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Nomor kamera yang dipakai pada mode kamera (0 = kamera default).",
    )
    parser.add_argument(
        "--simpan",
        metavar="BERKAS",
        help="Simpan hasil akhir ke berkas gambar.",
    )
    parser.add_argument(
        "--tanpa-jendela",
        action="store_true",
        help="Jangan buka jendela preview (berguna di server atau tanpa monitor).",
    )
    parser.add_argument(
        "--frame-maks",
        type=int,
        default=300,
        metavar="N",
        help=(
            "Batas frame pada mode kamera tanpa jendela. Diabaikan kalau "
            "jendela preview dibuka."
        ),
    )

    grup_param = parser.add_argument_group("parameter deteksi")
    grup_param.add_argument(
        "--scale-factor",
        type=float,
        default=fd.SCALE_FACTOR_DEFAULT,
        help="Langkah penskalaan detectMultiScale. Makin kecil = makin akurat.",
    )
    grup_param.add_argument(
        "--min-neighbors",
        type=int,
        default=fd.MIN_NEIGHBORS_DEFAULT,
        help="Jumlah area kandidat yang harus cocok agar dianggap wajah.",
    )
    grup_param.add_argument(
        "--min-size",
        type=fd.parse_min_size,
        default=fd.MIN_SIZE_DEFAULT,
        metavar="WxH",
        help="Ukuran wajah terkecil yang dicari, contoh: 30 atau 40x60.",
    )
    grup_param.add_argument(
        "--tanpa-label",
        action="store_true",
        help="Jangan tulis nomor urut di atas setiap bounding box.",
    )

    return parser


def jalankan_mode_gambar(
    path_gambar, args, model, tampil=tampilkan_window, tutup=tutup_jendela
):
    """Mode gambar statis: baca satu berkas, deteksi, lalu tampilkan hasilnya."""
    if not path_gambar.lower().endswith(EKSTENSI_GAMBAR):
        print(f"[error] Format berkas tidak didukung: {path_gambar}")
        print("        Gunakan berkas .jpg atau .png.")
        return 1

    gambar = cv2.imread(path_gambar)
    if gambar is None:
        print(f"[error] Gambar tidak dapat dibaca: {path_gambar}")
        return 1

    hasil, jumlah = fd.proses_frame(
        gambar,
        model,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_size=args.min_size,
        label_urutan=not args.tanpa_label,
    )

    print(f"Jumlah wajah yang terdeteksi: {jumlah}")

    if args.simpan:
        cv2.imwrite(args.simpan, hasil)
        print(f"[info] Hasil disimpan ke: {args.simpan}")

    if not args.tanpa_jendela:
        print("[info] Tekan tombol apa saja pada jendela hasil untuk menutupnya.")
        tampil("Hasil Deteksi Wajah", hasil, 0)
        tutup()
    elif not args.simpan:
        print("[info] Tanpa jendela: pakai --simpan bila ingin melihat hasilnya.")

    return 0


def jalankan_mode_kamera(
    args, model, capture_factory=None, tampil=tampilkan_window, tutup=tutup_jendela
):
    """
    Mode kamera real-time: baca frame, deteksi wajah, lalu gambar kotaknya.

    `capture_factory` dan `tampil` adalah titik injeksi: pada kondisi
    normal keduanya memakai fungsi OpenCV sungguhan, dan pada test keduanya
    diganti objek palsu supaya loop bisa dijalankan tanpa webcam.
    """
    if capture_factory is None:
        capture_factory = cv2.VideoCapture

    kamera = capture_factory(args.camera_index)
    if not kamera.isOpened():
        print(f"[error] Kamera index {args.camera_index} tidak dapat dibuka.")
        print("        Cek kamera terpasang, atau coba --camera-index 1.")
        return 1

    print("[info] Tekan 'q' atau ESC untuk keluar, 's' untuk menyimpan frame.")

    jumlah_wajah_terakhir = 0
    jumlah_frame = 0
    frame_terakhir = None
    waktu_sebelumnya = time.perf_counter()
    fps = 0.0

    try:
        while True:
            berhasil, frame = kamera.read()
            if not berhasil:
                print("[warn] Frame tidak terbaca, video mungkin sudah berhenti.")
                break

            # FPS dihitung dari selisih waktu antar frame.
            sekarang = time.perf_counter()
            fps = 1.0 / max(sekarang - waktu_sebelumnya, 1e-6)
            waktu_sebelumnya = sekarang

            frame, jumlah_wajah_terakhir = fd.proses_frame(
                frame,
                model,
                scale_factor=args.scale_factor,
                min_neighbors=args.min_neighbors,
                min_size=args.min_size,
                label_urutan=not args.tanpa_label,
            )
            frame_terakhir = frame

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                fd.WARNA_TEKS,
                2,
            )

            if args.tanpa_jendela:
                # Tanpa jendela tidak ada tombol yang bisa ditekan, jadi beri
                # batas jumlah frame supaya mode ini tetap bisa dijalankan
                # di server atau pada mesin tanpa monitor.
                jumlah_frame += 1
                if jumlah_frame >= args.frame_maks:
                    break
                continue

            tombol = tampil("Deteksi Wajah - Kamera", frame, 1)

            if tombol in (ord("q"), 27):  # 27 = tombol ESC
                break
            if tombol == ord("s"):
                nama = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(nama, frame)
                print(f"[info] Snapshot disimpan: {nama}")
    except KeyboardInterrupt:
        print("\n[info] Dihentikan oleh pengguna (Ctrl+C).")
    finally:
        # Selalu lepaskan kamera, baik keluar normal, error, atau Ctrl+C.
        kamera.release()
        if not args.tanpa_jendela:
            tutup()

    print(f"Jumlah wajah pada frame terakhir: {jumlah_wajah_terakhir}")

    if args.simpan and frame_terakhir is not None:
        cv2.imwrite(args.simpan, frame_terakhir)
        print(f"[info] Frame terakhir disimpan ke: {args.simpan}")
    elif not args.simpan:
        print("[info] Simpan frame terakhir dengan opsi --simpan berkas.jpg")

    return 0


def main(argv=None, **seam):
    """
    Titik masuk program.

    Kata kunci `seam` (mis. `tampil=...`) diteruskan ke mode yang dipilih
    untuk keperluan pengujian.
    """
    args = buat_parser().parse_args(argv)

    try:
        model = fd.muat_model()
        min_size = fd.parse_min_size(args.min_size)
    except (FileNotFoundError, RuntimeError, ValueError) as galat:
        print(f"[error] {galat}")
        return 1

    # argparse sudah memvalidasi lewat `type=`, tapi parser yang dibangun
    # manual (di test) bisa memasukkan nilai mentah, jadi normalisasi di sini.
    args.min_size = min_size

    if args.kamera:
        return jalankan_mode_kamera(args, model, **seam)

    return jalankan_mode_gambar(args.gambar or GAMBAR_DEFAULT, args, model, **seam)


if __name__ == "__main__":
    sys.exit(main())
