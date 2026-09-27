"""
Pengujian `face_detector`.

Prinsip yang dipakai:

  * Fungsi murni diuji langsung tanpa mocking.
  * Objek detector Haar Cascade diganti dengan stub yang mencatat
    argumennya, sehingga `deteksi_wajah` bisa diuji tanpa model sungguhan
    dan tanpa mocking `cv2`.
  * Hanya satu test yang memakai model sungguhan sebagai uji integrasi.
"""

import os

import cv2
import numpy as np
import pytest

import face_detector as fd

AKAR_PROYEK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTOH_GAMBAR = os.path.join(AKAR_PROYEK, "gambar_wajah.jpg")


class StubCascade:
    """
    Pengganti `cv2.CascadeClassifier`.

    Mengembalikan daftar kotak yang sudah ditentukan, sambil mencatat
    kwargs yang diterimanya supaya argumen pemanggilan bisa diperiksa.
    """

    def __init__(self, kotak=None):
        self.kotak = np.array(kotak if kotak is not None else [], dtype=np.int32)
        self.panggilan = []

    def detectMultiScale(self, gambar, **kwargs):  # noqa: N802 - nama API OpenCV
        self.panggilan.append({"gambar": gambar, **kwargs})
        return self.kotak


def frame_dummy(tinggi=120, lebar=160):
    """Frame BGR kosong untuk pengujian."""
    return np.zeros((tinggi, lebar, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# parse_min_size
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "masukan, harapan",
    [
        ("40", (40, 40)),
        ("40x60", (40, 60)),
        ("40X60", (40, 60)),
        ("40 x 60", (40, 60)),
        ("30*30", (30, 30)),
        ((25, 35), (25, 35)),
    ],
)
def test_parse_min_size_menerima_format_valid(masukan, harapan):
    assert fd.parse_min_size(masukan) == harapan


@pytest.mark.parametrize("masukan", ["abc", "", "40x", "0", "-5", "40x60x80"])
def test_parse_min_size_menolak_nilai_tidak_valid(masukan):
    with pytest.raises(ValueError):
        fd.parse_min_size(masukan)


# --------------------------------------------------------------------------
# Pencarian & pemuatan model
# --------------------------------------------------------------------------


def test_cari_path_model_memakai_folder_models_lokal(tmp_path):
    folder_models = tmp_path / "models"
    folder_models.mkdir()
    (folder_models / fd.NAMA_MODEL).write_text("<xml/>")

    assert fd.cari_path_model(str(tmp_path)) == str(folder_models / fd.NAMA_MODEL)


def test_cari_path_model_jatuh_ke_opencv_bawaan(tmp_path):
    """Folder skrip tanpa `models/` -> cari di folder cascades bawaan."""
    folder_cascades = tmp_path / "haarcascades"
    folder_cascades.mkdir()
    (folder_cascades / fd.NAMA_MODEL).write_text("<xml/>")

    hasil = fd.cari_path_model(str(tmp_path), str(folder_cascades))

    assert hasil == str(folder_cascades / fd.NAMA_MODEL)


def test_cari_path_model_mengutamakan_model_lokal(tmp_path):
    """Bila model lokal ada, local path harus menang atas cascades bawaan."""
    folder_models = tmp_path / "models"
    folder_models.mkdir()
    (folder_models / fd.NAMA_MODEL).write_text("<lokal/>")
    folder_cascades = tmp_path / "haarcascades"
    folder_cascades.mkdir()
    (folder_cascades / fd.NAMA_MODEL).write_text("<bawaan/>")

    hasil = fd.cari_path_model(str(tmp_path), str(folder_cascades))

    assert hasil == str(folder_models / fd.NAMA_MODEL)


def test_cari_path_model_mengembalikan_none_bila_semua_tidak_ada(tmp_path):
    assert fd.cari_path_model(str(tmp_path), str(tmp_path / "tidak_ada")) is None


def test_cari_path_model_mengembalikan_none_bila_folder_cascades_kosong(tmp_path):
    assert fd.cari_path_model(str(tmp_path), "") is None


def test_muat_model_gagal_bila_xml_tidak_ada(tmp_path):
    with pytest.raises(FileNotFoundError):
        fd.muat_model(str(tmp_path), str(tmp_path / "tidak_ada"))


def test_muat_model_berhasil_dan_model_tidak_kosong():
    model = fd.muat_model(AKAR_PROYEK)

    assert hasattr(model, "detectMultiScale")
    assert not model.empty()


def test_cek_kompatibilitas_opencv_lolos_pada_opencv_4():
    fd.cek_kompatibilitas_opencv()  # tidak melempar galat pada OpenCV 4.x


# --------------------------------------------------------------------------
# Konversi grayscale
# --------------------------------------------------------------------------


def test_ke_grayscale_menghasilkan_gambar_2_dimensi():
    gray = fd.ke_grayscale(frame_dummy())

    assert gray.ndim == 2
    assert gray.shape == (120, 160)


def test_ke_grayscale_meneruskan_gambar_yang_sudah_grayscale():
    gray_asli = np.zeros((10, 10), dtype=np.uint8)

    assert fd.ke_grayscale(gray_asli) is gray_asli


def test_ke_grayscale_menolak_frame_none():
    with pytest.raises(ValueError):
        fd.ke_grayscale(None)


# --------------------------------------------------------------------------
# Deteksi wajah
# --------------------------------------------------------------------------


def test_deteksi_wajah_mengembalikan_kotak_sebagai_tuple_int():
    stub = StubCascade([[10, 20, 30, 40], [50, 60, 70, 80]])

    wajah = fd.deteksi_wajah(frame_dummy(), stub)

    assert wajah == [(10, 20, 30, 40), (50, 60, 70, 80)]
    semua_int = all(isinstance(nilai, int) for kotak in wajah for nilai in kotak)
    assert semua_int


def test_deteksi_wajah_mengembalikan_list_kosong_bila_tidak_ada():
    stub = StubCascade([])

    assert fd.deteksi_wajah(frame_dummy(), stub) == []


def test_deteksi_wajah_meneruskan_parameter_ke_detectmultiscale():
    stub = StubCascade([[0, 0, 10, 10]])

    fd.deteksi_wajah(
        frame_dummy(),
        stub,
        scale_factor=1.1,
        min_neighbors=7,
        min_size=(40, 50),
    )

    kwargs = stub.panggilan[0]
    assert kwargs["scaleFactor"] == 1.1
    assert kwargs["minNeighbors"] == 7
    assert tuple(kwargs["minSize"]) == (40, 50)


def test_deteksi_wajah_mengirim_gambar_grayscale_ke_model():
    stub = StubCascade([])

    fd.deteksi_wajah(frame_dummy(), stub)

    assert stub.panggilan[0]["gambar"].ndim == 2


# --------------------------------------------------------------------------
# Penordesan bounding box
# --------------------------------------------------------------------------


def test_gambar_bounding_box_menggambar_pada_kotak_yang_diberikan():
    frame = frame_dummy()
    kotak = (20, 30, 40, 50)

    fd.gambar_bounding_box(frame, [kotak], label_urutan=False)

    # Titik di tepi kiri-atas dan tepi kanan-bawah kotak harus tergambar.
    assert frame[30, 20].any()
    assert frame[80, 60].any()
    # Titik di tengah kotak tidak boleh tergambar (kotak hanya garis tepi).
    assert not frame[55, 40].any()


def test_gambar_bounding_box_tidak_mengubah_frame_bila_tidak_ada_wajah():
    frame = frame_dummy()
    sebelum = frame.copy()

    fd.gambar_bounding_box(frame, [])

    assert np.array_equal(frame, sebelum)


def test_gambar_bounding_box_menulis_label_urutan_bila_diminta():
    kotak = (60, 120, 40, 40)  # kotak cukup jauh dari tepi atas

    dengan_label = frame_dummy(200, 200)
    tanpa_label = frame_dummy(200, 200)
    fd.gambar_bounding_box(dengan_label, [kotak], label_urutan=True)
    fd.gambar_bounding_box(tanpa_label, [kotak], label_urutan=False)

    # Label menambah piksel tergambar di area di atas kotak.
    area_label = dengan_label[100:118, 55:140].any(axis=1).sum()
    assert area_label > tanpa_label[100:118, 55:140].any(axis=1).sum()


def test_hitung_wajah_menulis_jumlah_di_frame():
    frame = frame_dummy()

    fd.hitung_wajah(frame, [(0, 0, 10, 10), (20, 20, 10, 10)])

    assert frame[0:40, 0:250].any()  # ada teks di pojok kiri atas


# --------------------------------------------------------------------------
# proses_frame
# --------------------------------------------------------------------------


def test_proses_frame_mengembalikan_frame_dan_jumlah():
    stub = StubCascade([[10, 10, 20, 20], [40, 40, 20, 20]])
    frame = frame_dummy()

    hasil, jumlah = fd.proses_frame(frame, stub, label_urutan=False)

    assert jumlah == 2
    assert hasil is frame
    assert hasil[10, 10].any()  # kotak pertama tergambar


def test_proses_frame_lolos_bila_tidak_ada_wajah():
    stub = StubCascade([])
    frame = frame_dummy()
    sebelum = frame.copy()

    hasil, jumlah = fd.proses_frame(frame, stub)

    assert jumlah == 0
    # Frame hanya berubah kalau ada anotasi (teks jumlah wajah).
    assert hasil.shape == sebelum.shape


# --------------------------------------------------------------------------
# Uji integrasi dengan model sungguhan
# --------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.isfile(CONTOH_GAMBAR), reason="gambar contoh tidak ada")
def test_integrasi_mendeteksi_wajah_pada_gambar_asli():
    """Deteksi ujung-ke-ujung memakai model Haar asli dan gambar asli."""
    model = fd.muat_model(AKAR_PROYEK)
    gambar = cv2.imread(CONTOH_GAMBAR)

    wajah = fd.deteksi_wajah(gambar, model)

    assert len(wajah) >= 1
    tinggi, lebar = gambar.shape[:2]
    for x, y, w, h in wajah:
        assert x >= 0 and y >= 0
        assert x + w <= lebar
        assert y + h <= tinggi


@pytest.mark.skipif(not os.path.isfile(CONTOH_GAMBAR), reason="gambar contoh tidak ada")
def test_integrasi_mendeteksi_lebih_dari_satu_wajah():
    """Gambar contoh berisi dua wajah, jadi keduanya harus ditemukan."""
    model = fd.muat_model(AKAR_PROYEK)
    gambar = cv2.imread(CONTOH_GAMBAR)

    assert len(fd.deteksi_wajah(gambar, model)) >= 2
