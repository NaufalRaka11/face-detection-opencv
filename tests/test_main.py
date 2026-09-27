"""
Pengujian `main.py`.

Loop kamera diuji dengan kamera palsu (`KameraPalsu`) dan jendela palsu,
sehingga test berjalan tanpa webcam dan tanpa monitor. Deteksi wajah
sendirinya memakai model Haar sungguhan pada frame contoh, jadi alur
deteksi ikut teruji tanpa perlu membuat wajah secara artificial.
"""

import os

import cv2
import pytest

import main as app
from main import GAMBAR_DEFAULT

AKAR_PROYEK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTOH_GAMBAR = os.path.join(AKAR_PROYEK, GAMBAR_DEFAULT)


class KameraPalsu:
    """Pengganti `cv2.VideoCapture` yang mengembalikan frame yang sudah disiapkan."""

    def __init__(self, frame, jumlah_frame=5, dibuka=True):
        self.frame = frame
        self.sisa = jumlah_frame
        self.dibuka = dibuka
        self.index = None
        self.dilepas = False
        self.jumlah_baca = 0

    def __call__(self, index):
        self.index = index
        return self

    def isOpened(self):  # noqa: N802 - nama API OpenCV
        return self.dibuka

    def read(self):
        self.jumlah_baca += 1
        if self.sisa <= 0:
            return False, None
        self.sisa -= 1
        # Salin supaya anotasi pada frame tidak menumpuk antar frame.
        return True, self.frame.copy()

    def release(self):
        self.dilepas = True


class JendelaPalsu:
    """Pengganti `tampilkan_window` yang mencatat pemanggilannya."""

    def __init__(self, tombol=0):
        self.tombol = tombol
        self.panggilan = []

    def __call__(self, nama, frame, delay_ms):
        self.panggilan.append((nama, frame, delay_ms))
        return self.tombol


@pytest.fixture
def gambar_contoh():
    if not os.path.isfile(CONTOH_GAMBAR):
        pytest.skip("gambar contoh tidak ada")
    return cv2.imread(CONTOH_GAMBAR)


# --------------------------------------------------------------------------
# Parsing argumen
# --------------------------------------------------------------------------


def test_parser_default_memakai_mode_gambar():
    args = app.buat_parser().parse_args([])

    assert args.kamera is False
    assert args.gambar is None
    assert args.camera_index == 0
    assert args.tanpa_jendela is False


def test_parser_gambar_tanpa_nilai_memakai_nama_default():
    args = app.buat_parser().parse_args(["--gambar"])

    assert args.gambar == GAMBAR_DEFAULT


def test_parser_menolak_kamera_dan_gambar_bersamaan():
    with pytest.raises(SystemExit):
        app.buat_parser().parse_args(["--kamera", "--gambar", "a.jpg"])


def test_parser_mengubah_min_size_menjadi_tuple():
    args = app.buat_parser().parse_args(["--min-size", "40x60"])

    assert args.min_size == (40, 60)


def test_parser_menolak_min_size_tidak_valid():
    with pytest.raises(SystemExit):
        app.buat_parser().parse_args(["--min-size", "besar"])


# --------------------------------------------------------------------------
# Mode gambar
# --------------------------------------------------------------------------


def test_mode_gambar_menyimpan_hasil_dan_mencetak_jumlah(tmp_path, gambar_contoh, capsys):
    sumber = tmp_path / "masukan.jpg"
    cv2.imwrite(str(sumber), gambar_contoh)
    tujuan = tmp_path / "hasil.jpg"

    kode = app.main(
        ["--gambar", str(sumber), "--tanpa-jendela", "--simpan", str(tujuan)]
    )

    keluaran = capsys.readouterr().out
    assert kode == 0
    assert "Jumlah wajah yang terdeteksi:" in keluaran
    assert tujuan.is_file()

    hasil = cv2.imread(str(tujuan))
    assert hasil is not None
    assert hasil.shape == gambar_contoh.shape


def test_mode_gambar_membuka_jendela_kecuali_diminta_lain(tmp_path, gambar_contoh):
    sumber = tmp_path / "masukan.jpg"
    cv2.imwrite(str(sumber), gambar_contoh)
    jendela = JendelaPalsu()
    ditutup = []

    kode = app.main(
        ["--gambar", str(sumber)],
        tampil=jendela,
        tutup=lambda: ditutup.append(True),
    )

    assert kode == 0
    assert len(jendela.panggilan) == 1
    assert jendela.panggilan[0][0] == "Hasil Deteksi Wajah"
    assert ditutup == [True]


def test_mode_gambar_tidak_membuka_jendela_saat_tanpa_jendela(tmp_path, gambar_contoh):
    sumber = tmp_path / "masukan.jpg"
    cv2.imwrite(str(sumber), gambar_contoh)
    jendela = JendelaPalsu()

    kode = app.main(
        ["--gambar", str(sumber), "--tanpa-jendela"],
        tampil=jendela,
        tutup=lambda: None,
    )

    assert kode == 0
    assert jendela.panggilan == []


def test_mode_gambar_menolak_ekstensi_tidak_didukung(tmp_path, capsys):
    berkas = tmp_path / "catatan.txt"
    berkas.write_text("bukan gambar")

    kode = app.main(["--gambar", str(berkas), "--tanpa-jendela"])

    assert kode == 1
    assert "Format berkas tidak didukung" in capsys.readouterr().out


def test_mode_gambar_menolak_berkas_yang_tidak_ada(tmp_path, capsys):
    kode = app.main(["--gambar", str(tmp_path / "hilang.jpg"), "--tanpa-jendela"])

    assert kode == 1
    assert "tidak dapat dibaca" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Mode kamera
# --------------------------------------------------------------------------


def test_mode_kamera_memproses_frame_sampai_kamera_kosong(gambar_contoh):
    kamera = KameraPalsu(gambar_contoh, jumlah_frame=4)
    jendela = JendelaPalsu()

    kode = app.main(
        ["--kamera"],
        capture_factory=kamera,
        tampil=jendela,
        tutup=lambda: None,
    )

    assert kode == 0
    # 4 frame terbaca, ditambah satu pembacaan terakhir yang False.
    assert kamera.jumlah_baca == 5
    assert len(jendela.panggilan) == 4
    assert kamera.dilepas is True


def test_mode_kamera_berhenti_saat_tombol_q_ditekan(gambar_contoh):
    kamera = KameraPalsu(gambar_contoh, jumlah_frame=50)
    jendela = JendelaPalsu(tombol=ord("q"))

    kode = app.main(
        ["--kamera"],
        capture_factory=kamera,
        tampil=jendela,
        tutup=lambda: None,
    )

    assert kode == 0
    assert len(jendela.panggilan) == 1
    assert kamera.dilepas is True


def test_mode_kamera_tanpa_jendela_dihentikan_oleh_frame_maks(gambar_contoh):
    kamera = KameraPalsu(gambar_contoh, jumlah_frame=50)

    kode = app.main(
        ["--kamera", "--tanpa-jendela", "--frame-maks", "7"],
        capture_factory=kamera,
    )

    assert kode == 0
    assert kamera.jumlah_baca == 7
    assert kamera.dilepas is True


def test_mode_kamera_menyimpan_frame_terakhir(gambar_contoh, tmp_path):
    kamera = KameraPalsu(gambar_contoh, jumlah_frame=3)
    tujuan = tmp_path / "terakhir.jpg"

    kode = app.main(
        ["--kamera", "--tanpa-jendela", "--frame-maks", "3", "--simpan", str(tujuan)],
        capture_factory=kamera,
    )

    assert kode == 0
    assert tujuan.is_file()
    assert cv2.imread(str(tujuan)) is not None


def test_mode_kamera_melempar_galat_bila_kamera_gagal_dibuka(gambar_contoh, capsys):
    kamera = KameraPalsu(gambar_contoh, dibuka=False)

    kode = app.main(
        ["--kamera", "--camera-index", "2"],
        capture_factory=kamera,
    )

    assert kode == 1
    assert kamera.index == 2
    assert "tidak dapat dibuka" in capsys.readouterr().out


def test_mode_kamera_mendeteksi_wajah_pada_frame_kamera(gambar_contoh, capsys):
    """Frame kamera ikut dideteksi wajah dan kotaknya digambar."""
    kamera = KameraPalsu(gambar_contoh, jumlah_frame=1)
    jendela = JendelaPalsu()

    app.main(
        ["--kamera"],
        capture_factory=kamera,
        tampil=jendela,
        tutup=lambda: None,
    )

    assert "Jumlah wajah pada frame terakhir: 2" in capsys.readouterr().out
