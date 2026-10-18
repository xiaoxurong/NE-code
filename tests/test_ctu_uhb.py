"""
Tests for data_provider/ctu_uhb.py on synthetic records written in the CTU-UHB
WFDB format (header comments copied from record 1001 on PhysioNet).

Run: python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest

wfdb = pytest.importorskip("wfdb")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider.ctu_uhb import (  # noqa: E402
    CTGStore, SAMPLES_PER_MIN, build_store, confound_report, make_windows,
    parse_header_comments, scale_fixed,
)

HEADER_1001 = """#----- Additional parameters for record 1001

#-- Outcome measures
#pH           {ph}
#BDecf        8.14
#pCO2         7.7
#BE           -10.5
#Apgar1       6
#Apgar5       8

#-- !NotReadyYet! Neonatology outcome measures !NotReadyYet!
#NICU days    0
#Seizures     0
#HIE          0
#Intubation   0
#Main diag.   0
#Other diag.  0

#-- Fetus/Neonate descriptors
#Gest. weeks  37
#Weight(g)    2660
#Sex          2

#-- Delivery descriptors
#I.stage      232
#II.stage     20
#Deliv. type  1

#-- Signal information
#dbID         974204
#Rec. type    1
#Pos. II.st.  {pos}
#Sig2Birth    {s2b}"""


def header(ph=7.14, pos=14400, s2b=0):
    return HEADER_1001.format(ph=ph, pos=pos, s2b=s2b).splitlines()


def write_record(d, name, n_min, ph, pos, s2b=0, missing=None, seed=0):
    rng = np.random.default_rng(seed)
    n = n_min * SAMPLES_PER_MIN
    fhr = np.round(140 + 10 * rng.standard_normal(n), 2)
    uc = np.round(np.abs(30 + 10 * rng.standard_normal(n)), 2)
    if missing is not None:
        a, b = missing
        fhr[a:b] = 0.0
    comments = [c.lstrip("#") for c in header(ph, pos, s2b) if c.strip()]
    wfdb.wrsamp(name, fs=4, units=["bpm", "nd"], sig_name=["FHR", "UC"],
                p_signal=np.stack([fhr, uc], 1), fmt=["16", "16"],
                adc_gain=[100, 100], baseline=[0, 0], comments=comments, write_dir=d)
    return fhr, uc


@pytest.fixture
def wfdb_dir(tmp_path):
    d = str(tmp_path)
    # 80 min, stage II starts at 60 min, acidemic
    write_record(d, "1001", 80, ph=7.01, pos=60 * SAMPLES_PER_MIN, seed=1)
    # 90 min, no stage II in the recording (position past the end), normal pH
    write_record(d, "1002", 90, ph=7.30, pos=10**6, s2b=5, seed=2)
    # 70 min, last 30 min of stage I mostly missing, normal pH
    write_record(d, "1003", 70, ph=7.25, pos=60 * SAMPLES_PER_MIN, seed=3,
                 missing=(35 * SAMPLES_PER_MIN, 60 * SAMPLES_PER_MIN))
    with open(os.path.join(d, "RECORDS"), "w") as f:
        f.write("1001\n1002\n1003\n")
    return d


def test_parse_header_keeps_known_fields_and_skips_not_ready():
    meta = parse_header_comments(header(ph=7.14, pos=14400, s2b=3))
    assert meta["ph"] == pytest.approx(7.14)
    assert meta["stage2_start_sample"] == 14400
    assert meta["sig_to_birth_min"] == 3
    assert meta["gest_weeks"] == 37
    assert meta["weight_g"] == 2660
    assert meta["stage1_min"] == 232 and meta["stage2_min"] == 20
    assert "hie" not in meta and "seizures" not in meta and "nicu_days" not in meta


def test_build_store_reads_signals_and_marks_zeros_missing(wfdb_dir):
    store = build_store(wfdb_dir)
    assert list(store.record_ids) == ["1001", "1002", "1003"]
    assert list(store.lengths()) == [80 * 240, 90 * 240, 70 * 240]
    fhr, _ = store.signals(2)
    gap = slice(35 * 240, 60 * 240)
    assert np.isnan(fhr[gap]).all()
    assert not np.isnan(fhr[:35 * 240]).any()
    assert np.nanmean(fhr) == pytest.approx(140, abs=2)


def test_store_roundtrip_without_pickle(wfdb_dir, tmp_path):
    store = build_store(wfdb_dir)
    npz, csv = str(tmp_path / "s.npz"), str(tmp_path / "m.csv")
    store.save(npz, csv)
    back = CTGStore.load(npz, csv)
    assert list(back.record_ids) == list(store.record_ids)
    np.testing.assert_array_equal(np.isnan(back.fhr), np.isnan(store.fhr))
    assert back.meta.loc["1001", "ph"] == pytest.approx(7.01)


def test_windows_anchor_at_stage1_end(wfdb_dir):
    store = build_store(wfdb_dir)
    w = make_windows(store, window_min=30, n_windows=1, anchor="stage1_end", min_valid_frac=0.5)
    assert list(w["ph"]) == pytest.approx([7.01, 7.30])
    assert w["recording_min"][0] == pytest.approx(80)
    # 1003's last 30 min of stage I is 25/30 missing, so it is dropped
    assert list(w["subject_ids"]) == ["1001", "1002"]
    fhr1001, _ = store.signals(0)
    np.testing.assert_allclose(w["X"][0, 0], fhr1001[30 * 240:60 * 240], equal_nan=True)
    # 1001: window ends at stage II start = 20 min before the end of an 80-min record
    assert w["min_before_birth"][0] == pytest.approx(20)
    # 1002: no stage II inside, so anchored at the end; Sig2Birth adds 5 min
    assert w["min_before_birth"][1] == pytest.approx(5)
    assert list(w["y"]) == [1, 0]


def test_windows_anchor_at_end_and_multiple_windows(wfdb_dir):
    store = build_store(wfdb_dir)
    w = make_windows(store, window_min=20, n_windows=3, anchor="end", min_valid_frac=0.5)
    for rid in np.unique(w["subject_ids"]):
        k = w["window_index"][w["subject_ids"] == rid]
        assert list(k) == sorted(k)
        assert w["min_before_anchor"][w["subject_ids"] == rid][0] == 0
    # every window of a record carries that record's label
    for rid in np.unique(w["subject_ids"]):
        assert len(set(w["y"][w["subject_ids"] == rid])) == 1
    assert w["X"].shape[1:] == (2, 20 * 240)


def test_windows_never_cross_record_start(wfdb_dir):
    store = build_store(wfdb_dir)
    w = make_windows(store, window_min=30, n_windows=10, anchor="end", min_valid_frac=0.0)
    counts = {rid: int((w["subject_ids"] == rid).sum()) for rid in store.record_ids}
    assert counts == {"1001": 2, "1002": 3, "1003": 2}


def test_scale_fixed_zeroes_missing_and_adds_masks(wfdb_dir):
    store = build_store(wfdb_dir)
    w = make_windows(store, window_min=30, n_windows=2, anchor="end", min_valid_frac=0.0)
    Z = scale_fixed(w["X"], w["mask"], add_mask_channels=True)
    assert Z.shape[1] == 4
    assert np.isfinite(Z).all()
    assert (Z[:, :2][~w["mask"]] == 0).all()
    np.testing.assert_array_equal(Z[:, 2:].astype(bool), w["mask"])
    assert abs(float(Z[:, 0][w["mask"][:, 0]].mean())) < 0.2


def test_confound_report_flags_a_planted_difference(wfdb_dir):
    store = build_store(wfdb_dir)
    rep = confound_report(store, ph_threshold=7.05).set_index("property")
    assert {"recording_min", "fhr_missing_frac", "stage2_min_in_recording"} <= set(rep.index)
    assert rep.loc["recording_min", "mean_pos"] == pytest.approx(80)


def test_bad_window_arguments_raise(wfdb_dir):
    store = build_store(wfdb_dir)
    with pytest.raises(ValueError):
        make_windows(store, window_min=0)
    with pytest.raises(ValueError):
        make_windows(store, anchor="middle")
    with pytest.raises(ValueError):
        make_windows(store, window_min=200)


def test_download_retries_and_resumes(tmp_path, monkeypatch):
    """Simulate PhysioNet with a fake server that fails intermittently (like a 502)."""
    import shutil
    from wfdb.io import download as wfdb_download
    from data_provider import ctu_uhb

    src = tmp_path / "server"
    src.mkdir()
    for i, name in enumerate(["1001", "1002", "1003"]):
        write_record(str(src), name, 40, ph=7.2, pos=10**6, seed=i)

    calls = {"n": 0}

    def flaky_dl(url, save_file_name):
        calls["n"] += 1
        if calls["n"] % 3 == 0:  # every third request fails
            raise ConnectionError("502 Bad Gateway")
        shutil.copy(src / os.path.basename(url), save_file_name)

    monkeypatch.setattr(wfdb_download, "get_version", lambda db: "1.0.0")
    monkeypatch.setattr(wfdb_download, "get_record_list", lambda db: ["1001", "1002", "1003"])
    monkeypatch.setattr(wfdb_download, "dl_full_file", flaky_dl)

    dest = tmp_path / "dl"
    records = ctu_uhb.download(str(dest), attempts=3, pause_s=0)
    assert records == ["1001", "1002", "1003"]
    assert not list(dest.glob("*.part"))
    assert len(build_store(str(dest))) == 3

    # a second run downloads nothing
    before = calls["n"]
    ctu_uhb.download(str(dest), attempts=3, pause_s=0)
    assert calls["n"] == before


def test_download_gives_a_clear_error_when_the_server_stays_down(tmp_path, monkeypatch):
    from wfdb.io import download as wfdb_download
    from data_provider import ctu_uhb

    def always_502(url, save_file_name):
        raise ConnectionError("502 Bad Gateway")

    monkeypatch.setattr(wfdb_download, "get_version", lambda db: "1.0.0")
    monkeypatch.setattr(wfdb_download, "get_record_list", lambda db: ["1001"])
    monkeypatch.setattr(wfdb_download, "dl_full_file", always_502)
    with pytest.raises(RuntimeError, match="rerun the same command to resume"):
        ctu_uhb.download(str(tmp_path / "dl"), attempts=2, pause_s=0)
