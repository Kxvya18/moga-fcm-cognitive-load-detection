





























import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch, find_peaks, butter, filtfilt
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")




DATA_ROOT   = "AffectiveROAD_Data/database"
DRIVER_IDS  = [1, 3, 4, 5, 6]

WINDOW_SEC  = 30
HOP_SEC     = 10

FS_ACC  = 32.0
FS_BVP  = 64.0
FS_EDA  =  4.0
FS_HR   =  1.0
FS_TEMP =  4.0
FS_BIO  =  1.0

OUT_FEATURES = "affective_normalized.csv"
OUT_LABELS   = "affective_gt_labels.csv"





def find_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    for c in candidates:
        m = [col for col in df.columns if c.lower() in col.lower()]
        if m: return m[0]
    return None


def safe_psd(sig, fs, lo, hi):
    if len(sig) < 8: return 0.0
    try:
        f, psd = welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 4)))
        band = (f >= lo) & (f <= hi)
        return float(np.trapz(psd[band], f[band])) if band.any() else 0.0
    except Exception:
        return 0.0


def spectral_entropy(sig, fs):
    if len(sig) < 8: return 0.0
    try:
        _, psd = welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 4)))
        psd /= (psd.sum() + 1e-12)
        return float(-np.sum(psd * np.log2(psd + 1e-12)))
    except Exception:
        return 0.0


def zero_crossing_rate(sig):
    if len(sig) < 2: return 0.0
    return float(np.sum(np.diff(np.sign(sig)) != 0) / (len(sig) - 1))


def hjorth(sig):
    if len(sig) < 4: return 0.0, 0.0, 0.0
    d1 = np.diff(sig); d2 = np.diff(d1)
    act = float(np.var(sig))
    mob = float(np.sqrt(np.var(d1) / (np.var(sig) + 1e-12)))
    cmp = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) / (mob + 1e-12))
    return act, mob, cmp


def sample_entropy(sig, m=2, r=0.2):
    if len(sig) < 10: return 0.0
    try:
        s = (sig - sig.mean()) / (sig.std() + 1e-12)
        N = len(s)
        tm = np.array([s[i:i+m]   for i in range(N - m)])
        tm1= np.array([s[i:i+m+1] for i in range(N - m)])
        cm = cm1 = 0
        for i in range(len(tm)):
            cm  += (np.abs(tm  - tm[i]).max(axis=1)  < r).sum() - 1
            cm1 += (np.abs(tm1 - tm1[i]).max(axis=1) < r).sum() - 1
        return float(-np.log(cm1 / (cm + 1e-12) + 1e-12)) if cm > 0 else 0.0
    except Exception:
        return 0.0


def lowpass(sig, cutoff, fs, order=3):
    try:
        b, a = butter(order, cutoff / (fs / 2), btype='low')
        return filtfilt(b, a, sig)
    except Exception:
        return np.full_like(sig, np.mean(sig))





def read_e4(path, is_ibi=False):

    if not os.path.exists(path): return None, None, None
    try:
        lines = open(path).read().strip().split('\n')
    except Exception:
        return None, None, None
    if len(lines) < 3: return None, None, None

    if is_ibi:
        try: start = float(lines[0].split(',')[0].strip())
        except Exception: return None, None, None
        pairs = []
        for ln in lines[1:]:
            p = ln.strip().split(',')
            try: pairs.append([float(p[0]), float(p[1])])
            except Exception: continue
        return (start, None, np.array(pairs)) if pairs else (None, None, None)

    try:
        start = float(lines[0].split(',')[0].strip())
        fs    = float(lines[1].split(',')[0].strip())
    except Exception:
        return None, None, None

    rows = []
    for ln in lines[2:]:
        p = ln.strip().split(',')
        try: rows.append([float(x) for x in p])
        except Exception: continue
    if not rows: return None, None, None
    data = np.array(rows)
    if data.ndim == 1: data = data.reshape(-1, 1)
    return start, fs, data


def load_e4(driver_id):
    base = os.path.join(DATA_ROOT, "E4", f"{driver_id}-E4-Drv{driver_id}")
    if not os.path.isdir(base):
        print(f"  [E4] Not found: {base}"); return {}
    out = {}
    for side in ["Right", "Left"]:
        sd = os.path.join(base, side)
        if not os.path.isdir(sd): continue
        for fname, mod, ibi_flag in [
            ("ACC.csv","acc",False), ("BVP.csv","bvp",False),
            ("EDA.csv","eda",False), ("HR.csv","hr",False),
            ("TEMP.csv","temp",False), ("IBI.csv","ibi",True)
        ]:
            if mod in out: continue
            t0, fs, data = read_e4(os.path.join(sd, fname), is_ibi=ibi_flag)
            if t0 is not None:
                out[mod] = (t0, fs, data)
                print(f"  [E4] Drv{driver_id} {side}/{fname}: n={len(data)}")
        if out: break
    return out





def e4_slice(e4_dict, mod, frac_start, frac_end):

    if mod not in e4_dict: return None
    _, _, data = e4_dict[mod]
    n = len(data)
    s = max(0, int(frac_start * n))
    e = min(n, int(frac_end   * n))
    if e <= s: return None
    return data[s:e]


def ibi_slice(e4_dict, frac_start, frac_end):

    if 'ibi' not in e4_dict: return None
    _, _, ibi_data = e4_dict['ibi']
    if len(ibi_data) == 0: return None
    offsets = ibi_data[:, 0]
    max_off = offsets[-1]
    lo = frac_start * max_off
    hi = frac_end   * max_off
    mask = (offsets >= lo) & (offsets < hi)
    return ibi_data[mask, 1] if mask.any() else None







def feat_acc(raw):
    d = dict.fromkeys([
        "acc_mag_mean","acc_mag_std","acc_mag_max","acc_mag_range",
        "acc_enmo_mean","acc_enmo_std",
        "acc_jerk_mean","acc_jerk_std","acc_jerk_max",
        "acc_x_std","acc_y_std","acc_z_std",
        "acc_low_power","acc_mid_power","acc_high_power",
        "acc_zcr","acc_spectral_entropy"
    ], 0.0)
    if raw is None or len(raw) < 4: return d
    if raw.ndim == 1: raw = raw.reshape(-1, 1)
    g = raw.astype(np.float64) / 64.0
    mag = np.sqrt((g ** 2).sum(axis=1))
    d["acc_mag_mean"]  = float(np.mean(mag))
    d["acc_mag_std"]   = float(np.std(mag))
    d["acc_mag_max"]   = float(np.max(mag))
    d["acc_mag_range"] = float(np.ptp(mag))
    enmo = np.maximum(mag - 1.0, 0.0)
    d["acc_enmo_mean"] = float(np.mean(enmo))
    d["acc_enmo_std"]  = float(np.std(enmo))
    jerk = np.diff(mag) * FS_ACC
    d["acc_jerk_mean"] = float(np.mean(np.abs(jerk)))
    d["acc_jerk_std"]  = float(np.std(jerk))
    d["acc_jerk_max"]  = float(np.max(np.abs(jerk)))
    for i, ax in enumerate(["x","y","z"]):
        d[f"acc_{ax}_std"] = float(np.std(g[:,i])) if g.shape[1]>i else 0.0
    d["acc_low_power"]        = safe_psd(mag, FS_ACC, 0.0, 1.0)
    d["acc_mid_power"]        = safe_psd(mag, FS_ACC, 1.0, 4.0)
    d["acc_high_power"]       = safe_psd(mag, FS_ACC, 4.0, 12.0)
    d["acc_zcr"]              = zero_crossing_rate(mag - mag.mean())
    d["acc_spectral_entropy"] = spectral_entropy(mag, FS_ACC)
    return d



def feat_bvp(raw):
    d = dict.fromkeys([
        "bvp_mean","bvp_std","bvp_range","bvp_skew",
        "bvp_hr_mean","bvp_hr_std","bvp_rr_mean","bvp_rr_std","bvp_peak_cnt",
        "bvp_vlf_power","bvp_lf_power","bvp_hf_power","bvp_lf_hf",
        "bvp_total_power","bvp_spectral_entropy",
        "bvp_hjorth_act","bvp_hjorth_mob","bvp_hjorth_cmp",
        "bvp_amp_mean","bvp_amp_std",
        "bvp_zcr","bvp_sample_entropy"
    ], 0.0)
    if raw is None or len(raw) < 32: return d
    sig = raw.astype(np.float64).ravel()
    d["bvp_mean"]  = float(np.mean(sig))
    d["bvp_std"]   = float(np.std(sig))
    d["bvp_range"] = float(np.ptp(sig))
    d["bvp_skew"]  = float(stats.skew(sig))
    try:
        pk, props = find_peaks(sig, distance=int(FS_BVP*0.35), prominence=0.01, height=True)
        if len(pk) > 2:
            rr = np.diff(pk)/FS_BVP
            d["bvp_hr_mean"]  = float(60/np.mean(rr))
            d["bvp_hr_std"]   = float(np.std(60/rr))
            d["bvp_rr_mean"]  = float(np.mean(rr))
            d["bvp_rr_std"]   = float(np.std(rr))
            d["bvp_peak_cnt"] = float(len(pk))
            d["bvp_amp_mean"] = float(np.mean(props["peak_heights"]))
            d["bvp_amp_std"]  = float(np.std(props["peak_heights"]))
    except Exception: pass
    d["bvp_vlf_power"]        = safe_psd(sig, FS_BVP, 0.003, 0.04)
    d["bvp_lf_power"]         = safe_psd(sig, FS_BVP, 0.04,  0.15)
    d["bvp_hf_power"]         = safe_psd(sig, FS_BVP, 0.15,  0.40)
    d["bvp_lf_hf"]            = d["bvp_lf_power"]/(d["bvp_hf_power"]+1e-9)
    d["bvp_total_power"]      = safe_psd(sig, FS_BVP, 0.003, 0.40)
    d["bvp_spectral_entropy"] = spectral_entropy(sig, FS_BVP)
    act, mob, cmp             = hjorth(sig)
    d["bvp_hjorth_act"] = act; d["bvp_hjorth_mob"] = mob; d["bvp_hjorth_cmp"] = cmp
    d["bvp_zcr"]            = zero_crossing_rate(sig - sig.mean())
    d["bvp_sample_entropy"] = sample_entropy(sig[:min(200,len(sig))])
    return d



def feat_eda(raw):
    d = dict.fromkeys([
        "eda_mean","eda_std","eda_max","eda_range","eda_skew",
        "eda_scl_mean","eda_scl_std","eda_scl_range",
        "eda_scr_count","eda_scr_amp","eda_scr_rate","eda_phasic_std",
        "eda_phasic_power","eda_tonic_power","eda_vlf_power","eda_spectral_entropy",
        "eda_hjorth_act","eda_hjorth_mob",
        "eda_slope","eda_zcr"
    ], 0.0)
    if raw is None or len(raw) < 8: return d
    sig = raw.astype(np.float64).ravel()
    sig = sig[np.isfinite(sig)]
    if len(sig) < 8: return d
    d["eda_mean"]  = float(np.mean(sig))
    d["eda_std"]   = float(np.std(sig))
    d["eda_max"]   = float(np.max(sig))
    d["eda_range"] = float(np.ptp(sig))
    d["eda_skew"]  = float(stats.skew(sig))
    scl    = lowpass(sig, 0.05, FS_EDA)
    phasic = sig - scl
    d["eda_scl_mean"]  = float(np.mean(scl))
    d["eda_scl_std"]   = float(np.std(scl))
    d["eda_scl_range"] = float(np.ptp(scl))
    try:
        pk, props = find_peaks(phasic, height=0.005, distance=int(FS_EDA), prominence=0.002)
        d["eda_scr_count"]  = float(len(pk))
        d["eda_scr_amp"]    = float(np.mean(props["peak_heights"])) if len(pk)>0 else 0.0
        d["eda_scr_rate"]   = float(len(pk)/(len(sig)/FS_EDA+1e-9))
        d["eda_phasic_std"] = float(np.std(phasic))
    except Exception: pass
    d["eda_phasic_power"]     = safe_psd(phasic, FS_EDA, 0.0, 1.5)
    d["eda_tonic_power"]      = safe_psd(scl,    FS_EDA, 0.0, 0.05)
    d["eda_vlf_power"]        = safe_psd(sig,    FS_EDA, 0.0, 0.05)
    d["eda_spectral_entropy"] = spectral_entropy(sig, FS_EDA)
    act, mob, _               = hjorth(sig)
    d["eda_hjorth_act"] = act; d["eda_hjorth_mob"] = mob
    d["eda_slope"] = float(np.polyfit(np.arange(len(sig)), sig, 1)[0]) if len(sig)>2 else 0.0
    d["eda_zcr"]   = zero_crossing_rate(phasic)
    return d



def feat_e4_hr(raw):
    d = dict.fromkeys([
        "e4_hr_mean","e4_hr_std","e4_hr_max","e4_hr_min","e4_hr_range",
        "e4_hr_rmssd","e4_hr_slope","e4_hr_zcr"
    ], 0.0)
    if raw is None or len(raw) < 2: return d
    sig = raw.astype(np.float64).ravel()
    sig = sig[np.isfinite(sig) & (sig>20) & (sig<220)]
    if len(sig) < 2: return d
    d["e4_hr_mean"]  = float(np.mean(sig))
    d["e4_hr_std"]   = float(np.std(sig))
    d["e4_hr_max"]   = float(np.max(sig))
    d["e4_hr_min"]   = float(np.min(sig))
    d["e4_hr_range"] = float(np.ptp(sig))
    d["e4_hr_rmssd"] = float(np.sqrt(np.mean(np.diff(sig)**2))) if len(sig)>1 else 0.0
    d["e4_hr_slope"] = float(np.polyfit(np.arange(len(sig)),sig,1)[0]) if len(sig)>2 else 0.0
    d["e4_hr_zcr"]   = zero_crossing_rate(sig-sig.mean())
    return d



def feat_hrv(ibi_durs):
    d = dict.fromkeys([
        "hrv_mean_rr","hrv_sdnn","hrv_rmssd","hrv_sdsd",
        "hrv_pnn20","hrv_pnn50","hrv_cv","hrv_range_rr",
        "hrv_vlf_power","hrv_lf_power","hrv_hf_power","hrv_lf_hf",
        "hrv_total_power","hrv_spectral_entropy",
        "hrv_sd1","hrv_sd2","hrv_sd1_sd2","hrv_sample_entropy",
        "hrv_stress_index","hrv_mode_rr"
    ], 0.0)
    if ibi_durs is None or len(ibi_durs) < 4: return d
    rr = ibi_durs[np.isfinite(ibi_durs)]
    rr = rr[(rr>0.3)&(rr<2.0)]
    if len(rr) < 4: return d
    d["hrv_mean_rr"]  = float(np.mean(rr))
    d["hrv_sdnn"]     = float(np.std(rr))
    diff_rr           = np.diff(rr)
    d["hrv_rmssd"]    = float(np.sqrt(np.mean(diff_rr**2)))
    d["hrv_sdsd"]     = float(np.std(diff_rr))
    adiff             = np.abs(diff_rr)
    d["hrv_pnn20"]    = float(np.mean(adiff>0.020))
    d["hrv_pnn50"]    = float(np.mean(adiff>0.050))
    d["hrv_cv"]       = float(np.std(rr)/(np.mean(rr)+1e-9))
    d["hrv_range_rr"] = float(np.ptp(rr))
    try:
        t = np.cumsum(rr)-rr[0]
        if t[-1]>0:
            t_e = np.arange(0, t[-1], 0.25)
            rre = np.interp(t_e, t, rr)
            d["hrv_vlf_power"]       = safe_psd(rre,4.0,0.003,0.04)
            d["hrv_lf_power"]        = safe_psd(rre,4.0,0.04,0.15)
            d["hrv_hf_power"]        = safe_psd(rre,4.0,0.15,0.40)
            d["hrv_lf_hf"]           = d["hrv_lf_power"]/(d["hrv_hf_power"]+1e-9)
            d["hrv_total_power"]     = d["hrv_vlf_power"]+d["hrv_lf_power"]+d["hrv_hf_power"]
            d["hrv_spectral_entropy"]= spectral_entropy(rre,4.0)
    except Exception: pass
    if len(rr)>2:
        sd1 = float(np.std(diff_rr)/np.sqrt(2))
        sd2 = float(np.sqrt(max(2*np.var(rr)-sd1**2,0)))
        d["hrv_sd1"]     = sd1
        d["hrv_sd2"]     = sd2
        d["hrv_sd1_sd2"] = sd1/(sd2+1e-9)
    d["hrv_sample_entropy"] = sample_entropy(rr[:min(len(rr),50)])
    try:
        hist, edges = np.histogram(rr, bins=max(4,len(rr)//4))
        mo  = float(edges[np.argmax(hist)]+(edges[1]-edges[0])/2)
        amo = float(np.sum(np.abs(rr-mo)<0.05)/(len(rr)+1e-9))
        mx  = float(np.ptp(rr))
        d["hrv_stress_index"] = float(amo/(2*mo*mx+1e-9))
        d["hrv_mode_rr"]      = mo
    except Exception: pass
    return d



def feat_temp(raw):
    d = dict.fromkeys([
        "temp_mean","temp_std","temp_min","temp_max","temp_range","temp_skew",
        "temp_slope","temp_deriv_mean","temp_deriv_std","temp_power"
    ], 0.0)
    if raw is None or len(raw) < 4: return d
    sig = raw.astype(np.float64).ravel()
    sig = sig[np.isfinite(sig) & (sig>20) & (sig<45)]
    if len(sig) < 4: return d
    d["temp_mean"]  = float(np.mean(sig))
    d["temp_std"]   = float(np.std(sig))
    d["temp_min"]   = float(np.min(sig))
    d["temp_max"]   = float(np.max(sig))
    d["temp_range"] = float(np.ptp(sig))
    d["temp_skew"]  = float(stats.skew(sig))
    d["temp_slope"] = float(np.polyfit(np.arange(len(sig)),sig,1)[0]) if len(sig)>2 else 0.0
    drv = np.diff(sig)*FS_TEMP
    d["temp_deriv_mean"] = float(np.mean(np.abs(drv)))
    d["temp_deriv_std"]  = float(np.std(drv))
    d["temp_power"]      = safe_psd(sig, FS_TEMP, 0.0, 0.5)
    return d



def feat_bio_hr(arr):
    d = dict.fromkeys([
        "bio_hr_mean","bio_hr_std","bio_hr_max","bio_hr_min","bio_hr_range",
        "bio_hr_rmssd","bio_hr_slope","bio_hr_cv"
    ], 0.0)
    if arr is None or len(arr)<2: return d
    sig = arr[np.isfinite(arr)&(arr>20)&(arr<220)]
    if len(sig)<2: return d
    d["bio_hr_mean"]  = float(np.mean(sig))
    d["bio_hr_std"]   = float(np.std(sig))
    d["bio_hr_max"]   = float(np.max(sig))
    d["bio_hr_min"]   = float(np.min(sig))
    d["bio_hr_range"] = float(np.ptp(sig))
    d["bio_hr_rmssd"] = float(np.sqrt(np.mean(np.diff(sig)**2))) if len(sig)>1 else 0.0
    d["bio_hr_slope"] = float(np.polyfit(np.arange(len(sig)),sig,1)[0]) if len(sig)>2 else 0.0
    d["bio_hr_cv"]    = float(np.std(sig)/(np.mean(sig)+1e-9))
    return d



def feat_bio_br(arr):
    d = dict.fromkeys([
        "bio_br_mean","bio_br_std","bio_br_max","bio_br_min","bio_br_range",
        "bio_br_rmssd","bio_br_slope","bio_br_cv"
    ], 0.0)
    if arr is None or len(arr)<2: return d
    sig = arr[np.isfinite(arr)&(arr>0)&(arr<60)]
    if len(sig)<2: return d
    d["bio_br_mean"]  = float(np.mean(sig))
    d["bio_br_std"]   = float(np.std(sig))
    d["bio_br_max"]   = float(np.max(sig))
    d["bio_br_min"]   = float(np.min(sig))
    d["bio_br_range"] = float(np.ptp(sig))
    d["bio_br_rmssd"] = float(np.sqrt(np.mean(np.diff(sig)**2))) if len(sig)>1 else 0.0
    d["bio_br_slope"] = float(np.polyfit(np.arange(len(sig)),sig,1)[0]) if len(sig)>2 else 0.0
    d["bio_br_cv"]    = float(np.std(sig)/(np.mean(sig)+1e-9))
    return d



def feat_posture(arr):
    d = dict.fromkeys([
        "posture_mean","posture_std","posture_range","posture_skew",
        "posture_zcr","posture_slope"
    ], 0.0)
    if arr is None or len(arr)<2: return d
    sig = arr[np.isfinite(arr)]
    if len(sig)<2: return d
    d["posture_mean"]  = float(np.mean(sig))
    d["posture_std"]   = float(np.std(sig))
    d["posture_range"] = float(np.ptp(sig))
    d["posture_skew"]  = float(stats.skew(sig))
    d["posture_zcr"]   = zero_crossing_rate(sig-sig.mean())
    d["posture_slope"] = float(np.polyfit(np.arange(len(sig)),sig,1)[0]) if len(sig)>2 else 0.0
    return d



def feat_activity(arr):
    d = dict.fromkeys([
        "activity_mean","activity_std","activity_max","activity_range","activity_power"
    ], 0.0)
    if arr is None or len(arr)<2: return d
    sig = arr[np.isfinite(arr)]
    if len(sig)<2: return d
    d["activity_mean"]  = float(np.mean(sig))
    d["activity_std"]   = float(np.std(sig))
    d["activity_max"]   = float(np.max(sig))
    d["activity_range"] = float(np.ptp(sig))
    d["activity_power"] = safe_psd(sig, FS_BIO, 0.0, 0.5)
    return d



def feat_cross(eda_1d, e4_hr_1d, bio_hr_1d, bio_br_1d, acc_mag_1hz,
               hrv_hf, eda_scl_mean):
    d = dict.fromkeys([
        "cross_eda_hr_corr","cross_hr_hr_corr","cross_hr_hr_diff",
        "cross_br_acc_corr","cross_sympathetic_idx","cross_autonomic_balance",
        "cross_bvp_eda_corr","cross_motion_eda_corr","cross_hr_br_corr"
    ], 0.0)

    def safe_corr(a, b):
        n = min(len(a), len(b))
        if n < 4: return 0.0
        try:
            c = np.corrcoef(a[:n], b[:n])[0,1]
            return 0.0 if not np.isfinite(c) else float(c)
        except Exception:
            return 0.0

    if eda_1d is not None and e4_hr_1d is not None:
        d["cross_eda_hr_corr"] = safe_corr(eda_1d, e4_hr_1d)
    if bio_hr_1d is not None and e4_hr_1d is not None:
        d["cross_hr_hr_corr"] = safe_corr(bio_hr_1d, e4_hr_1d)
        n = min(len(bio_hr_1d), len(e4_hr_1d))
        if n >= 4:
            d["cross_hr_hr_diff"] = float(np.mean(np.abs(bio_hr_1d[:n]-e4_hr_1d[:n])))
    if bio_br_1d is not None and acc_mag_1hz is not None:
        d["cross_br_acc_corr"] = safe_corr(bio_br_1d, acc_mag_1hz)
    if bio_hr_1d is not None and bio_br_1d is not None:
        d["cross_hr_br_corr"] = safe_corr(bio_hr_1d, bio_br_1d)
    eda_m = float(np.mean(eda_1d))  if eda_1d   is not None and len(eda_1d)>0  else 0.0
    hr_m  = float(np.mean(e4_hr_1d))if e4_hr_1d is not None and len(e4_hr_1d)>0 else 0.0
    d["cross_sympathetic_idx"]   = eda_m * hr_m / 1000.0
    d["cross_autonomic_balance"] = hrv_hf / (eda_scl_mean + 1e-9)
    if eda_1d is not None and acc_mag_1hz is not None:
        d["cross_motion_eda_corr"] = safe_corr(
            np.interp(np.linspace(0,1,len(eda_1d)),
                      np.linspace(0,1,len(acc_mag_1hz)), acc_mag_1hz),
            eda_1d)
    return d





_all_keys = (
    list(feat_acc(None).keys())       +
    list(feat_bvp(None).keys())       +
    list(feat_eda(None).keys())       +
    list(feat_e4_hr(None).keys())     +
    list(feat_hrv(None).keys())       +
    list(feat_temp(None).keys())      +
    list(feat_bio_hr(None).keys())    +
    list(feat_bio_br(None).keys())    +
    list(feat_posture(None).keys())   +
    list(feat_activity(None).keys())  +
    list(feat_cross(None,None,None,None,None,0,0).keys())
)
assert len(_all_keys) == len(set(_all_keys)),\
    f"DUPLICATE feature names: {[k for k in _all_keys if _all_keys.count(k)>1]}"
print(f"Total feature keys: {len(_all_keys)}  (target: 133)")





all_feat_rows  = []
all_label_rows = []

for drv in DRIVER_IDS:
    print(f"\n{'─'*65}\n  Driver {drv}\n{'─'*65}")


    bio_path = os.path.join(DATA_ROOT,"Bioharness",f"Bio_Drv{drv}.csv")
    if not os.path.exists(bio_path):
        print(f"  Bio not found: {bio_path}"); continue
    try:
        bio = pd.read_csv(bio_path, sep=';')
        if bio.shape[1] < 2:
            bio = pd.read_csv(bio_path)
    except Exception:
        print(f"  Bio parse error"); continue
    bio.columns = [c.strip() for c in bio.columns]
    print(f"  Bio: {len(bio)} rows, cols={list(bio.columns)}")

    sm_vals = None
    sm_path = os.path.join(DATA_ROOT,"Subj_metric",f"SM_Drv{drv}.csv")
    if os.path.exists(sm_path):
        sm_df = pd.read_csv(sm_path, header=None)
        best_col, best_n = 0, 0
        for c in sm_df.columns:
            n = pd.to_numeric(sm_df.iloc[:,c],errors='coerce').notna().sum()
            if n > best_n: best_n,best_col = n,c
        sm_vals = pd.to_numeric(sm_df.iloc[:,best_col],errors='coerce')\
                    .interpolate().ffill().bfill().values.astype(np.float64)
        print(f"  SM:  {len(sm_vals)} samples, range=[{sm_vals.min():.2f},{sm_vals.max():.2f}]")
    if sm_vals is None:
        print(f"  SM not found"); continue

    e4 = load_e4(drv)


    HR_C  = find_col(bio,["HR","heart_rate","HeartRate"])
    BR_C  = find_col(bio,["BR","breathing_rate","Resp"])
    POS_C = find_col(bio,["Posture","posture"])
    ACT_C = find_col(bio,["Activity","activity","Accel"])

    n_bio = len(bio)


    if len(sm_vals) != n_bio:
        sm_vals = np.interp(np.linspace(0,1,n_bio),
                             np.linspace(0,1,len(sm_vals)), sm_vals)
    sm_disc = np.zeros(n_bio, dtype=int)
    if np.ptp(sm_vals) > 1e-9:
        q33,q66 = np.percentile(sm_vals,[33,66])
        sm_disc[sm_vals>q33]=1
        sm_disc[sm_vals>q66]=2

    n_win = WINDOW_SEC
    hop   = HOP_SEC

    if n_bio < n_win:
        print(f"  Too short ({n_bio}); skip."); continue

    feat_rows  = []
    label_rows = []

    for idx in range(0, n_bio - n_win + 1, hop):
        bw = bio.iloc[idx:idx+n_win]

        def _bcol(col):
            if col is None or col not in bw.columns: return None
            v = pd.to_numeric(bw[col],errors='coerce').values.astype(np.float64)
            return v[np.isfinite(v)]


        frac_s = idx / max(n_bio - 1, 1)
        frac_e = (idx + n_win) / max(n_bio - 1, 1)

        acc_raw  = e4_slice(e4,'acc', frac_s, frac_e)
        bvp_raw  = e4_slice(e4,'bvp', frac_s, frac_e)
        eda_raw  = e4_slice(e4,'eda', frac_s, frac_e)
        hr_raw   = e4_slice(e4,'hr',  frac_s, frac_e)
        temp_raw = e4_slice(e4,'temp',frac_s, frac_e)
        ibi_durs = ibi_slice(e4, frac_s, frac_e)


        eda_1d  = eda_raw.ravel()  if eda_raw  is not None else None
        hr_1d   = hr_raw.ravel()   if hr_raw   is not None else None
        temp_1d = temp_raw.ravel() if temp_raw is not None else None
        bvp_1d  = bvp_raw.ravel()  if bvp_raw  is not None else None


        if acc_raw is not None and len(acc_raw) >= 4:
            if acc_raw.ndim==1: acc_raw=acc_raw.reshape(-1,1)
            g = acc_raw.astype(np.float64)/64.0
            mag_hi = np.sqrt((g**2).sum(axis=1))
            acc_mag_1hz = np.array([mag_hi[int(i*len(mag_hi)/n_win)]
                                    for i in range(n_win)])
        else:
            acc_mag_1hz = None

        f = {}
        f.update(feat_acc(acc_raw))
        f.update(feat_bvp(bvp_1d))
        f.update(feat_eda(eda_1d))
        f.update(feat_e4_hr(hr_1d))
        f.update(feat_hrv(ibi_durs))
        f.update(feat_temp(temp_1d))
        f.update(feat_bio_hr(_bcol(HR_C)))
        f.update(feat_bio_br(_bcol(BR_C)))
        f.update(feat_posture(_bcol(POS_C)))
        f.update(feat_activity(_bcol(ACT_C)))
        f.update(feat_cross(
            eda_1d, hr_1d, _bcol(HR_C), _bcol(BR_C), acc_mag_1hz,
            f.get("hrv_hf_power",0.0), f.get("eda_scl_mean",1.0)
        ))

        gt_w  = sm_disc[idx:idx+n_win]
        gt_lb = int(stats.mode(gt_w,keepdims=True)[0][0])
        gt_sm = float(np.mean(sm_vals[idx:idx+n_win]))

        f["driver_id"]    = int(drv)
        f["window_start"] = int(idx)
        f["gt_label"]     = gt_lb
        feat_rows.append(f)
        label_rows.append({
            "driver_id":    int(drv),
            "window_start": int(idx),
            "gt_label":     gt_lb,
            "gt_sm_mean":   round(gt_sm,4),
            "gt_sm_level":  ["LOW_LOAD","MEDIUM_LOAD","HIGH_LOAD"][gt_lb],
        })

    all_feat_rows.extend(feat_rows)
    all_label_rows.extend(label_rows)
    ld = {0:0,1:0,2:0}
    for r in label_rows: ld[r["gt_label"]]+=1
    print(f"  → {len(feat_rows)} windows  L={ld[0]} M={ld[1]} H={ld[2]}")





if not all_feat_rows:
    print("\n!! No features extracted."); raise SystemExit(1)

full_df   = pd.DataFrame(all_feat_rows)
meta_cols = ["driver_id","window_start","gt_label"]
feat_cols = [c for c in full_df.columns if c not in meta_cols]


full_df[feat_cols] = full_df[feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)


var = full_df[feat_cols].var()
keep = var[var>1e-10].index.tolist()

print(f"\n{'='*65}")
print(f"  FEATURE SUMMARY")
print(f"{'='*65}")
print(f"  Raw features defined : {len(feat_cols)}")
print(f"  Non-zero variance    : {len(keep)}")
print(f"  Total windows        : {len(full_df)}")
for d in DRIVER_IDS:
    n=(full_df["driver_id"]==d).sum()
    if n>0: print(f"    Driver {d}: {n} windows")

scaler   = MinMaxScaler()
feat_mat = scaler.fit_transform(full_df[keep].values)
feat_df  = pd.DataFrame(feat_mat, columns=keep)
out_df   = pd.concat([feat_df,
                       full_df[meta_cols].reset_index(drop=True)], axis=1)
out_df.to_csv(OUT_FEATURES, index=False)
pd.DataFrame(all_label_rows).to_csv(OUT_LABELS, index=False)

print(f"\n  Saved: {OUT_FEATURES}  shape={out_df.shape}")
print(f"  Saved: {OUT_LABELS}")
ldf = pd.DataFrame(all_label_rows)
for lv in ["LOW_LOAD","MEDIUM_LOAD","HIGH_LOAD"]:
    n=(ldf["gt_sm_level"]==lv).sum()
    print(f"    {lv}: {n} ({100*n/len(ldf):.1f}%)")

print(f"\n  Feature list ({len(keep)}):")
for i,f in enumerate(keep):
    print(f"    {i+1:3d}. {f}")
