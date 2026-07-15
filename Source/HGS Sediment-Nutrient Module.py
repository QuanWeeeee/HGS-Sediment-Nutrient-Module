import matplotlib.pyplot as plt 
import matplotlib.backends.backend_pdf
from tkinter import filedialog
import math
import numpy as np
import re
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.colors as colors
from matplotlib import cm
import pandas as pd
# Runtime logging and stdout configuration
import os, sys, time, threading, logging, faulthandler, signal
import runpy

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else (os.path.dirname(SOURCE_DIR) if os.path.basename(SOURCE_DIR).lower() == "source" else SOURCE_DIR)
RUN_LOG_PATH = os.path.join(APP_DIR, "Run_Log.txt")

# Master switch: True opens the GUI; False runs the model directly.
ENABLE_GUI = True

faulthandler.enable()
faulthandler.dump_traceback_later(600, repeat=True)

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(RUN_LOG_PATH, mode="a", encoding="utf-8")]
)
def log(msg): logging.info(msg)

def load_params_file(path="Model_Config.txt"):
    if not os.path.isabs(path):
        path = os.path.join(APP_DIR, path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Required parameter file not found: {path}")

    log(f"[INFO] Loading params from: {path}")
    cfg = runpy.run_path(path)
    return {k: v for k, v in cfg.items() if not k.startswith("__")}

def require_param(d, key):
    if key not in d:
        raise KeyError(f"Missing required parameter in Model_Config.txt: {key}")
    return d[key]

def apply_params(d):
    global prf, SC, cch, P_USLE, L_USLE, rock, Afallow, Acrop
    global C_USLE, c_sp, sp_exp, K_ch, a, b, conc0_global
    global Qcrit_m3_day, resus_k_mgL, resus_b
    global C_base_mgL, C_resus_cap_mgL, conc0_global , Qmin_station_m3_day
    global depth_threshold, chan_map_dist, converce_difference, ROUTING_MAX_ITER
    global HILLSLOPE_CARRY_DECAY, CHANNEL_EXPORT_FRAC, USE_POST_ROUTING_CONC
    global QPEAK_METHOD, MANNING_N, QPEAK_MANNING_FACTOR
    global CHANNEL_NEIGHBOR_DEPTH_THRESHOLD, CHANNEL_NEIGHBOR_RINGS, CHANNEL_NEIGHBOR_DEPTH_SOURCE
    global USE_DYNAMIC_CHANNEL_MASK
    global USE_BASE_CONC, USE_RESUSPENSION, USE_MUSLE_EVENT, MUSLE_SSC_WEIGHT
    global USE_EVENT_SATURATION, EVENT_SATURATION_CAP_MGL, EVENT_SATURATION_GAMMA
    global ENABLE_DAILY_LOOP, DAILY_LOOP_USE_LAST, USE_TIME_ROUTING_BINARY
    global PHI_PRODUC_TIME_FRAC, EPS_V, plot_sediment_map_mode, plot_exchange_flux_map_mode
    global pest_mode, WRITE_CELL_SSC_OUTPUTS, ENABLE_NUTRIENT_MODULE, NUTRIENT_USE_PROXY_FROM_OLF
    global SOIL_N_CONC_KG_PER_TON, SOIL_P_CONC_KG_PER_TON
    global ER_A, ER_B, ER_MIN, ER_MAX, SED_EPS_T_DAY
    global N_EXCHANGE_TO_KG_PER_M3, P_SPECIFIED_TO_KG_PER_M3
    if not d:
        raise ValueError("Model_Config.txt is empty; all model parameters must come from this file")

    prf = float(require_param(d, "prf"))
    SC = float(require_param(d, "SC"))
    cch = float(require_param(d, "cch"))
    P_USLE = float(require_param(d, "P_USLE"))
    L_USLE = float(require_param(d, "L_USLE"))
    rock = float(require_param(d, "rock"))
    Afallow = float(require_param(d, "Afallow"))
    Acrop = float(require_param(d, "Acrop"))
    C_USLE = float(require_param(d, "C_USLE"))
    c_sp = float(require_param(d, "c_sp"))
    sp_exp = int(require_param(d, "sp_exp"))
    K_ch = float(require_param(d, "K_ch"))

    a = float(require_param(d, "a"))
    b = float(require_param(d, "b"))
    conc0_global = float(require_param(d, "conc0_global"))
    Qcrit_m3_day = float(require_param(d, "Qcrit_m3_day"))
    resus_k_mgL = float(require_param(d, "resus_k_mgL"))
    resus_b = float(require_param(d, "resus_b"))

    C_base_mgL = float(require_param(d, "C_base_mgL"))
    C_resus_cap_mgL = float(require_param(d, "C_resus_cap_mgL"))
    Qmin_station_m3_day = float(require_param(d, "Qmin_station_m3_day"))
    depth_threshold = float(require_param(d, "depth_mask_threshold"))
    chan_map_dist = float(require_param(d, "chan_map_dist_m"))
    converce_difference = float(require_param(d, "converce_difference"))
    ROUTING_MAX_ITER = int(require_param(d, "routing_max_iter"))
    HILLSLOPE_CARRY_DECAY = float(require_param(d, "hillslope_carry_decay"))
    CHANNEL_EXPORT_FRAC = float(require_param(d, "channel_export_frac"))
    USE_POST_ROUTING_CONC = bool(int(require_param(d, "use_post_routing_conc")))
    QPEAK_METHOD = str(require_param(d, "qpeak_method")).lower()
    MANNING_N = float(require_param(d, "manning_n"))
    QPEAK_MANNING_FACTOR = float(require_param(d, "qpeak_manning_factor"))
    CHANNEL_NEIGHBOR_DEPTH_THRESHOLD = float(require_param(d, "channel_neighbor_depth_threshold"))
    CHANNEL_NEIGHBOR_RINGS = int(require_param(d, "channel_neighbor_rings"))
    CHANNEL_NEIGHBOR_DEPTH_SOURCE = str(require_param(d, "channel_neighbor_depth_source")).lower()
    USE_DYNAMIC_CHANNEL_MASK = bool(int(require_param(d, "use_dynamic_channel_mask")))
    USE_BASE_CONC = bool(int(require_param(d, "use_base_conc")))
    USE_RESUSPENSION = bool(int(require_param(d, "use_resuspension")))
    USE_MUSLE_EVENT = bool(int(require_param(d, "use_musle_event")))
    MUSLE_SSC_WEIGHT = float(require_param(d, "musle_ssc_weight"))
    USE_EVENT_SATURATION = bool(int(require_param(d, "use_event_saturation")))
    EVENT_SATURATION_CAP_MGL = float(require_param(d, "event_saturation_cap_mgL"))
    EVENT_SATURATION_GAMMA = float(require_param(d, "event_saturation_gamma"))
    ENABLE_DAILY_LOOP = bool(int(require_param(d, "enable_daily_loop")))
    DAILY_LOOP_USE_LAST = bool(int(require_param(d, "daily_loop_use_last")))
    USE_TIME_ROUTING_BINARY = bool(int(require_param(d, "use_time_routing_binary")))
    PHI_PRODUC_TIME_FRAC = float(require_param(d, "phi_produc_time_frac"))
    EPS_V = float(require_param(d, "eps_v"))
    plot_sediment_map_mode = bool(int(require_param(d, "plot_sediment_map_mode")))
    plot_exchange_flux_map_mode = bool(int(require_param(d, "plot_exchange_flux_map_mode")))
    pest_mode = bool(int(require_param(d, "pest_mode")))
    # Per-cell SSC files are intentionally disabled for this run variant.
    WRITE_CELL_SSC_OUTPUTS = False
    ENABLE_NUTRIENT_MODULE = bool(int(require_param(d, "enable_nutrient_module")))
    NUTRIENT_USE_PROXY_FROM_OLF = bool(int(require_param(d, "nutrient_use_proxy_from_olf")))
    SOIL_N_CONC_KG_PER_TON = float(require_param(d, "soil_n_conc_kg_per_ton"))
    SOIL_P_CONC_KG_PER_TON = float(require_param(d, "soil_p_conc_kg_per_ton"))
    ER_A = float(require_param(d, "er_a"))
    ER_B = float(require_param(d, "er_b"))
    ER_MIN = float(require_param(d, "er_min"))
    ER_MAX = float(require_param(d, "er_max"))
    SED_EPS_T_DAY = float(require_param(d, "sed_eps_t_day"))
    N_EXCHANGE_TO_KG_PER_M3 = float(require_param(d, "n_exchange_to_kg_per_m3"))
    P_SPECIFIED_TO_KG_PER_M3 = float(require_param(d, "p_specified_to_kg_per_m3"))
def _heartbeat():
    while True:
        log("[RUN] alive")
        time.sleep(2)

threading.Thread(target=_heartbeat, daemon=True).start()
log("Booting script...")
prf = 1.0 
SC = 1 
cch = 1.0 
P_USLE = 0.8 
L_USLE = 1.14 
rock = 0.4 
Afallow = 1.0 
Acrop = 1.0 
C_USLE = 0.0035
c_sp = 1.0e-4 
sp_exp = 1.5 
K_ch = 0
olf_data = {}            # {time: {section_name: list_of_floats}}
solution_times = []   
track_records = []  
a = 0.02   # SDR size
b = 1.2  # SDR curve
conc0_global = 20.0  # mg/L
Qcrit_m3_day = 200.0
resus_k_mgL = 0.005
resus_b = 0.5
_AREA_M2_CACHE = None
C_base_mgL = 3.0
C_resus_cap_mgL = 30.0
Qmin_station_m3_day = 1e-7
HILLSLOPE_CARRY_DECAY = 0.85
CHANNEL_EXPORT_FRAC = 0.10
USE_POST_ROUTING_CONC = False
QPEAK_METHOD = "manning"
MANNING_N = 0.05
QPEAK_MANNING_FACTOR = 0.2
CHANNEL_NEIGHBOR_DEPTH_THRESHOLD = 1e-3
CHANNEL_NEIGHBOR_RINGS = 1
CHANNEL_NEIGHBOR_DEPTH_SOURCE = "current"
USE_DYNAMIC_CHANNEL_MASK = True
USE_BASE_CONC = True
USE_RESUSPENSION = True
USE_MUSLE_EVENT = True
MUSLE_SSC_WEIGHT = 0.25
USE_EVENT_SATURATION = True
EVENT_SATURATION_CAP_MGL = 500
EVENT_SATURATION_GAMMA = 1.5
# Mass-balance cumulative trackers
MASSBAL_IN_CUM = 0.0
MASSBAL_OUT_CUM = 0.0
MASSBAL_STORAGE_LAST = 0.0  # last computed in-domain storage
depth_threshold = 1e-6
chan_map_dist = 100.0
DT_SEC = 86400.0  # 1 day in seconds
DT_DAY = 1.0
# Nutrient transport defaults
SOIL_N_CONC_KG_PER_TON = 1.0   
SOIL_P_CONC_KG_PER_TON = 0.2
ER_A = 1.0
ER_B = 0.2
ER_MIN = 0.1
ER_MAX = 5.0
SED_EPS_T_DAY = 1.0e-12
N_EXCHANGE_TO_KG_PER_M3 = 1.0
P_SPECIFIED_TO_KG_PER_M3 = 1.0  
converce_difference = 5.0e-3
ROUTING_MAX_ITER = 500
MESH_TIME = None
CHANNEL_MASK = None   # combined mask = DEPTH_MASK OR CHAN_MASK
CHAN_MASK = None      # true CHAN channel elements
DEPTH_MASK = None     # active-water cells from depth
# Mesh fields are read from the mesh timestep.
MESH_KEYS = {
    "node_x": ["x", "node x", "X"],
    "node_y": ["y", "node y", "Y"],
    "node_z": ["z", "node z", "Z"],
    "elem_nodes": ["element node lists", "element nodes", "tri node list"],
    "zone": ["zone (cell centred)","zone"]
}
# Field variables are read at each solution timestep.
FIELD_KEYS = {
    "flow_rate_node": [
        "flow rate", "Flow rate", "flowrate",
        "surface flow rate", "overland flow rate"
    ],
    "vx_cell": [
        "x linear velocity (cell centred)",
        "x velocity (cell centred)", "x velocity (cell-centered)"
    ],
    "vy_cell": [
        "y linear velocity (cell centred)",
        "y velocity (cell centred)", "y velocity (cell-centered)"
    ],
    "vz_cell": [
        "z linear velocity (cell centred)",
        "z velocity (cell centred)"
    ],
    "depth_node": [
        "depth", "Depth", "water depth"
    ],
    "Precipitation": [
        "Precipitation", "precipitation", "Rain", "rain"
    ],
    "exchange_flux_node":[
        "Exchange flux", "exchange flux", "Exchange Flux",
        "exchange_flux", "EXCHANGE FLUX","specific"
    ],
    "nutrient_solute_node": [
        "Groundwater_N", "groundwater_n", "Groundwater N",
        "solute concentration", "Solute concentration"
    ],
    "specified_flux_node":[
        "Specified flux", "specified flux", "Specified Flux",
        "specified_flux", "SPECIFIED FLUX","specific"
    ]
}

USE_GEOM_AREA = True

global_coords = {}       # {'x1','y1','x2','y2','x3','y3'} for mesh
global_triangles = []    # list of triangles coords (for plotting)
global_neighbors = {}

# Runtime switches
ENABLE_DAILY_LOOP = True
DAILY_LOOP_USE_LAST = True       # plot last simulated day when daily loop is enabled
USE_TIME_ROUTING_BINARY = True
PHI_PRODUC_TIME_FRAC = 0.0
EPS_V = 1e-9
plot_sediment_map_mode = False
plot_exchange_flux_map_mode = False
pest_mode = False
WRITE_CELL_SSC_OUTPUTS = False
# Optional user-supplied overland distance to channel (m) per cell.
EXTERNAL_OVERLAND_LENGTHS = None
ENABLE_NUTRIENT_MODULE = False
NUTRIENT_USE_PROXY_FROM_OLF = False




_time_series_cache_binary = None  # {time: np.ndarray of routed sediment (t/day per cell)}


def _characteristic_length_per_cell():
    """Return per-cell overland distance to the channel in meters."""
    import numpy as _np
    N = triangle_count()
    try:
        if EXTERNAL_OVERLAND_LENGTHS is not None:
            arr = _np.asarray(EXTERNAL_OVERLAND_LENGTHS, dtype=float).reshape(-1)
            if arr.size == N and _np.all(_np.isfinite(arr)) and _np.all(arr >= 0):
                return [float(max(1e-6, x)) for x in arr.tolist()]
    except Exception:
        pass
    try:
        if olf_data:
            any_t = next(iter(olf_data))
            candidate = olf_data[any_t].get('overland_length', None)
            if candidate is not None:
                arr = _np.asarray(candidate, dtype=float).reshape(-1)
                if arr.size == N:
                    return [float(max(1e-6, x)) for x in arr.tolist()]
    except Exception:
        pass
    areas_ha = compute_triangle_areas()
    areas_m2 = [a*10000.0 for a in areas_ha]
    return [max(1e-6, (A**0.5)) for A in areas_m2]

def _tau_h_days_for_time(time, qch_list, lam_list):
    """Compute hillslope travel time per cell in days: tau_h = L / v (m / (m/s)) / 86400."""
    tau_days = []
    for v, L in zip(qch_list, lam_list):
        vv = float(v) if v is not None else 0.0
        tau = (L / max(vv, EPS_V)) / 86400.0
        tau_days.append(tau)
    return tau_days
def calc_station_discharge_m3s(time, station_mask_cells, chan_mask_cell=None, mode="possum"):

    Q_cell = cal_Qflow_cell_m3_s(time)  # m3/s per cell (signed possible)

    # optional: restrict to channel cells only
    if chan_mask_cell is not None:
        Q_cell = np.where(chan_mask_cell, Q_cell, 0.0)

    Q_sel = Q_cell[station_mask_cells]

    if mode == "net":
        Q_station = float(np.sum(Q_sel))
    elif mode == "abssum":
        Q_station = float(np.sum(np.abs(Q_sel)))
    else:  # "possum"
        Q_station = float(np.sum(np.maximum(Q_sel, 0.0)))

    return max(Q_station, 0.0)

def _cell_area_m2_array():
    """Return cached cell areas in m2."""
    global _AREA_M2_CACHE
    if _AREA_M2_CACHE is None:
        _AREA_M2_CACHE = np.asarray(compute_triangle_areas(), dtype=float) * 10000.0
    return _AREA_M2_CACHE


def cal_Qflow_raw_cell_m3_s(time):
    """Original OLF flow rate, node-based averaged to cells, m3/s."""
    Q_node = np.asarray(get_required(time, FIELD_KEYS["flow_rate_node"]), dtype=float)
    Q_cell = np.asarray(average_node_field_to_cells(Q_node.tolist()), dtype=float)
    Q_cell = np.where(np.isfinite(Q_cell), Q_cell, 0.0)
    return np.maximum(Q_cell, 0.0)


def cal_exchange_q_cell_m3_s(time):
    """Exchange flux converted from m/s to per-cell discharge m3/s."""
    area_m2 = _cell_area_m2_array()
    exchange_node = np.asarray(get_node_field_or_zeros(time, FIELD_KEYS["exchange_flux_node"]), dtype=float)
    exchange_cell = np.asarray(average_node_field_to_cells(exchange_node.tolist()), dtype=float)
    exchange_cell = np.where(np.isfinite(exchange_cell), exchange_cell, 0.0)
    return exchange_cell * area_m2


def cal_Qflow_cell_m3_s(time):
    """Return the original non-negative OLF discharge in m3/s per cell."""
    Q_raw = cal_Qflow_raw_cell_m3_s(time)
    Q_raw = np.where(np.isfinite(Q_raw), Q_raw, 0.0)
    return np.maximum(Q_raw, 0.0)


def cal_hgs_nutrient_cell_kg_m3(time):
    """Return HGS dissolved nutrient concentration by cell in kg/m3."""
    n_cells = triangle_count()
    try:
        values = np.asarray(
            get_required(time, FIELD_KEYS["nutrient_solute_node"]),
            dtype=float,
        )
    except KeyError:
        log(f"[NUTRIENT] Groundwater_N missing at t={float(time):.0f}; using dissolved solute = 0.")
        return np.zeros(n_cells, dtype=float)

    if values.size == n_cells:
        cell_values = values
    else:
        cell_values = np.asarray(
            average_node_field_to_cells(values.tolist()), dtype=float
        )

    cell_values = np.where(np.isfinite(cell_values), cell_values, 0.0)
    return np.maximum(cell_values, 0.0)


def nutrient_enrichment_ratio(ssc_mgL):
    """Calculate a bounded sediment nutrient enrichment ratio from SSC."""
    sediment_kg_m3 = max(float(ssc_mgL), 0.0) / 1000.0
    er = ER_A * max(sediment_kg_m3, SED_EPS_T_DAY) ** (-ER_B)
    return float(np.clip(er, ER_MIN, ER_MAX))
def cal_Q_s(time, station_mask_cells, chan_mask_cell):
    return calc_station_discharge_m3s(time, station_mask_cells, chan_mask_cell)
def cal_Qflow_cell_m3_day(time):
    """Return per-cell OLF discharge in m3/day."""
    Q_cell_m3_s = np.asarray(cal_Qflow_cell_m3_s(time), dtype=float)
    Q_cell_m3_s = np.where(np.isfinite(Q_cell_m3_s), Q_cell_m3_s, 0.0)
    Q_cell_m3_s = np.maximum(Q_cell_m3_s, 0.0)
    return Q_cell_m3_s * DT_SEC
    


def build_daily_schedule(solution_times, dt=86400.0):
    """Build the daily routing schedule from HGS solution times."""
    times = sorted(solution_times)
    schedule = []

    if len(times) == 0:
        return schedule

    t0 = float(times[0])
    schedule.append((t0, t0))

    for i in range(1, len(times)):
        t_i = float(times[i])
        if i < len(times) - 1:
            t_next = float(times[i+1])
            n_days = int(round((t_next - t_i) / dt))
        else:
            n_days = 1

        if n_days <= 0:
            continue

        for k in range(n_days):
            day_time = t_i + k * dt
            schedule.append((day_time, t_i))

    return schedule
def detect_events_from_series(values, threshold=0.0, min_gap_steps=1, min_event_steps=1):
    vals = np.asarray(values, dtype=float)
    active = np.isfinite(vals) & (vals > threshold)

    raw_events = []
    in_event = False
    start = None

    for i, flag in enumerate(active):
        if flag and not in_event:
            start = i
            in_event = True
        elif (not flag) and in_event:
            raw_events.append((start, i - 1))
            in_event = False

    if in_event:
        raw_events.append((start, len(vals) - 1))

    if not raw_events:
        return []

    merged = [raw_events[0]]

    for s, e in raw_events[1:]:
        ps, pe = merged[-1]
        gap = s - pe - 1

        if gap < min_gap_steps:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    final_events = []
    for s, e in merged:
        if (e - s + 1) >= min_event_steps:
            final_events.append((s, e))

    return final_events


def build_station_event_qpeak_table(df_station,
                                    q_col="q_station_m_s",
                                    Q_col="Q_station_m3_s",
                                    old_qpeak_col="old_qpeak_station_m3_s",
                                    old_qpeak_specific_col="old_qpeak_station_m_s",
                                    threshold=0.0,
                                    min_gap_steps=1,
                                    min_event_steps=1):

    events = detect_events_from_series(
        df_station[q_col].values,
        threshold=threshold,
        min_gap_steps=min_gap_steps,
        min_event_steps=min_event_steps
    )

    rows = []

    for eid, (s, e) in enumerate(events, start=1):

        sub = df_station.iloc[s:e+1].copy()

        i_new_q = sub[q_col].astype(float).idxmax()
        qpeak_new = float(df_station.loc[i_new_q, q_col])
        qpeak_new_time = float(df_station.loc[i_new_q, "time_sec"])

        i_new_Q = sub[Q_col].astype(float).idxmax()
        Qpeak_new = float(df_station.loc[i_new_Q, Q_col])
        Qpeak_new_time = float(df_station.loc[i_new_Q, "time_sec"])

        if old_qpeak_col in df_station.columns:
            i_old_Q = sub[old_qpeak_col].astype(float).idxmax()
            qpeak_old_Q = float(df_station.loc[i_old_Q, old_qpeak_col])
            qpeak_old_Q_time = float(df_station.loc[i_old_Q, "time_sec"])
        else:
            qpeak_old_Q = np.nan
            qpeak_old_Q_time = np.nan

        if old_qpeak_specific_col in df_station.columns:
            i_old_q = sub[old_qpeak_specific_col].astype(float).idxmax()
            qpeak_old_q = float(df_station.loc[i_old_q, old_qpeak_specific_col])
            qpeak_old_q_time = float(df_station.loc[i_old_q, "time_sec"])
        else:
            qpeak_old_q = np.nan
            qpeak_old_q_time = np.nan

        rows.append({
            "event_id": eid,
            "start_idx": int(s),
            "end_idx": int(e),
            "start_time_sec": float(df_station.iloc[s]["time_sec"]),
            "end_time_sec": float(df_station.iloc[e]["time_sec"]),
            "n_steps": int(e - s + 1),

            "qpeak_new_m_s": qpeak_new,
            "qpeak_new_time_sec": qpeak_new_time,

            "Qpeak_new_m3_s": Qpeak_new,
            "Qpeak_new_time_sec": Qpeak_new_time,

            "qpeak_old_m3_s": qpeak_old_Q,
            "qpeak_old_time_sec": qpeak_old_Q_time,

            "qpeak_old_m_s": qpeak_old_q,
            "qpeak_old_specific_time_sec": qpeak_old_q_time
        })

    return pd.DataFrame(rows)
def simulate_time_series_binary():
    """Run distance-based daily sediment routing with per-cell carry-over."""
    import numpy as _np

    global _time_series_cache_binary
    if _time_series_cache_binary is not None:
        return _time_series_cache_binary

    times = list(solution_times)
    if not times:
        return {}

    N = triangle_count()
    L_list = _characteristic_length_per_cell()  # per-cell characteristic length (m)

    # pending queues per cell: list of [remaining_distance_m, mass_t]
    pending = [[] for _ in range(N)]

    day_results = {}

    for d_idx, t in enumerate(times):
        # hydro and production terms
        Q_surf, qch, q_peak, sed_by_ts, sed_all = parameters_cal(t)
        sed_all_arr = _np.array(sed_all, dtype=float)   # t/day per cell (MUSLE)
        v_list = _np.array([float(v) if v is not None else 0.0 for v in qch], dtype=float)  # m/s
        D_list = v_list * 86400.0  # daily travel distance (m)

        # today's source vector after advancing pendings and immediate entries
        src = _np.zeros(N, dtype=float)

        # 1) advance pendings by D; release those reaching channel
        for i in range(N):
            if not pending[i]:
                continue
            D = float(D_list[i])
            if D <= 0.0:
                continue  # cannot advance today
            new_queue = []
            for rem_dist, mass in pending[i]:
                rem_new = rem_dist - D
                if rem_new <= 0.0:
                    src[i] += mass  # reaches channel today
                else:
                    new_queue.append([rem_new, mass])
            pending[i] = new_queue

        # 2) handle today's new production with binary rule
        for i in range(N):
            L = float(L_list[i])
            D = float(D_list[i])
            m = float(sed_all_arr[i])
            if m <= 0.0:
                continue
            if D >= L - 1e-9:
                # can reach today
                src[i] += m
            else:
                # queue with remaining distance
                rem = max(0.0, L - D)
                pending[i].append([rem, m])

        # 3) route in-channel using existing solver (applies SDR internally)
        _ , routed = sediment_transport_variableSDR(t, src)
        day_results[t] = _np.asarray(routed, dtype=float)

    _time_series_cache_binary = day_results
    return day_results


def simulate_daily_loop():
    """Run sediment routing sequentially with retained sediment carried forward."""

    delivered_series = {}
    retained_series = {}

    N = triangle_count()
    carry = np.zeros(N, dtype=float)

    for idx, t in enumerate(solution_times):
        log(f"[DAILY LOOP] Processing time={t} index={idx}")

        # 1. New sediment generated at this timestep
        _, _, _, _, sediment_yield_all = parameters_cal(t)
        production = np.asarray(sediment_yield_all, dtype=float)
        production = np.where(np.isfinite(production), production, 0.0)
        production = np.maximum(production, 0.0)

        # 2. Available sediment = new production + retained storage from previous timestep
        src = production + carry

        # 3. Route sediment
        retained, delivered = sediment_transport_variableSDR(t, src)

        retained = np.asarray(retained, dtype=float)
        delivered = np.asarray(delivered, dtype=float)

        retained = np.where(np.isfinite(retained), retained, 0.0)
        delivered = np.where(np.isfinite(delivered), delivered, 0.0)

        retained = np.maximum(retained, 0.0)
        delivered = np.maximum(delivered, 0.0)

        # 4. Save outputs
        delivered_series[t] = delivered.copy()
        retained_series[t] = retained.copy()

        # 5. Carry a fraction of retained sediment forward to smooth/delay event response.
        carry_decay = float(np.clip(HILLSLOPE_CARRY_DECAY, 0.0, 1.0))
        carry = retained.copy() * carry_decay

        log(
            f"[DAILY LOOP] time={t}, "
            f"production_sum={np.sum(production):.3e}, "
            f"delivered_sum={np.sum(delivered):.3e}, "
            f"retained_sum={np.sum(retained):.3e}, "
            f"carry_sum={np.sum(carry):.3e}"
        )

    return retained_series, delivered_series



def open_file():
    return filedialog.askopenfilename(filetypes=[('dat files', '*.dat'), ('All files', '*.*')])
# OLF and CHAN file readers
def read_olf_variables(file_path):
    data_by_time = {}
    current_time = None
    current_section = None

    zone_header_re = re.compile(
        r'''^\s*ZONE\b.*?T\s*=\s*"[^"]*?"\s*,\s*SOLUTIONTIME\s*=\s*(?P<time>[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)''',
        re.IGNORECASE | re.VERBOSE
    )

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = zone_header_re.match(line)
            if m:
                current_time = float(m.group('time'))
                data_by_time[current_time] = {}
                current_section = None
                continue

            if current_time is None:
                continue

            stripped = line.lstrip()
            if stripped.startswith("#"):
                sec = stripped[1:].strip()
                data_by_time[current_time][sec] = []
                current_section = sec
                continue

            if current_section is not None:
                nums = re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+", line)
                if nums:
                    data_by_time[current_time][current_section].extend([float(x) for x in nums])

    return data_by_time
def read_chan_channel_zones(chan_path):

    def _next_nonempty_line(f):
        while True:
            line = f.readline()
            if not line:
                return None
            s = line.strip()
            if s == "" or s.startswith("#"):
                continue
            if s.upper().startswith("DT="):
                continue
            return line

    def _skip_n_floats(f, n):
        # Skip n numeric tokens in Tecplot BLOCK format.
        k = 0
        while k < n:
            line = _next_nonempty_line(f)
            if line is None:
                raise EOFError("Unexpected EOF while skipping BLOCK floats in CHAN file.")
            parts = line.strip().split()
            k += len(parts)

    def _read_n_floats(f, n):
        vals = np.empty(n, dtype=float)
        k = 0
        while k < n:
            line = _next_nonempty_line(f)
            if line is None:
                raise EOFError("Unexpected EOF while reading BLOCK floats in CHAN file.")
            parts = line.strip().split()
            m = min(len(parts), n - k)
            vals[k:k+m] = np.array(parts[:m], dtype=float)
            k += m
        return vals

    with open(chan_path, "r", encoding="utf-8", errors="replace") as f:
        # Locate VARIABLES.
        varnames = None
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("VARIABLES"):
                varnames = re.findall(r'"([^"]+)"', line)
                break
        if not varnames:
            raise ValueError(f"Cannot find VARIABLES in chan file: {chan_path}")

        # Locate ZONE.
        zone_line = None
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("ZONE"):
                zone_line = line
                break
        if zone_line is None:
            raise ValueError(f"Cannot find ZONE in chan file: {chan_path}")

        # Parse NODES, ELEMENTS, and VARLOCATION.
        mn = re.search(r"NODES=(\d+)", zone_line)
        me = re.search(r"ELEMENTS=(\d+)", zone_line)
        if not (mn and me):
            raise ValueError(f"Bad ZONE header (missing NODES/ELEMENTS): {zone_line}")

        nn = int(mn.group(1))
        ne = int(me.group(1))

        # Tecplot variable ids are 1-based.
        cellcentered_ids = set()
        mvl = re.search(r"VARLOCATION=\(\[([0-9,\s]+)\]=CELLCENTERED\)", zone_line)
        if mvl:
            ids = [int(x.strip()) for x in mvl.group(1).split(",") if x.strip()]
            cellcentered_ids = set(ids)

        # Locate the Zone variable.
        if "Zone" not in varnames:
            raise ValueError(f'CHAN VARIABLES does not include "Zone". Vars={varnames}')
        zone_vid = varnames.index("Zone") + 1  # 1-based

        # Read the Zone block.
        for vid, vname in enumerate(varnames, start=1):
            nread = ne if vid in cellcentered_ids else nn
            if vid < zone_vid:
                _skip_n_floats(f, nread)
            elif vid == zone_vid:
                z = _read_n_floats(f, nread)
                zone_ids = set(int(round(v)) for v in z if np.isfinite(v))
                return zone_ids

    raise RuntimeError("Failed to read Zone block from CHAN file.")
def read_chan_element_centroids(chan_path):
    """Read CHAN geometry and return element centroids."""

    def _next_nonempty_line(f):
        while True:
            line = f.readline()
            if not line:
                return None
            s = line.strip()
            if s == "" or s.startswith("#"):
                continue
            return line

    def _read_n_floats(f, n):
        vals = []
        while len(vals) < n:
            line = _next_nonempty_line(f)
            if line is None:
                raise EOFError("Unexpected EOF while reading CHAN numeric block.")

            parts = line.strip().split()
            try:
                vals.extend([float(x) for x in parts])
            except ValueError:
                # skip non-numeric Tecplot metadata lines
                continue

        return np.asarray(vals[:n], dtype=float)

    def _read_connectivity(f, ne):
        """Read Tecplot FE line connectivity."""
        conn = []
        while len(conn) < ne:
            line = _next_nonempty_line(f)
            if line is None:
                raise EOFError("Unexpected EOF while reading CHAN connectivity.")
            parts = line.strip().split()
            if len(parts) >= 2:
                conn.append((int(parts[0]) - 1, int(parts[1]) - 1))
        return conn

    with open(chan_path, "r", encoding="utf-8", errors="replace") as f:
        # ---- 1) Find VARIABLES ----
        varnames = None
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip().startswith("VARIABLES"):
                varnames = re.findall(r'"([^"]+)"', line)
                break

        if not varnames:
            raise ValueError(f"Cannot find VARIABLES line in CHAN file: {chan_path}")

        # ---- 2) Find ZONE ----
        zone_line = None
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip().startswith("ZONE"):
                zone_line = line.strip()
                break

        if zone_line is None:
            raise ValueError(f"Cannot find ZONE line in CHAN file: {chan_path}")

        mn = re.search(r"NODES=(\d+)", zone_line)
        me = re.search(r"ELEMENTS=(\d+)", zone_line)
        if not (mn and me):
            raise ValueError(f"Cannot parse NODES/ELEMENTS from CHAN ZONE header: {zone_line}")

        nn = int(mn.group(1))
        ne = int(me.group(1))

        log(f"[CHAN] NODES={nn}, ELEMENTS={ne}")

        # ---- 3) VARLOCATION: determine which variables are cell-centered ----
        cellcentered_ids = set()
        mvl = re.search(r"VARLOCATION=\(\[([0-9,\s]+)\]=CELLCENTERED\)", zone_line)
        if mvl:
            ids = [int(x.strip()) for x in mvl.group(1).split(",") if x.strip()]
            cellcentered_ids = set(ids)

        # ---- 4) Read BLOCK variables ----
        var_data = {}

        for vid, vname in enumerate(varnames, start=1):
            nread = ne if vid in cellcentered_ids else nn
            arr = _read_n_floats(f, nread)
            var_data[vname] = arr

        # ---- 5) Read connectivity ----
        conn = _read_connectivity(f, ne)

    # ---- 6) Find x/y variable names ----
    x_key = None
    y_key = None

    for k in var_data.keys():
        if k.strip().lower() in ["x", "node x"]:
            x_key = k
        if k.strip().lower() in ["y", "node y"]:
            y_key = k

    if x_key is None or y_key is None:
        raise KeyError(f"Cannot find x/y in CHAN variables. Available variables: {list(var_data.keys())}")

    x = var_data[x_key]
    y = var_data[y_key]

    centroids = np.zeros((ne, 2), dtype=float)

    for i, (n1, n2) in enumerate(conn):
        centroids[i, 0] = 0.5 * (x[n1] + x[n2])
        centroids[i, 1] = 0.5 * (y[n1] + y[n2])

    return centroids
def build_chan_mask_by_centroid_mapping(chan_path, max_dist_m=50.0):
    """Map CHAN element centroids to nearest OLF surface cells."""

    from scipy.spatial import cKDTree

    N = triangle_count()

    # OLF surface mesh centroids
    olf_centroids = np.asarray([triangle_centroid(i) for i in range(N)], dtype=float)

    # CHAN element centroids
    chan_centroids = read_chan_element_centroids(chan_path)

    tree = cKDTree(olf_centroids)
    dist, idx = tree.query(chan_centroids, k=1)

    valid = np.isfinite(dist) & (dist <= max_dist_m)

    chan_mask = np.zeros(N, dtype=bool)
    chan_mask[idx[valid]] = True

    log(
        f"[INIT] CHAN centroid mapping: "
        f"CHAN elements={len(chan_centroids)}, "
        f"mapped={int(valid.sum())}, "
        f"unique OLF cells={int(chan_mask.sum())}, "
        f"max_dist_m={max_dist_m}"
    )

    if valid.sum() == 0:
        log("[WARN] No CHAN elements mapped to OLF cells. Try increasing max_dist_m.")

    log(
        f"[INIT] CHAN mapping distance: "
        f"min={float(np.nanmin(dist)):.3f} m, "
        f"median={float(np.nanmedian(dist)):.3f} m, "
        f"max={float(np.nanmax(dist)):.3f} m"
    )
    valid = np.isfinite(dist) & (dist <= max_dist_m)

    chan_mask = np.zeros(N, dtype=bool)
    chan_mask[idx[valid]] = True

    log(
        f"[INIT] CHAN centroid mapping: "
        f"CHAN elements={len(chan_centroids)}, "
        f"mapped={int(valid.sum())}, "
        f"unique OLF cells={int(chan_mask.sum())}, "
        f"max_dist_m={max_dist_m}"
    )
    return chan_mask
def has_any_key(dct, keys):
    return any(k in dct for k in keys)

def find_mesh_time():
    for t in solution_times:
        keys = olf_data.get(t, {})
        if has_any_key(keys, MESH_KEYS["node_x"]) and \
           has_any_key(keys, MESH_KEYS["node_y"]) and \
           has_any_key(keys, MESH_KEYS["elem_nodes"]):
            return t
    raise KeyError("cant find x/y/element node lists--MESH_KEYS")

def get_required(t_sec, key_or_list, *, allow_mesh_fallback=False):
    """Return a required OLF field by exact key or candidate-key list."""
    # ---- pick the dict for this time ----
    data = olf_data.get(t_sec, None)
    if data is None:
        if allow_mesh_fallback:
            data = olf_data.get(MESH_TIME, None)
        if data is None:
            raise KeyError(f"Time {t_sec} not found in olf_data (and mesh fallback failed).")

    # ---- normalize candidates ----
    if isinstance(key_or_list, (list, tuple)):
        candidates = list(key_or_list)
    else:
        candidates = [key_or_list]

    # ---- try at this time ----
    for k in candidates:
        if k in data:
            return data[k]

    # ---- optional mesh fallback if not already using it ----
    if allow_mesh_fallback and t_sec != MESH_TIME:
        data_m = olf_data.get(MESH_TIME, {})
        for k in candidates:
            if k in data_m:
                return data_m[k]

    # ---- not found ----
    sample_keys = list(data.keys())[:20]
    raise KeyError(
        f"Missing keys {candidates} at time {t_sec}. "
        f"Available keys sample: {sample_keys}"
    )

def get_flow_rate_node(time):
    """Return OLF flow rate, falling back to velocity magnitude if needed."""
    # Try real flow rate from OLF
    try:
        return get_required(time, FIELD_KEYS["flow_rate_node"])
    except KeyError:
        pass

    # Fallback: compute from velocities
    vx = np.array(get_required(time, FIELD_KEYS["vx_cell"]), dtype=float)
    vy = np.array(get_required(time, FIELD_KEYS["vy_cell"]), dtype=float)

    # vz may or may not exist
    try:
        vz = np.array(get_required(time, FIELD_KEYS["node_z"]), dtype=float)
    except KeyError:
        vz = None

    if vz is None:
        flow_mag = np.sqrt(vx**2 + vy**2)
    else:
        flow_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    return flow_mag.tolist()

def get_node_numbers():
    node_numbers = get_required(MESH_TIME, MESH_KEYS["elem_nodes"], allow_mesh_fallback=False)
    if len(node_numbers) % 4 == 0:
        return node_numbers, 4
    if len(node_numbers) % 3 == 0:
        return node_numbers, 3
    raise ValueError("not 3/4 long in node number")

def triangle_count():
    node_numbers, stride = get_node_numbers()
    return len(node_numbers) // stride
# x,y,z in unit m
def prepare_global_coords():
    global global_coords, global_triangles
    x_coords = get_required(MESH_TIME, MESH_KEYS["node_x"], allow_mesh_fallback=False)
    y_coords = get_required(MESH_TIME, MESH_KEYS["node_y"], allow_mesh_fallback=False)
    z_coords = get_required(MESH_TIME, MESH_KEYS["node_z"], allow_mesh_fallback=False)
    node_numbers, stride = get_node_numbers()
    ntri = len(node_numbers) // stride

    x1 = [0.0]*ntri; y1 = [0.0]*ntri
    x2 = [0.0]*ntri; y2 = [0.0]*ntri
    x3 = [0.0]*ntri; y3 = [0.0]*ntri

    for i in range(ntri):
        base = i*stride
        n1 = int(node_numbers[base+0]) - 1
        n2 = int(node_numbers[base+1]) - 1
        n3 = int(node_numbers[base+2]) - 1
        x1[i], y1[i] = x_coords[n1], y_coords[n1]
        x2[i], y2[i] = x_coords[n2], y_coords[n2]
        x3[i], y3[i] = x_coords[n3], y_coords[n3]

    global_coords = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'x3': x3, 'y3': y3, 'z':z_coords}
    global_triangles = [(x1[i], y1[i], x2[i], y2[i], x3[i], y3[i]) for i in range(ntri)]
### All coordinates are in unit m
def triangle_centroid(idx):
    x1 = global_coords['x1'][idx]; y1 = global_coords['y1'][idx]
    x2 = global_coords['x2'][idx]; y2 = global_coords['y2'][idx]
    x3 = global_coords['x3'][idx]; y3 = global_coords['y3'][idx]
    return (x1 + x2 + x3)/3.0, (y1 + y2 + y3)/3.0

def prepare_neighbors():
    global global_neighbors
    node_numbers, stride = get_node_numbers()
    ntri = len(node_numbers)//stride
    neighbors = {i: [] for i in range(ntri)}
    edge_map = {}

    for i in range(ntri):
        base = i*stride
        n1 = int(node_numbers[base+0]) - 1
        n2 = int(node_numbers[base+1]) - 1
        n3 = int(node_numbers[base+2]) - 1
        edges = [(n1,n2), (n2,n3), (n3,n1)]
        for a,b in edges:
            e = (a,b) if a < b else (b,a)
            edge_map.setdefault(e, []).append(i)

    for tris in edge_map.values():
        if len(tris) == 2:
            a,b = tris
            neighbors[a].append(b)
            neighbors[b].append(a)

    global_neighbors = neighbors
####UNIT ha, convert to ha and prepared to be used in MUSLE cal
def compute_triangle_areas():
    x1 = global_coords['x1']; y1 = global_coords['y1']
    x2 = global_coords['x2']; y2 = global_coords['y2']
    x3 = global_coords['x3']; y3 = global_coords['y3']
    n = len(x1)
    areas = [0.0]*n
    for i in range(n):
        a_x = x2[i] - x1[i]; a_y = y2[i] - y1[i]
        b_x = x3[i] - x1[i]; b_y = y3[i] - y1[i]
        areas[i] = abs(a_x*b_y - a_y*b_x) * 0.5
    areas_ha = [a / 10000 for a in areas]
    return areas_ha

def average_node_field_to_cells(node_field):
    node_numbers, stride = get_node_numbers()
    ntri = len(node_numbers) // stride
    cell_vals = [0.0]*ntri
    for i in range(ntri):
        base = i*stride
        n1 = int(node_numbers[base+0]) - 1
        n2 = int(node_numbers[base+1]) - 1
        n3 = int(node_numbers[base+2]) - 1
        cell_vals[i] = (node_field[n1] + node_field[n2] + node_field[n3]) / 3.0
    return cell_vals

def calculate_KUSLE(SC, cch):
    tau_c = (0.1 + 0.1779*SC + 0.0028*SC**2 - 2.34e-5*SC**3)*cch
    kd = 0.2 * (tau_c**-0.5) if tau_c > 0 else 0.0
    return kd

def flow_mag(cell_idx, time):
    vx = get_required(time, "x linear velocity (cell centred)")
    vy = get_required(time, "y linear velocity (cell centred)")
    return float(np.hypot(vx[cell_idx], vy[cell_idx]))

def flow_dir_deg(cell_idx, time):
    """Return flow direction in degrees."""
    vx = get_required(time, "x linear velocity (cell centred)")
    vy = get_required(time, "y linear velocity (cell centred)")
    ang = math.degrees(math.atan2(vy[cell_idx], vx[cell_idx]))
    if ang < 0: ang += 360.0
    return ang
#   UNIT    m
def cal_depth(time):
    Depth_s = get_required(time, FIELD_KEYS["depth_node"])
    node_numbers, stride = get_node_numbers()
    ntri = len(node_numbers)//stride
    Depth_cell = [0.0]*ntri
    for i in range(ntri):
        base = i*stride
        n1 = int(node_numbers[base+0]) - 1
        n2 = int(node_numbers[base+1]) - 1
        n3 = int(node_numbers[base+2]) - 1
        Depth_cell[i] = (Depth_s[n1] + Depth_s[n2] + Depth_s[n3]) / 3.0
    return Depth_cell

def zoneN(time):
    wl = cal_depth(time)   
    #print("length:",len(wl))
    zone = []
    for d in wl:
        if d > 0.001:       
            zone.append(1)
        else:
            zone.append(0)
    return zone


def cal_EF(t):
    Exchange_flux_node = get_node_field_or_zeros(t, "exchange_flux_node")  # node array
    node_numbers, stride = get_node_numbers()
    ntri = len(node_numbers)//stride
    Exchange_flux_cell = [0.0]*ntri
    for i in range(ntri):
        base = i*stride
        n1 = int(node_numbers[base+0]) - 1
        n2 = int(node_numbers[base+1]) - 1
        n3 = int(node_numbers[base+2]) - 1
        Exchange_flux_cell[i] = (Exchange_flux_node[n1] + Exchange_flux_node[n2] + Exchange_flux_node[n3]) / 3.0
    return Exchange_flux_cell



def cal_Qproxy_m3_day(time, areas_m2):
    """Build a per-cell discharge proxy from velocity, depth, and cell area."""
    vx = np.asarray(get_required(time, "x linear velocity (cell centred)"), dtype=float)
    vy = np.asarray(get_required(time, "y linear velocity (cell centred)"), dtype=float)
    vmag = np.hypot(vx, vy)  # m/s

    depth = np.asarray(cal_depth(time), dtype=float)  # m

    A = np.asarray(areas_m2, dtype=float)             # m2
    L = np.sqrt(np.maximum(A, 0.0))                   # m

    Q_m3s = vmag * depth * L
    Q_m3s = np.where(np.isfinite(Q_m3s), Q_m3s, 0.0)
    Q_m3s = np.maximum(Q_m3s, 0.0)

    Q_m3_day = Q_m3s * DT_SEC
    return Q_m3_day
def get_node_count():
    x_nodes = get_required(MESH_TIME, MESH_KEYS["node_x"], allow_mesh_fallback=False)
    return len(x_nodes)

def get_node_field_or_zeros(t_sec, field_key_name_or_list, *, allow_mesh_fallback=True):
    """Return an OLF field as an array, or a zero array if the field is missing."""
    import numpy as np

    if isinstance(field_key_name_or_list, (list, tuple)):
        candidates = list(field_key_name_or_list)
    else:
        candidates = FIELD_KEYS.get(field_key_name_or_list, field_key_name_or_list)

    try:
        arr = np.asarray(get_required(t_sec, candidates, allow_mesh_fallback=allow_mesh_fallback), dtype=float)
        return arr
    except Exception:
        data_t = olf_data.get(t_sec, {})
        ref = None

        for ref_key in (
            "depth_node", "depth", "Depth", "water_depth",
            "q_node", "Q", "discharge", "Precipitation", "precipitation"
        ):
            if ref_key in data_t:
                ref = data_t[ref_key]
                break

        if ref is None and isinstance(data_t, dict) and data_t:
            for v in data_t.values():
                try:
                    vv = np.asarray(v, dtype=float)
                    if vv.ndim == 1 and vv.size > 0:
                        ref = vv
                        break
                except Exception:
                    continue

        if ref is None:
            data_m = olf_data.get(MESH_TIME, {}) if "MESH_TIME" in globals() else {}
            for v in (data_m.values() if isinstance(data_m, dict) else []):
                try:
                    vv = np.asarray(v, dtype=float)
                    if vv.ndim == 1 and vv.size > 0:
                        ref = vv
                        break
                except Exception:
                    continue

        if ref is None:
            return np.zeros((0,), dtype=float)

        ref = np.asarray(ref, dtype=float)
        return np.zeros((ref.size,), dtype=float)

def get_rain_node(t_sec):
    try:
        return get_node_field_or_zeros(t_sec, FIELD_KEYS["Precipitation"])
    except Exception:
        return np.zeros(get_node_count(), dtype=float)



def cal_Precipitation(time):
    rain_node = get_rain_node(time)
    node_numbers, stride = get_node_numbers()
    ntri = len(node_numbers) // stride
    Rain_cell = [0.0] * ntri

    for i in range(ntri):
        base = i * stride
        n1 = int(node_numbers[base + 0]) - 1
        n2 = int(node_numbers[base + 1]) - 1
        n3 = int(node_numbers[base + 2]) - 1
        Rain_cell[i] = (rain_node[n1] + rain_node[n2] + rain_node[n3]) / 3.0

    return Rain_cell

def build_dynamic_channel_mask_for_time(time):
    """Return the timestep-specific channel mask for sediment routing."""
    if CHAN_MASK is None:
        if CHANNEL_MASK is not None:
            return np.asarray(CHANNEL_MASK, dtype=bool).copy()
        return np.asarray(zoneN(time), dtype=bool)

    base_mask = np.asarray(CHAN_MASK, dtype=bool).copy()
    if (not USE_DYNAMIC_CHANNEL_MASK) or CHANNEL_NEIGHBOR_RINGS <= 0:
        return base_mask

    depth = np.asarray(cal_depth(time), dtype=float)
    depth = np.where(np.isfinite(depth), depth, 0.0)
    neighbor_depth_threshold = max(float(depth_threshold), float(CHANNEL_NEIGHBOR_DEPTH_THRESHOLD))
    wet_neighbor_mask = depth > neighbor_depth_threshold

    channel_mask = base_mask.copy()
    frontier = set(int(i) for i in np.where(base_mask)[0])

    for _ in range(max(0, int(CHANNEL_NEIGHBOR_RINGS))):
        next_frontier = set()
        for i in frontier:
            for j in global_neighbors.get(i, []):
                j = int(j)
                if channel_mask[j]:
                    continue
                if wet_neighbor_mask[j]:
                    channel_mask[j] = True
                    next_frontier.add(j)
        if not next_frontier:
            break
        frontier = next_frontier

    return channel_mask

def calculate_CUSLE(Acrop, Afallow, C_USLE=None):
    if C_USLE is not None:
        return float(C_USLE)
    return (Acrop / Afallow) if Afallow != 0 else 0.0

def calculate_CFRG(rock):
    return max(0.0, 1.0 - rock)

def compute_cell_slope_and_LS(time):
    """Compute per-cell slope and LS factor from mesh elevations."""
    z_nodes = get_required(MESH_TIME, MESH_KEYS["node_z"], allow_mesh_fallback=False)  # meters
    x_nodes = get_required(MESH_TIME, MESH_KEYS["node_x"], allow_mesh_fallback=False)
    y_nodes = get_required(MESH_TIME, MESH_KEYS["node_y"], allow_mesh_fallback=False)
    node_numbers, stride = get_node_numbers()
    ntri = len(node_numbers)//stride

    slope_sin = [0.0]*ntri
    LS_list   = [1.14]*ntri  # fallback

    def _clip(v, lo, hi):
        return max(lo, min(hi, v))

    areas_ha = compute_triangle_areas()
    areas_m2 = [a*10000.0 for a in areas_ha]
    lam = [max(1e-3, math.sqrt(a)) for a in areas_m2]  # meters

    for i in range(ntri):
        base = i*stride
        n1 = int(node_numbers[base+0]) - 1
        n2 = int(node_numbers[base+1]) - 1
        n3 = int(node_numbers[base+2]) - 1

        x1,y1,z1 = x_nodes[n1], y_nodes[n1], z_nodes[n1]
        x2,y2,z2 = x_nodes[n2], y_nodes[n2], z_nodes[n2]
        x3,y3,z3 = x_nodes[n3], y_nodes[n3], z_nodes[n3]

        # plane normal from triangle
        ux, uy, uz = (x2-x1), (y2-y1), (z2-z1)
        vx, vy, vz = (x3-x1), (y3-y1), (z3-z1)
        nx = uy*vz - uz*vy
        ny = uz*vx - ux*vz
        nz = ux*vy - uy*vx

        denom = math.sqrt(nx*nx + ny*ny + nz*nz)
        if denom <= 0.0 or abs(nz) < 1e-12:
            theta = 0.0
        else:
            # tan(theta) = sqrt(nx^2+ny^2)/|nz|
            tan_theta = math.sqrt(nx*nx + ny*ny) / abs(nz)
            theta = math.atan(max(0.0, tan_theta))

        s = math.sin(theta)                    # needed by SWAT S-factor
        slope_sin[i] = s

        # SWAT-style slope-length factor.
        s_cl = _clip(s, 0.0, 0.9999)
        denom_beta = 3.0*(s_cl**0.8) + 0.56
        beta = (s_cl/0.0896)/denom_beta if denom_beta > 0 else 0.0
        m = beta/(1.0 + beta) if beta > 0 else 0.0
        L_fac = (lam[i]/22.13)**m
        if s_cl < math.sin(math.atan(0.09)):  # slope < 9%
            S_fac = 10.8*s_cl + 0.03
        else:
            S_fac = 16.8*s_cl - 0.50
        LS_list[i] = max(0.1, L_fac * S_fac)  # floor to avoid 0

    return slope_sin, LS_list

# MUSLE: convert inputs to required units inside the function
# Q_surf (m^3/day)  Q_mm_per_ha (mm/ha/day) using 1 mmha = 10 m^3
# q_peak stays in m^3/s; area_hru in ha
def _as_seq(val, n):
    if isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    return [val]*n

def calculate_sediment_yield(Q_surf, q_peak, area_hru, K_USLE, C_USLE, P_USLE, L_USLE, CFRG):
    out = []
    n = len(area_hru)
    K_seq   = _as_seq(K_USLE, n)
    C_seq   = _as_seq(C_USLE, n)
    P_seq   = _as_seq(P_USLE, n)
    L_seq   = _as_seq(L_USLE, n)
    CFRG_seq= _as_seq(CFRG,  n)
    eps = 1e-20
    for Q_m3_day, qp, a_ha, k, c, p, l, cfrg in zip(Q_surf, q_peak, area_hru, K_seq, C_seq, P_seq, L_seq, CFRG_seq):
        a_ha = float(max(a_ha, eps))
        Q_mm_per_ha = float(Q_m3_day) / (10.0 * a_ha)   # mm/ha/day
        KCP = float(k) * float(c) * float(p) * float(l) * float(cfrg)
        val = max(Q_mm_per_ha * float(qp) * a_ha, eps)  # V(mm/ham3)q_peakarea
        sed = 11.8 * (val ** 0.56) * KCP                # t/day for the HRU
        out.append(sed)
    return out

def apply_event_saturation(ssc_event_mgL):
    """Apply a smooth saturation cap to event SSC."""
    x = max(float(ssc_event_mgL), 0.0)
    if (not USE_EVENT_SATURATION) or x <= 0.0:
        return x

    cap = max(float(EVENT_SATURATION_CAP_MGL), 1e-12)
    gamma = max(float(EVENT_SATURATION_GAMMA), 1e-12)
    ratio_g = (x / cap) ** gamma
    return cap * ratio_g / (1.0 + ratio_g)

def calculate_qpeak_manning(depth_cell, area_hru_m2, slope_sin, n_manning, factor=1.0):
    """Estimate q_peak from Manning flow with simplified cell geometry."""
    depth = np.asarray(depth_cell, dtype=float)
    area_m2 = np.asarray(area_hru_m2, dtype=float)
    slope = np.asarray(slope_sin, dtype=float)

    width = np.sqrt(np.maximum(area_m2, 1e-12))
    flow_area = np.maximum(depth, 0.0) * width
    wetted_perimeter = width + 2.0 * np.maximum(depth, 0.0)
    hydraulic_radius = np.divide(
        flow_area,
        np.maximum(wetted_perimeter, 1e-12)
    )

    slope_eff = np.sqrt(np.maximum(slope, 1e-8))
    n_eff = max(float(n_manning), 1e-6)
    q_m3_s = (1.0 / n_eff) * flow_area * (hydraulic_radius ** (2.0 / 3.0)) * slope_eff
    q_m3_s = float(factor) * np.where(np.isfinite(q_m3_s), q_m3_s, 0.0)
    return np.maximum(q_m3_s, 0.0).tolist()

def parameters_cal(time):
    zone = zoneN(time)

    vx = get_required(time, "x linear velocity (cell centred)")
    vy = get_required(time, "y linear velocity (cell centred)")
    qch = [float(np.hypot(v_x, v_y)) for v_x, v_y in zip(vx, vy)]

    Q_cell = cal_Qflow_cell_m3_s(time)                    # m^3/s
    Depth_cell = cal_depth(time)  
    if USE_GEOM_AREA:
        area_hru_m2 = [a*10000.0 for a in compute_triangle_areas()]  # ha -> m
    else:
        area_hru_m2 = [1.0]*len(qch)
    b_char = [max(1e-3, (A**0.5)) for A in area_hru_m2]

    Q_surf = [q * DT_SEC for q in Q_cell]  #  m^3/day
    flux_mps_node = get_node_field_or_zeros(time, FIELD_KEYS["Precipitation"])              # m/s (node), safe
    flux_mps_cell = average_node_field_to_cells(flux_mps_node.tolist())          # m/s (cell)
    i_mmhr = [f * 3600.0 * 1000.0 for f in flux_mps_cell]                        # mm/hr

    if USE_GEOM_AREA:
        area_ha = compute_triangle_areas()                             # ha
    else:
        area_ha = [1.0]*len(i_mmhr)
    A_km2 = [a / 100.0 for a in area_ha]                               # km^2

    slope_sin, LS_list = compute_cell_slope_and_LS(time)

    if QPEAK_METHOD == "manning":
        q_peak = calculate_qpeak_manning(
            Depth_cell,
            area_hru_m2,
            slope_sin,
            MANNING_N,
            factor=prf * QPEAK_MANNING_FACTOR
        )
    elif QPEAK_METHOD == "flowrate":
        Q_arr = np.asarray(Q_cell, dtype=float)
        q_peak = np.where(np.isfinite(Q_arr), Q_arr, 0.0)
        q_peak = np.maximum(q_peak, 0.0).tolist()
    else:
        raise ValueError(f"Unknown qpeak_method={QPEAK_METHOD!r}. Use 'manning' or 'flowrate'.")

    K_loc = calculate_KUSLE(SC, cch)
    C_val  = calculate_CUSLE(Acrop, Afallow, C_USLE)
    CFRG   = calculate_CFRG(rock)

    sediment_yield_all = calculate_sediment_yield(
        Q_surf, q_peak, area_ha,
        K_loc, C_val, P_USLE, LS_list, CFRG
    )
    sed_by_timesteps   = [se for se in sediment_yield_all]
    if time == solution_times[0]:
        log(f"[DIAG] i_mmhr min/max = {min(i_mmhr):.3e}, {max(i_mmhr):.3e}")
        log(f"[DIAG] qpeak_method = {QPEAK_METHOD}, manning_n={MANNING_N}, qpeak_manning_factor={QPEAK_MANNING_FACTOR}")
        log(f"[DIAG] q_peak m3/s min/max = {min(q_peak):.3e}, {max(q_peak):.3e}")
        log(f"[DIAG] Q_surf m3/day min/max = {min(Q_surf):.3e}, {max(Q_surf):.3e}")
        log(f"[DIAG] area_ha min/max = {min(area_ha):.3e}, {max(area_ha):.3e}")

    return Q_surf, qch, q_peak, sed_by_timesteps, sediment_yield_all

def specified_flux_node_to_cell(time, *, to_day=True, cm_unit=False):
    I_node = get_node_field_or_zeros(time, FIELD_KEYS["Precipitation"]).tolist()
    scale = 1.0
    if to_day:
        scale *= DT_SEC
    if cm_unit:
        scale *= 100.0
    if scale != 1.0:
        I_node = [i * scale for i in I_node]
    I_cell = average_node_field_to_cells(I_node)
    return I_cell

def cal_SDR(time):
    """Calculate the hybrid sediment delivery ratio for each cell."""

    global CHAN_MASK, CHANNEL_MASK

    Q_day = np.asarray(cal_Qflow_cell_m3_day(time), dtype=float)
    depth = np.asarray(cal_depth(time), dtype=float)

    # rainfall / specified flux
    I_cell = np.asarray(
        specified_flux_node_to_cell(time, to_day=True, cm_unit=False),
        dtype=float
    )

    N = len(Q_day)

    chan_mask = np.asarray(CHAN_MASK, dtype=bool)
    active_mask = build_dynamic_channel_mask_for_time(time)

    SDR = np.zeros(N, dtype=float)

    # 1. True fixed CHAN cells: full delivery
    SDR[chan_mask] = 1.0

    # 2. Active non-channel cells
    overland_active = active_mask & (~chan_mask) & (Q_day > 0.0) & (depth > depth_threshold)

    if np.any(overland_active):
        Q_ref = np.nanpercentile(Q_day[overland_active], 90)
        Q_ref = max(Q_ref, 1e-12)

        Q_norm = Q_day / Q_ref

        # rainfall factor: rainier cells get stronger delivery,
        # but no-rain cells are not forced to zero if Q exists
        I_active = I_cell[overland_active]
        I_ref = np.nanpercentile(I_active[I_active > 0], 90) if np.any(I_active > 0) else 1.0
        I_ref = max(I_ref, 1e-12)

        I_norm = I_cell / I_ref

        # combined delivery:
        # Q controls hydrologic connectivity, I enhances rainfall-driven mobilization
        SDR_raw = a * (Q_norm ** b) * (1.0 + 0.5 * I_norm)

        SDR[overland_active] = SDR_raw[overland_active]

    SDR = np.clip(SDR, 0.0, 1.0)

    return SDR.tolist()

def compute_T(time):
    vx = np.array(get_required(time, FIELD_KEYS["vx_cell"]), dtype=float)
    vy = np.array(get_required(time, FIELD_KEYS["vy_cell"]), dtype=float)

    N = len(global_coords['x1'])
    centroids = [triangle_centroid(i) for i in range(N)]

    # T will be a list of dicts: T[i] = {j: weight_ij, ...}
    T_rows = [dict() for _ in range(N)]

    for i in range(N):
        v = np.array([vx[i], vy[i]], dtype=float)
        nv = np.linalg.norm(v)
        if nv == 0.0:
            continue

        weights = {}
        for j in global_neighbors.get(i, []):
            if j == i:
                continue
            d = np.array(centroids[j]) - np.array(centroids[i])
            nd = np.linalg.norm(d)
            if nd == 0.0:
                continue
            cos_theta = float(np.dot(v, d) / (nv * nd))
            if cos_theta > 0.0:
                weights[j] = cos_theta

        s = sum(weights.values())
        if s > 0.0:
            inv_s = 1.0 / s
            for j, w in weights.items():
                T_rows[i][j] = w * inv_s   

    return T_rows

def compute_T_channel(time, channel_idx):
    """Build downstream routing weights among channel cells."""

    T_all = compute_T(time)

    channel_set = set(int(i) for i in channel_idx)
    T_ch = [dict() for _ in range(len(T_all))]

    for i in channel_idx:
        row = T_all[i]

        # only keep downstream neighbors that are also channel cells
        keep = {j: w for j, w in row.items() if j in channel_set}

        s = sum(keep.values())
        if s > 0:
            for j, w in keep.items():
                T_ch[i][j] = w / s

    return T_ch

def sediment_transport_variableSDR(time, Q0, max_iter=None, tol=None,
                                   sdr_cap=0.999, progress_every=10, final_step=False):

    if max_iter is None:
        max_iter = ROUTING_MAX_ITER
    if tol is None:
        tol = converce_difference

    SDR = np.asarray(cal_SDR(time), dtype=float)
    Q0 = np.asarray(Q0, dtype=float)

    N = Q0.size

    if len(SDR) != N:
        raise RuntimeError(f"[Dimension error] len(SDR)={len(SDR)} != len(Q0)={N}.")

    T_rows = compute_T(time)

    # Important: keep slightly below 1 for numerical convergence
    SDR = np.clip(SDR, 0.0, sdr_cap)

    # f = local available sediment before export
    f = Q0.copy()

    for it in range(1, max_iter + 1):

        incoming = np.zeros_like(f)

        # cell i exports sediment to downstream neighbor j
        for i, row in enumerate(T_rows):
            if not row:
                continue

            exported_i = SDR[i] * f[i]

            for j, w in row.items():
                incoming[j] += w * exported_i

        # local available sediment = local production + upstream incoming
        f_next = Q0 + incoming

        diff = np.sum(np.abs(f_next - f))
        base = max(np.sum(np.abs(f)), 1.0)

        if it % progress_every == 0:
            log(f"[routing/fixpoint] iter {it}: diff={diff:.3e}, rel={diff/base:.3e}")

        f = f_next

        if diff < tol * base:
            log(f"[routing/fixpoint] converged at iter {it} (rel={diff/base:.3e})")
            break

    else:
        log(f"[routing/fixpoint] WARNING: not converged in {max_iter} iters (last rel={diff/base:.3e})")

    delivered = SDR * f
    retained = (1.0 - SDR) * f

    return retained, delivered

def calculate_conc_sed_i(swi, sfi, v_store, v_flow):
    denom = (v_store + v_flow)
    return 0.0 if denom <= 0 else (swi + sfi)/denom

def calculate_conc_sed_max(c_sp, sp_exp, time):
    vx = get_required(time, "x linear velocity (cell centred)")
    vy = get_required(time, "y linear velocity (cell centred)")
    vmax = []
    for Vx, Vy in zip(vx, vy):
        qc = float(np.hypot(Vx, Vy))
        vmax.append(max(0.0, c_sp*(qc**sp_exp)))
    return vmax

def calculate_sed_ch(sed_i, conc_sed_i, conc_sed_max, area_hru, water_level, K_ch, cch):
    V_ch = area_hru * water_level
    if conc_sed_i < conc_sed_max:
        seddeg = max(0.0, (conc_sed_max - conc_sed_i) * V_ch * K_ch * cch)
        seddep = 0.0
    elif conc_sed_i > conc_sed_max:
        seddep = max(0.0, (conc_sed_i - conc_sed_max) * V_ch)
        seddeg = 0.0
    else:
        seddep = 0.0
        seddeg = 0.0
    return sed_i + seddeg - seddep

def cal_final_distribution(time):
    _, _, _, _, sediment_yield_all = parameters_cal(time)
    Q0 = np.asarray(sediment_yield_all, dtype=float)

    retained, delivered = sediment_transport_variableSDR(time, Q0)

    return delivered

def sed_dis_time():
    out = {}
    for t in solution_times:
        out[t] = cal_final_distribution(t)
    return out

def calculate_suspended_sediment(time):

    # 1. get routed sediment mass (t/day)
    routed = cal_final_distribution(time)   # t/day for each cell
    routed = np.array(routed, dtype=float)

    # 2. water volume for each cell
    area_ha = compute_triangle_areas()
    area_m2 = np.array(area_ha) * 10000.0   # m2
    depth = np.array(cal_depth(time))       # m
    V_m3 = area_m2 * depth                  # m3

    # 3. suspended concentration (mg/L)
    # conc (mg/L) = (t * 1e6 mg/kg * 1000kg/t) / (m3 * 1000 L/m3)
    conc_mgL = (routed * 1e9) / (V_m3 * 1000.0)
    conc_mgL = np.nan_to_num(conc_mgL, nan=0.0, posinf=0.0, neginf=0.0)

    # 4. suspended load (kg/day)
    suspended_load_kg_day = conc_mgL * V_m3 / 1e6  # mg/L  kg/day

    return conc_mgL, suspended_load_kg_day
def flow_in_river_with_inflow(
    total_time,
    sed_flowin_river=None,
    initial_sed=None,
    return_details=False,
    channel_export_frac=None,
    return_mass_balance=False
):
    """Route channel sediment with local storage and downstream transfer."""

    N = triangle_count()

    if sed_flowin_river is None:
        sed_flowin_river = np.zeros(N, dtype=float)

    # ----- choose channel cells for this timestep -----
    if CHAN_MASK is not None or CHANNEL_MASK is not None:
        channel_mask_t = build_dynamic_channel_mask_for_time(total_time)
        channel_idx = np.where(np.asarray(channel_mask_t, dtype=bool))[0]
    else:
        zone = np.asarray(zoneN(total_time))
        channel_idx = np.where(zone != 0)[0]

    # ----- inflow sediment, t/day -> t/step -----
    sed_flowin_river = np.asarray(sed_flowin_river, dtype=float)
    sed_in_step_t = np.where(np.isfinite(sed_flowin_river), sed_flowin_river, 0.0) * DT_DAY

    # ----- water flow volume -----
    Qflow_m3_s = np.asarray(cal_Qflow_cell_m3_s(total_time), dtype=float)
    Qflow_m3_s = np.where(np.isfinite(Qflow_m3_s), Qflow_m3_s, 0.0)
    Qflow_m3_s = np.maximum(Qflow_m3_s, 0.0)

    Qflow_m3_day = Qflow_m3_s * DT_SEC
    V_flow_step_m3 = Qflow_m3_day * DT_DAY

    # ----- previous storage -----
    if initial_sed is None:
        sed_wb_i_river = np.zeros(N, dtype=float)
    else:
        sed_wb_i_river = np.asarray(initial_sed, dtype=float)
        sed_wb_i_river = np.where(np.isfinite(sed_wb_i_river), sed_wb_i_river, 0.0)
        sed_wb_i_river = np.maximum(sed_wb_i_river, 0.0)

    # ----- geometry / stored water -----
    global _AREA_M2_CACHE
    if _AREA_M2_CACHE is None:
        _AREA_M2_CACHE = np.asarray(compute_triangle_areas(), dtype=float) * 10000.0

    area_m2 = _AREA_M2_CACHE
    depth = np.asarray(cal_depth(total_time), dtype=float)
    depth = np.where(np.isfinite(depth), depth, 0.0)
    depth = np.maximum(depth, 0.0)

    V_stored_river = area_m2 * depth

    # ----- max concentration, t/m3 -----
    conc_sed_max_river = np.asarray(
        calculate_conc_sed_max(c_sp, sp_exp, total_time),
        dtype=float
    )
    conc_sed_max_river = np.where(np.isfinite(conc_sed_max_river), conc_sed_max_river, 0.0)
    conc_sed_max_river = np.maximum(conc_sed_max_river, 0.0)

    # Optional cap, avoids extreme concentration
    max_conc_mgL = 2000.0
    max_conc_t_m3 = max_conc_mgL * 1e-6
    conc_sed_max_river = np.minimum(conc_sed_max_river, max_conc_t_m3)

    # ----- channel-only downstream routing weights -----
    T_ch = compute_T_channel(total_time, channel_idx)

    # ----- output arrays -----
    sed_next = np.zeros(N, dtype=float)          # storage after downstream routing
    incoming_downstream = np.zeros(N, dtype=float)
    conc_sed_i_array = np.zeros(N, dtype=float)  # t/m3

    channel_set = set(int(i) for i in channel_idx)

    storage_before_t = float(np.sum(sed_wb_i_river[channel_idx]))
    input_t = float(np.sum(sed_in_step_t[channel_idx]))
    deposition_t = 0.0
    generation_t = 0.0
    outlet_export_t = 0.0

    # ----- local mixing, deposition/degradation, and export -----
    for node in channel_idx:
        node = int(node)

        sed_wb_i = float(sed_wb_i_river[node])       # t
        sed_in_t = float(sed_in_step_t[node])        # t/step

        V_stored = float(V_stored_river[node])       # m3
        V_flowin = float(V_flow_step_m3[node])       # m3/step

        # fully mixed concentration before deposition/routing, t/m3
        conc_sed_i = calculate_conc_sed_i(
            sed_wb_i,
            sed_in_t,
            V_stored,
            V_flowin
        )
        conc_sed_i_array[node] = conc_sed_i

        conc_sed_max = float(conc_sed_max_river[node])

        # local deposition / degradation
        V_ch = max(V_stored + V_flowin, 0.0)

        if conc_sed_i < conc_sed_max:
            sed_deg = (conc_sed_max - conc_sed_i) * V_ch * K_ch * cch
            sed_dep = 0.0
        elif conc_sed_i > conc_sed_max:
            sed_dep = (conc_sed_i - conc_sed_max) * V_ch
            sed_deg = 0.0
        else:
            sed_deg = 0.0
            sed_dep = 0.0

        generation_t += max(float(sed_deg), 0.0)
        deposition_t += max(float(sed_dep), 0.0)
        sed_available = sed_wb_i + sed_in_t + sed_deg - sed_dep
        sed_available = max(0.0, sed_available)

        # export only if cell has downstream channel neighbor
        row = T_ch[node]

        if row:
            # Lower export fraction increases in-channel storage and smooths event peaks.
            export_frac_src = CHANNEL_EXPORT_FRAC if channel_export_frac is None else channel_export_frac
            export_frac = float(np.clip(export_frac_src, 0.0, 1.0))

            exported = export_frac * sed_available
            retained = sed_available - exported

            sed_next[node] += retained

            for j, w in row.items():
                incoming_downstream[j] += exported * w

        else:
            # outlet or no downstream channel neighbor: keep as local storage
            sed_next[node] += sed_available

    # Add sediment exported from upstream channel cells to receivers
    for node in channel_idx:
        sed_next[node] += incoming_downstream[node]

    # non-channel storage remains zero
    sed_next = np.where(np.isfinite(sed_next), sed_next, 0.0)
    sed_next = np.maximum(sed_next, 0.0)
    # post-routing concentration for station extraction
    conc_post = np.zeros(N, dtype=float)

    for node in channel_idx:
        node = int(node)

        V_stored = float(V_stored_river[node])
        V_flowin = float(V_flow_step_m3[node])
        denom = V_stored + V_flowin

        if denom > 0:
            conc_post[node] = sed_next[node] / denom
        else:
            conc_post[node] = 0.0

    conc_post = np.where(np.isfinite(conc_post), conc_post, 0.0)
    conc_post = np.maximum(conc_post, 0.0)
    storage_after_t = float(np.sum(sed_next[channel_idx]))
    residual_t = (
        storage_before_t + input_t + generation_t
        - storage_after_t - deposition_t - outlet_export_t
    )
    balance_scale_t = max(
        abs(storage_before_t) + abs(input_t) + abs(generation_t),
        abs(storage_after_t) + abs(deposition_t) + abs(outlet_export_t),
        1.0e-30,
    )
    mass_balance = {
        "storage_before_t": storage_before_t,
        "input_t": input_t,
        "generation_t": generation_t,
        "deposition_t": deposition_t,
        "outlet_export_t": outlet_export_t,
        "storage_after_t": storage_after_t,
        "residual_t": residual_t,
        "relative_error_pct": 100.0 * residual_t / balance_scale_t,
    }

    if not return_details:
        if return_mass_balance:
            return sed_next, mass_balance
        return sed_next

    conc_for_station = conc_post if USE_POST_ROUTING_CONC else conc_sed_i_array
    if return_mass_balance:
        return sed_next, conc_for_station, conc_sed_max_river, mass_balance
    return sed_next, conc_for_station, conc_sed_max_river
def export_river_conc_clamped(time, initial_sed=None, csv_path="river_conc_clamped.csv",
                              sed_flowin_river=None):
    """Export clamped river concentration for one timestep."""
    N = triangle_count()

    if sed_flowin_river is None:
        sed_flowin_river = np.zeros(N, dtype=float)

    sed_ch, conc_sed_i, conc_sed_max = flow_in_river_with_inflow(
        time,
        sed_flowin_river=sed_flowin_river,
        initial_sed=initial_sed,
        return_details=True
    )

    max_conc_mgL = 2000.0
    max_conc_t_per_m3 = max_conc_mgL * 1e-6
    conc_sed_max = np.minimum(np.asarray(conc_sed_max, dtype=float), max_conc_t_per_m3)

    conc_sed_i = np.asarray(conc_sed_i, dtype=float)
    conc_eff   = np.minimum(conc_sed_i, conc_sed_max)
    conc_eff_mgL = conc_eff * 1e6

    centroids = [triangle_centroid(i) for i in range(N)]
    areas_ha = np.array(compute_triangle_areas(), dtype=float)
    areas_ha = np.where(areas_ha <= 0, 1.0, areas_ha)
    areas_m2 = areas_ha * 10000.0
    depth = np.array(cal_depth(time), dtype=float)
    V_ch_m3 = areas_m2 * depth

    rows = []
    for elem_id in range(N):
        x, y = centroids[elem_id]
        rows.append({
            "time_sec": time,
            "elem_id": elem_id,
            "x": x,
            "y": y,
            "area_ha": float(areas_ha[elem_id]),
            "V_ch_m3": float(V_ch_m3[elem_id]),
            "conc_sed_i_t_per_m3": float(conc_sed_i[elem_id]),
            "conc_sed_max_t_per_m3": float(conc_sed_max[elem_id]),
            "conc_eff_t_per_m3": float(conc_eff[elem_id]),
            "conc_eff_mgL": float(conc_eff_mgL[elem_id]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    log(f"[RIVER] clamped concentration exported to: {csv_path}")

def get_mesh_boundary_segments(x1, y1, x2, y2, x3, y3, ndigits=6):
    """Extract outer boundary edges from a triangular mesh."""

    from collections import defaultdict

    edge_count = defaultdict(int)
    edge_geom = {}

    N = len(x1)

    for i in range(N):
        pts = [
            (float(x1[i]), float(y1[i])),
            (float(x2[i]), float(y2[i])),
            (float(x3[i]), float(y3[i]))
        ]

        edges = [
            (pts[0], pts[1]),
            (pts[1], pts[2]),
            (pts[2], pts[0])
        ]

        for p1, p2 in edges:
            # rounded key avoids floating-point mismatch
            k1 = (round(p1[0], ndigits), round(p1[1], ndigits))
            k2 = (round(p2[0], ndigits), round(p2[1], ndigits))

            key = tuple(sorted([k1, k2]))

            edge_count[key] += 1
            edge_geom[key] = [p1, p2]

    boundary_segments = [
        edge_geom[key]
        for key, count in edge_count.items()
        if count == 1
    ]

    return boundary_segments

def plot_shapes(time,out_dir,sediment_values_final):
    if not olf_data:
        print("[ERROR] olf_data is not loaded.")
        return
    x1 = global_coords['x1']; y1 = global_coords['y1']
    x2 = global_coords['x2']; y2 = global_coords['y2']
    x3 = global_coords['x3']; y3 = global_coords['y3']
    #print("lengthx",len(x1))
    zone = zoneN(time)
    sediment_values_final = np.asarray(sediment_values_final, dtype=float)
    # === convert to t/ha for map output ===
    areas_ha_for_plot = np.array(compute_triangle_areas(), dtype=float)
    areas_ha_for_plot = np.where(areas_ha_for_plot <= 0, 1.0, areas_ha_for_plot)
    sediment_values_final = sediment_values_final / areas_ha_for_plot

    plt.figure(figsize=(16, 20))
    # ---------------- color scale ----------------
    sed_plot = np.asarray(sediment_values_final, dtype=float)
    sed_plot = np.where(np.isfinite(sed_plot), sed_plot, 0.0)

    sed_plot = np.asarray(sed_plot, dtype=float)
    sed_plot = np.where(np.isfinite(sed_plot), sed_plot, 0.0)

    # mask 0 or negative values so they appear white
    sed_plot_masked = np.ma.masked_less_equal(sed_plot, 0.0)

    # fixed color scale for ALL figures
    vmin = 1e-10
    vmax = 1e0
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    # darker = more sediment
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="white")
    triangles = np.array(
        [[[x1[i], y1[i]], [x2[i], y2[i]], [x3[i], y3[i]]]
         for i in range(len(zone))],
        dtype=float
    )
    #patches = PolyCollection(triangles, array=sediment_values_final, cmap=cmap, norm=norm, edgecolors='none') # none edge color
    #patches = PolyCollection(triangles, array=sediment_values_final, cmap=cmap, norm=norm, edgecolors='black') # black edge
    patches = PolyCollection(triangles, array=sed_plot_masked, cmap=cmap, norm=norm, edgecolors='face') # same color with mesh
    #patches = PolyCollection(triangles, array=zone, cmap=cmap, norm=norm, edgecolors='face') # same color with mesh
    ax = plt.gca()
    ax.add_collection(patches)
    boundary_segments = get_mesh_boundary_segments(x1, y1, x2, y2, x3, y3)
    boundary_lc = LineCollection(
        boundary_segments,
        colors="black",
        linewidths=1.2,
        zorder=10
    )

    ax.add_collection(boundary_lc)
    xmin = min(np.min(x1), np.min(x2), np.min(x3))
    xmax = max(np.max(x1), np.max(x2), np.max(x3))
    ymin = min(np.min(y1), np.min(y2), np.min(y3))
    ymax = max(np.max(y1), np.max(y2), np.max(y3))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(patches, ax=ax, label="Sediment (t/ha per day)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Sediment Transport Simulation @ time={time} (dt = 1 day, t/ha)")
    plt.grid(True)
    
    out_map_dir = os.path.join(out_dir, "maps_sed_tpha")
    os.makedirs(out_map_dir, exist_ok=True)

    out_png = os.path.join(out_map_dir, f"sediment_map_{int(time)}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    log(f"[FIG] sediment map exported: {out_png}")

    plt.close()

def plot_zones(time):
    if not olf_data:
        print("[ERROR] olf_data is not loaded.")
        return
    x1 = global_coords['x1']; y1 = global_coords['y1']
    x2 = global_coords['x2']; y2 = global_coords['y2']
    x3 = global_coords['x3']; y3 = global_coords['y3']

    zone = zoneN(time)

    plt.figure(figsize=(16, 20))
    vmin = 0.0
    vmax = 10
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.Reds

    triangles = np.array(
        [[[x1[i], y1[i]], [x2[i], y2[i]], [x3[i], y3[i]]]
         for i in range(len(zone))],
        dtype=float
    )
    #patches = PolyCollection(triangles, array=sediment_values_final, cmap=cmap, norm=norm, edgecolors='none') # none edge color
    #patches = PolyCollection(triangles, array=sediment_values_final, cmap=cmap, norm=norm, edgecolors='black') # black edge
    #patches = PolyCollection(triangles, array=sediment_values_final, cmap=cmap, norm=norm, edgecolors='face') # same color with mesh
    patches = PolyCollection(triangles, array=zone, cmap=cmap, norm=norm, edgecolors='face') # same color with mesh
    ax = plt.gca()
    ax.add_collection(patches)
    ax.autoscale_view()
    plt.colorbar(
    patches,
    ax=ax,
    label="Sediment load (t ha$^{-1}$ day$^{-1}$)"
)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Sediment Transport Simulation @ time={time} (dt = 1 day, t/ha)")
    plt.grid(True)
    plt.show()
    print('finish')

def build_initial_sed_from_conc(time0, conc0_mgL):
    """Build initial river sediment storage from background concentration."""
    areas_ha = np.array(compute_triangle_areas(), dtype=float)
    areas_ha = np.where(areas_ha <= 0, 1.0, areas_ha)
    areas_m2 = areas_ha * 10000.0

    depth0 = np.array(cal_depth(time0), dtype=float)   # m

    V0_m3 = areas_m2 * depth0                          # m^3

    conc0_arr = np.array(conc0_mgL, dtype=float)
    if conc0_arr.size == 1:
        conc0_arr = np.full_like(V0_m3, conc0_arr.item())
    elif conc0_arr.size != V0_m3.size:
        raise ValueError("conc0_mgL length does not match the number of cells N")

    sed0_t = conc0_arr * V0_m3 * 1e-6   # t

    zone = np.array(zoneN(time0), dtype=float)
    sed0_t = np.where(zone > 0, sed0_t, 0.0)

    return sed0_t
def export_daily_Q_timeseries(selected_elems, out_csv_path):
    """Export daily routed sediment time series for selected elements."""
    import numpy as _np

    if not solution_times:
        log("[EXPORT] solution_times ")
        return

    schedule = build_daily_schedule(solution_times, dt=DT_SEC)

    rows = []

    for day_idx, (day_time, hydro_time) in enumerate(schedule):
        Q_surf, qch, q_peak, sed_by_ts, sed_all = parameters_cal(hydro_time)
        Q0 = _np.array(sed_all, dtype=float)

        _, Q_total = sediment_transport_variableSDR(hydro_time, Q0)
        Q_total = _np.asarray(Q_total, dtype=float)  # t/day per cell

        for elem_id in selected_elems:
            if elem_id < 0 or elem_id >= Q_total.size:
                continue

            rows.append({
                "day_index": day_idx,
                "day_time_sec": float(day_time),
                "day_time_day": float(day_time / DT_SEC),   # 
                "hydro_time_sec": float(hydro_time),
                "elem_id": int(elem_id),
                "Q_total_t_day": float(Q_total[elem_id]),   #  t/day
            })

    if not rows:
        log("[EXPORT]  selected_elems ")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    log(f"[EXPORT] daily Q_total timeseries exported to: {out_csv_path}")




def cal_Qproxy_from_vh(time, areas_m2):
    """Build a discharge proxy from velocity, depth, and cell area."""
    vx = np.asarray(get_required(time, FIELD_KEYS["vx_cell"]), dtype=float)
    vy = np.asarray(get_required(time, FIELD_KEYS["vy_cell"]), dtype=float)
    vmag = np.hypot(vx, vy)  # m/s

    depth = np.asarray(cal_depth(time), dtype=float)  # m
    L = np.sqrt(np.asarray(areas_m2, dtype=float))    # m

    Q_proxy_m3s = vmag * depth * L                    # m3/s proxy
    Q_proxy_m3s = np.maximum(Q_proxy_m3s, 0.0)
    return Q_proxy_m3s * DT_SEC                       # m3/day
def save_sediment_map(time, sediment_t_per_ha, out_png_path, title_prefix="Sediment map"):
    """Save a sediment map from per-cell t/ha/day values."""
    x1 = global_coords['x1']; y1 = global_coords['y1']
    x2 = global_coords['x2']; y2 = global_coords['y2']
    x3 = global_coords['x3']; y3 = global_coords['y3']

    sediment_t_per_ha = np.asarray(sediment_t_per_ha, dtype=float)

    # robust vmax (avoid single extreme point blowing up color)
    vmin = 0.0
    #vmax = float(np.nanpercentile(sediment_t_per_ha, 99))
    vmax = 0.5
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(sediment_t_per_ha)) if sediment_t_per_ha.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = "inferno"  
    #cmap = "magma"
    #cmap = "viridis"


    triangles = np.array(
        [[[x1[i], y1[i]], [x2[i], y2[i]], [x3[i], y3[i]]]
         for i in range(len(x1))],
        dtype=float
    )

    fig = plt.figure(figsize=(14, 18))
    ax = plt.gca()

    patches = PolyCollection(
        triangles,
        array=sediment_t_per_ha,
        cmap=cmap,
        norm=norm,
        edgecolors='face',  # mesh 
        linewidths=0.0
    )
    ax.add_collection(patches)
    ax.autoscale_view()

    cb = plt.colorbar(patches, ax=ax)
    cb.set_label("Sediment (t/ha per day)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"{title_prefix} @ time={time}")

    fig.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
def plot_exchange_map(time, out_png=None, dpi=300, mask_zero=True):
    # --- mesh coords ---
    x1 = global_coords['x1']; y1 = global_coords['y1']
    x2 = global_coords['x2']; y2 = global_coords['y2']
    x3 = global_coords['x3']; y3 = global_coords['y3']

    # --- exchange flux per cell ---
    ex = np.asarray(cal_EF(time), dtype=float)
    ex[np.abs(ex) < 1e-12] = 0.0   # treat tiny noise as zero

    if mask_zero:
        ex_plot = np.ma.masked_equal(ex, 0.0)   # 0 -> white
    else:
        ex_plot = ex

    # --- robust symmetric range ---
    vmax = float(np.nanpercentile(np.abs(ex), 99))
    if (not np.isfinite(vmax)) or vmax <= 0:
        vmax = 1.0

    # symmetric around 0
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    cmap = cm.RdBu_r.copy()   # negative=blue, positive=red
    cmap.set_bad("white")

    # --- triangles ---
    triangles = np.array(
        [[[x1[i], y1[i]], [x2[i], y2[i]], [x3[i], y3[i]]]
         for i in range(len(x1))],
        dtype=float
    )

    # Use the same figure style as the sediment map output.
    fig, ax = plt.subplots(figsize=(16, 20), dpi=dpi)

    patches = PolyCollection(
        triangles,
        array=ex_plot,
        cmap=cmap,
        norm=norm,
        edgecolors='face',
        linewidths=0
    )

    ax.add_collection(patches)
    boundary_segments = get_mesh_boundary_segments(x1, y1, x2, y2, x3, y3)
    boundary_lc = LineCollection(
        boundary_segments,
        colors="black",
        linewidths=1.2,
        zorder=10
    )
    ax.add_collection(boundary_lc)
    xmin = min(np.min(x1), np.min(x2), np.min(x3))
    xmax = max(np.max(x1), np.max(x2), np.max(x3))
    ymin = min(np.min(y1), np.min(y2), np.min(y3))
    ymax = max(np.max(y1), np.max(y2), np.max(y3))

    xpad = (xmax - xmin) * 0.02
    ypad = (ymax - ymin) * 0.02

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal", adjustable="box")

    # colorbar
    cbar = fig.colorbar(patches, ax=ax)
    cbar.set_label("Exchange flux (negative = losing, positive = gaining)")

    # keep coordinates
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # no title
    # ax.set_title(...)

    # optional: same tick style
    ax.tick_params(axis="both", labelsize=10)

    # optional grid, if you want same as sediment output
    ax.grid(True)

    fig.tight_layout()

    if out_png is not None:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
def write_pest_txt_from_station_csv(station_csv_path, out_dir):
    import pandas as pd
    import numpy as np
    import os

    # Read station SSC values.
    df = pd.read_csv(station_csv_path)

    ssc_col = "SSC_station_mgL"

    if ssc_col not in df.columns:
        raise ValueError(
            f"Missing column '{ssc_col}' in {station_csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    s = df[ssc_col].astype(float).to_numpy()
    s = s[np.isfinite(s)]

    if len(s) == 0:
        ssc_last = 0.0
        ssc_peak = 0.0
        ssc_mean = 0.0
    else:
        ssc_last = float(s[-1])
        ssc_peak = float(np.max(s))
        ssc_mean = float(np.mean(s))

    # Write PEST observation values.
    pest_txt = os.path.join(out_dir, "pest_values.txt")

    with open(pest_txt, "w", encoding="utf-8") as f:
        f.write(f"ssc_last {ssc_last:.6f}\n")
        f.write(f"ssc_peak {ssc_peak:.6f}\n")
        f.write(f"ssc_mean {ssc_mean:.6f}\n")

    return pest_txt
def write_pest_txt_from_station_df(df_station, out_dir):
    import os, numpy as np

    # prefer explicit SSC column names
    if "SSC_station_mgL" in df_station.columns:
        ssc_col = "SSC_station_mgL"
    elif "river_SSC_station_mgL" in df_station.columns:
        ssc_col = "river_SSC_station_mgL"
    else:
        # fallback: try to auto-detect
        ssc_col = None
        for c in df_station.columns:
            cl = c.lower()
            if ("ssc" in cl) and ("mg" in cl or "mgl" in cl):
                ssc_col = c
                break
        if ssc_col is None:
            raise ValueError(f"Cannot find SSC column. Columns={list(df_station.columns)}")

    time_col = "time_sec" if "time_sec" in df_station.columns else df_station.columns[0]
    times = df_station[time_col].to_numpy()
    ssc = df_station[ssc_col].astype(float).to_numpy()

    pest_txt = os.path.join(out_dir, "pest_values.txt")
    with open(pest_txt, "w") as f:
        for i, (t, v) in enumerate(zip(times, ssc)):
            if np.isfinite(v):
                f.write(f"ssc_{i:03d} {v:.8e}\n")

    return pest_txt
def plot_mesh_and_channel_mask(channel_mask, out_dir, max_mesh_lines=171609):
    """Export the mesh and active channel-mask diagnostic figure."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    os.makedirs(out_dir, exist_ok=True)

    N = triangle_count()
    channel_mask = np.asarray(channel_mask, dtype=bool)

    if len(channel_mask) != N:
        raise ValueError(f"channel_mask length mismatch: {len(channel_mask)} vs mesh cells {N}")

    # ---- Build triangle polygons and mesh edges ----
    x1 = np.asarray(global_coords["x1"], dtype=float)
    y1 = np.asarray(global_coords["y1"], dtype=float)
    x2 = np.asarray(global_coords["x2"], dtype=float)
    y2 = np.asarray(global_coords["y2"], dtype=float)
    x3 = np.asarray(global_coords["x3"], dtype=float)
    y3 = np.asarray(global_coords["y3"], dtype=float)

    polygons = [
        [(x1[i], y1[i]), (x2[i], y2[i]), (x3[i], y3[i])]
        for i in range(N)
    ]

    # For panel (a), draw mesh as edges.
    # If too slow, only draw a subset, but default uses all surface elements.
    draw_idx = np.arange(N)
    if max_mesh_lines is not None and N > max_mesh_lines:
        draw_idx = np.linspace(0, N - 1, max_mesh_lines).astype(int)

    mesh_segments = []
    for i in draw_idx:
        mesh_segments.extend([
            [(x1[i], y1[i]), (x2[i], y2[i])],
            [(x2[i], y2[i]), (x3[i], y3[i])],
            [(x3[i], y3[i]), (x1[i], y1[i])]
        ])

    channel_polygons = [polygons[i] for i in np.where(channel_mask)[0]]

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

    # Panel (a): surface triangular mesh
    ax = axes[0]
    lc = LineCollection(mesh_segments, colors="0.60", linewidths=0.12, alpha=0.6)
    ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_title("(a) Surface triangular mesh", fontsize=11)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.tick_params(axis="both", labelsize=8)
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=5)

    # Panel (b): channel mask / hydraulically connected elements
    ax = axes[1]

    # light background mesh boundary
    bg = PolyCollection(
        polygons,
        facecolors="0.92",
        edgecolors="none",
        linewidths=0
    )
    ax.add_collection(bg)

    # highlighted channel / connected elements
    if len(channel_polygons) > 0:
        pc = PolyCollection(
            channel_polygons,
            facecolors="#1f78b4",
            edgecolors="none",
            alpha=0.9
        )
        ax.add_collection(pc)

    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_title("(b) Channel and hydraulically active cells", fontsize=11)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.tick_params(axis="both", labelsize=8)
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=5)

    legend_items = [
        Patch(facecolor="0.92", edgecolor="none", label="Surface mesh domain"),
        Patch(facecolor="#1f78b4", edgecolor="none", label="Hydraulically active cells")
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=True)


    out_png = os.path.join(out_dir, "figure_mesh_channel_mask.png")
    out_pdf = os.path.join(out_dir, "figure_mesh_channel_mask.pdf")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log(f"[FIG] Mesh/channel mask figure exported: {out_png}")
    log(f"[FIG] Mesh/channel mask figure exported: {out_pdf}")

    return out_png, out_pdf

def write_cell_ssc_output(
    out_dir,
    t,
    idx,
    centroids,
    areas_m2,
    channel_mask_t,
    rain,
    Qflow_raw_m3_s,
    Qexchange_m3_s,
    Qflow_m3_s,
    delivered,
    conc_i,
    conc_max,
):
    """Write whole-basin, per-cell SSC diagnostics for one timestep."""
    cell_dir = os.path.join(out_dir, "cell_SSC_outputs")
    os.makedirs(cell_dir, exist_ok=True)

    centroids_arr = np.asarray(centroids, dtype=float)
    areas_m2 = np.asarray(areas_m2, dtype=float)
    channel_mask_t = np.asarray(channel_mask_t, dtype=bool)
    rain = np.asarray(rain, dtype=float)
    Qflow_raw_m3_s = np.asarray(Qflow_raw_m3_s, dtype=float)
    Qexchange_m3_s = np.asarray(Qexchange_m3_s, dtype=float)
    Qflow_m3_s = np.asarray(Qflow_m3_s, dtype=float)
    delivered = np.asarray(delivered, dtype=float)
    conc_i = np.asarray(conc_i, dtype=float)
    conc_max = np.asarray(conc_max, dtype=float)

    conc_eff = np.minimum(conc_i, conc_max)
    conc_i = np.where(np.isfinite(conc_i), conc_i, 0.0)
    conc_max = np.where(np.isfinite(conc_max), conc_max, 0.0)
    conc_eff = np.where(np.isfinite(conc_eff), conc_eff, 0.0)

    Qflow_raw_m3_day = Qflow_raw_m3_s * DT_SEC
    Qexchange_m3_day = Qexchange_m3_s * DT_SEC
    Qflow_eff_m3_day = Qflow_m3_s * DT_SEC
    delta_Q_m3_day = Qflow_eff_m3_day - Qflow_raw_m3_day

    with np.errstate(divide="ignore", invalid="ignore"):
        exchange_fraction_of_raw = np.where(
            Qflow_raw_m3_day > 0.0,
            delta_Q_m3_day / Qflow_raw_m3_day,
            np.nan,
        )
        SSC_from_delivered_rawQ_mgL = np.where(
            Qflow_raw_m3_day > 0.0,
            delivered * 1.0e6 / Qflow_raw_m3_day,
            0.0,
        )
        SSC_from_delivered_effQ_mgL = np.where(
            Qflow_eff_m3_day > 0.0,
            delivered * 1.0e6 / Qflow_eff_m3_day,
            0.0,
        )

    SSC_from_delivered_rawQ_mgL = np.where(np.isfinite(SSC_from_delivered_rawQ_mgL), SSC_from_delivered_rawQ_mgL, 0.0)
    SSC_from_delivered_effQ_mgL = np.where(np.isfinite(SSC_from_delivered_effQ_mgL), SSC_from_delivered_effQ_mgL, 0.0)
    delta_SSC_exchange_mgL = SSC_from_delivered_effQ_mgL - SSC_from_delivered_rawQ_mgL

    df_cell = pd.DataFrame({
        "time_sec": float(t),
        "step_index": int(idx),
        "cell_id": np.arange(len(areas_m2), dtype=int),
        "x_centroid": centroids_arr[:, 0],
        "y_centroid": centroids_arr[:, 1],
        "area_m2": areas_m2,
        "is_channel_mask": channel_mask_t.astype(int),
        "rain_cell": rain,
        "Q_raw_m3_s": Qflow_raw_m3_s,
        "Q_exchange_m3_s": Qexchange_m3_s,
        "Q_eff_m3_s": Qflow_m3_s,
        "Q_raw_m3_day": Qflow_raw_m3_day,
        "Q_exchange_m3_day": Qexchange_m3_day,
        "Q_eff_m3_day": Qflow_eff_m3_day,
        "delta_Q_m3_day": delta_Q_m3_day,
        "exchange_fraction_of_raw": exchange_fraction_of_raw,
        "delivered_sediment_t_day": delivered,
        "SSC_from_delivered_rawQ_mgL": SSC_from_delivered_rawQ_mgL,
        "SSC_from_delivered_effQ_mgL": SSC_from_delivered_effQ_mgL,
        "delta_SSC_exchange_mgL": delta_SSC_exchange_mgL,
        "SSC_river_raw_mgL": conc_i * 1e6,
        "SSC_river_eff_mgL": conc_eff * 1e6,
        "SSC_river_max_mgL": conc_max * 1e6,
    })

    out_csv = os.path.join(cell_dir, f"cell_SSC_t{int(t)}_step{int(idx):04d}.csv")
    df_cell.to_csv(out_csv, index=False)
    return out_csv

def main():
    global olf_data, solution_times, MESH_TIME
    global _AREA_M2_CACHE
    global MASSBAL_IN_CUM, MASSBAL_OUT_CUM, MASSBAL_STORAGE_LAST

    # Read configuration and apply global parameters.
    cfg = load_params_file(os.environ.get("GWSWI_CONFIG_PATH", os.path.join(APP_DIR, "Model_Config.txt")))
    apply_params(cfg)

    olf_path = str(require_param(cfg, "olf_path"))
    out_dir_cfg = str(require_param(cfg, "out_dir"))
    chan_path = str(require_param(cfg, "chan_path"))
    log(
        "[PARAMS] "
        f"SC={SC}, C_USLE={C_USLE}, K_ch={K_ch}, "
        f"a={a}, b={b}, conc0_global={conc0_global}, "
        f"Qcrit_m3_day={Qcrit_m3_day}, resus_k_mgL={resus_k_mgL}, resus_b={resus_b}, "
        f"C_base_mgL={C_base_mgL}, C_resus_cap_mgL={C_resus_cap_mgL}, "
        f"conc0_global={conc0_global}, Qmin_station_m3_day={Qmin_station_m3_day}"
    )

    log(f"[INFO] Reading OLF data from: {olf_path}")
    if chan_path:
        log(f"[INFO] CHAN file for channel mask: {chan_path}")
    else:
        log("[INFO] No CHAN file provided; fallback to OLF depth threshold.")

    # Reset caches and cumulative trackers for this run.
    _AREA_M2_CACHE = None
    MASSBAL_IN_CUM = 0.0
    MASSBAL_OUT_CUM = 0.0
    MASSBAL_STORAGE_LAST = 0.0

    # Read HGS OLF data.
    olf_data = read_olf_variables(olf_path)
    solution_times = sorted(olf_data.keys())

    if not solution_times:
        log("[ERROR] No solution times found in OLF file.")
        return

    if ENABLE_NUTRIENT_MODULE:
        nutrient_keys = list(FIELD_KEYS["nutrient_solute_node"])
        nutrient_times = []
        last_nutrient = {}
        for t in solution_times:
            data_t = olf_data.get(t, {})
            found = False
            for key in nutrient_keys:
                if key in data_t:
                    last_nutrient[key] = data_t[key]
                    found = True
            if found:
                nutrient_times.append(t)
            elif last_nutrient:
                data_t.update(last_nutrient)

        if nutrient_times:
            filled_count = len(solution_times) - len(nutrient_times)
            log(
                f"[NUTRIENT] Found Groundwater_N at {len(nutrient_times)}/{len(solution_times)} "
                f"solution time(s); filled {filled_count} shared/missing time(s) from the previous nutrient block."
            )
        else:
            log("[NUTRIENT] No Groundwater_N blocks found; dissolved nutrient will be zero for all times.")

    olf_data = {t: olf_data[t] for t in solution_times}

    log(
        f"[INFO] Times sorted: first={solution_times[0]}, "
        f"last={solution_times[-1]}, total={len(solution_times)}"
    )
    log(f"[INFO] Parsed {len(solution_times)} solution times.")

    MESH_TIME = find_mesh_time()
    log(f"[DIAG] MESH_TIME = {MESH_TIME}")

    # Prepare geometry.
    prepare_global_coords()
    prepare_neighbors()

    N = triangle_count()
    log(f"[INFO] Triangle / mesh count = {N}")

    centroids = [triangle_centroid(i) for i in range(N)]

    areas_ha = np.asarray(compute_triangle_areas(), dtype=float)
    areas_ha = np.where(np.isfinite(areas_ha) & (areas_ha > 0.0), areas_ha, 1.0)
    areas_m2 = areas_ha * 10000.0

    # Build masks and station geometry.
    def build_channel_mask():
        """Build depth, CHAN, and sediment-routing masks."""

        global DEPTH_MASK, CHAN_MASK, CHANNEL_MASK

        # Depth-based active-water mask.
        md_node = np.asarray(
            get_required(MESH_TIME, FIELD_KEYS["depth_node"]),
            dtype=float
        )

        md_cell = np.asarray(
            average_node_field_to_cells(md_node.tolist()),
            dtype=float
        )

        depth_mask = md_cell > depth_threshold

        log(
            f"[INIT] depth-based active-water mask cells = "
            f"{int(depth_mask.sum())} / {len(depth_mask)} "
            f"(threshold={depth_threshold})"
        )

        # CHAN-based true channel mask.
        chan_mask = np.zeros_like(depth_mask, dtype=bool)

        if chan_path and os.path.exists(chan_path):
            try:
                chan_mask = build_chan_mask_by_centroid_mapping(
                    chan_path,
                    max_dist_m=chan_map_dist
                )

                log(
                    f"[INIT] CHAN-based true channel cells = "
                    f"{int(chan_mask.sum())} / {len(chan_mask)}"
                )

            except Exception as e:
                log(f"[WARN] Failed to build CHAN-based mask: {e}")
                log("[WARN] Using empty CHAN mask.")
        else:
            log("[WARN] CHAN file not found. Using empty CHAN mask.")

        # Fixed base sediment-transport mask.
        channel_mask = chan_mask.copy()

        DEPTH_MASK = depth_mask
        CHAN_MASK = chan_mask
        CHANNEL_MASK = channel_mask

        log(
            f"[INIT] fixed base sediment transport mask = "
            f"{int(channel_mask.sum())} / {len(channel_mask)} "
            f"(CHAN={int(chan_mask.sum())}, dynamic_neighbors_enabled={int(USE_DYNAMIC_CHANNEL_MASK)}, "
            f"rings={int(CHANNEL_NEIGHBOR_RINGS)}, "
            f"neighbor_depth_threshold={float(CHANNEL_NEIGHBOR_DEPTH_THRESHOLD)}, "
            f"depth_active={int(depth_mask.sum())})"
        )

        return channel_mask

    def build_station_window_utm(station_xy, centroids, channel_mask, R):
        sx, sy = station_xy
        R2 = R * R
        ids = []
        for i, (x, y) in enumerate(centroids):
            if not channel_mask[i]:
                continue
            dx = x - sx
            dy = y - sy
            if dx * dx + dy * dy <= R2:
                ids.append(i)
        return ids

    CHANNEL_MASK = build_channel_mask()
    station_xy = (
        float(require_param(cfg, "station_xy_x")),
        float(require_param(cfg, "station_xy_y")),
    )
    STATION_R = float(require_param(cfg, "station_r"))

    log(f"[DIAG] station_xy used = {station_xy}")
    log(f"[DIAG] station_r used  = {STATION_R}")

    station_ids = build_station_window_utm(station_xy, centroids, CHANNEL_MASK, R=STATION_R)
    station_ids_arr = np.asarray(station_ids, dtype=int) if len(station_ids) > 0 else np.asarray([], dtype=int)

    if station_ids_arr.size > 0:
        station_area_m2 = float(np.sum(areas_m2[station_ids_arr]))
    else:
        station_area_m2 = np.nan

    log(f"[STA] station window size = {len(station_ids)} (R={STATION_R} m)")
    log(f"[STA] station area = {station_area_m2:.6e} m2")

    # Output directory.
    if bool(int(require_param(cfg, "timestamp_output_dir"))):
        run_label = time.strftime(str(require_param(cfg, "output_timestamp_format")))
        out_dir = os.path.join(out_dir_cfg, run_label)
    else:
        out_dir = out_dir_cfg

    os.makedirs(out_dir, exist_ok=True)
    log(f"[INFO] Output directory: {out_dir}")
    if not pest_mode:
        mesh_fig_dir = os.path.join(out_dir, "figures_mesh")
        plot_mesh_and_channel_mask(CHANNEL_MASK, mesh_fig_dir)
    log(f"[INFO] pest_mode={pest_mode}, write_cell_ssc_outputs={WRITE_CELL_SSC_OUTPUTS}")
    log("[MAIN] Start combined hillslope+SDR + river routing loop.")

    # Initial river storage from background concentration.
    time0 = solution_times[0]
    sed_prev = build_initial_sed_from_conc(time0, conc0_global)

    # Output containers.
    station_ts = []
    cell_output_records = []
    retained_series = {}
    delivered_series = {}

    if ENABLE_DAILY_LOOP:
        retained_series, delivered_series = simulate_daily_loop()
    Q_MIN = Qmin_station_m3_day
    C_BG = conc0_global 

    # Main time loop.
    massbal_cumulative_residual_t = 0.0
    for idx, t in enumerate(solution_times):
        log(f"[STEP] Processing time = {t} s (index {idx})")

        # Hillslope production and SDR routing.
        Q_surf, _, q_peak, _, sediment_yield_all = parameters_cal(t)
        q_peak = np.asarray(q_peak, dtype=float)
        Q0 = np.asarray(sediment_yield_all, dtype=float)

        if ENABLE_DAILY_LOOP:
            retained = np.asarray(retained_series.get(t, np.zeros_like(Q0)), dtype=float)
            delivered = np.asarray(delivered_series.get(t, np.zeros_like(Q0)), dtype=float)
        else:
            retained, delivered = sediment_transport_variableSDR(t, Q0)
            retained = np.asarray(retained, dtype=float)
            delivered = np.asarray(delivered, dtype=float)

            if not pest_mode:
                retained_series[t] = retained.copy()
                delivered_series[t] = delivered.copy()

        # Rainfall diagnostics.
        rain = np.asarray(cal_Precipitation(t), dtype=float)
        rain = np.where(np.isfinite(rain), rain, 0.0)

        # Channel discharge.
        Qflow_raw_m3_s = np.asarray(cal_Qflow_raw_cell_m3_s(t), dtype=float)
        Qexchange_m3_s = np.asarray(cal_exchange_q_cell_m3_s(t), dtype=float)
        Qflow_m3_s = np.asarray(cal_Qflow_cell_m3_s(t), dtype=float)
        Qflow_raw_m3_s = np.where(np.isfinite(Qflow_raw_m3_s), Qflow_raw_m3_s, 0.0)
        Qexchange_m3_s = np.where(np.isfinite(Qexchange_m3_s), Qexchange_m3_s, 0.0)
        Qflow_m3_s = np.where(np.isfinite(Qflow_m3_s), Qflow_m3_s, 0.0)
        Qflow_raw_m3_s = np.maximum(Qflow_raw_m3_s, 0.0)
        Qflow_m3_s = np.maximum(Qflow_m3_s, 0.0)

        # River routing.
        sed_prev, conc_i, conc_max, massbal = flow_in_river_with_inflow(
            total_time=t,
            sed_flowin_river=delivered,
            initial_sed=sed_prev,
            return_details=True,
            return_mass_balance=True,
        )

        MASSBAL_IN_CUM += massbal["input_t"] + massbal["generation_t"]
        MASSBAL_OUT_CUM += massbal["deposition_t"] + massbal["outlet_export_t"]
        MASSBAL_STORAGE_LAST = massbal["storage_after_t"]
        massbal_cumulative_residual_t += massbal["residual_t"]
        log(
            f"[MASS BALANCE] idx={idx}, t={t:.0f} s, "
            f"storage_before={massbal['storage_before_t']:.6e} t, "
            f"input={massbal['input_t']:.6e} t, "
            f"generation={massbal['generation_t']:.6e} t, "
            f"deposition={massbal['deposition_t']:.6e} t, "
            f"outlet={massbal['outlet_export_t']:.6e} t, "
            f"storage_after={massbal['storage_after_t']:.6e} t, "
            f"step_residual={massbal['residual_t']:.6e} t, "
            f"step_error={massbal['relative_error_pct']:.6e} %, "
            f"cumulative_residual={massbal_cumulative_residual_t:.6e} t"
        )

        sed_prev = np.asarray(sed_prev, dtype=float)
        conc_i = np.asarray(conc_i, dtype=float)
        conc_max = np.asarray(conc_max, dtype=float)

        channel_mask_t = build_dynamic_channel_mask_for_time(t)
        station_ids = build_station_window_utm(station_xy, centroids, channel_mask_t, R=STATION_R)
        station_ids_arr = np.asarray(station_ids, dtype=int) if len(station_ids) > 0 else np.asarray([], dtype=int)
        if station_ids_arr.size > 0:
            station_area_m2 = float(np.sum(areas_m2[station_ids_arr]))
        else:
            station_area_m2 = np.nan

        if WRITE_CELL_SSC_OUTPUTS:
            cell_csv = write_cell_ssc_output(
                out_dir=out_dir,
                t=t,
                idx=idx,
                centroids=centroids,
                areas_m2=areas_m2,
                channel_mask_t=channel_mask_t,
                rain=rain,
                Qflow_raw_m3_s=Qflow_raw_m3_s,
                Qexchange_m3_s=Qexchange_m3_s,
                Qflow_m3_s=Qflow_m3_s,
                delivered=delivered,
                conc_i=conc_i,
                conc_max=conc_max,
            )
            cell_output_records.append({
                "time_sec": float(t),
                "step_index": int(idx),
                "cell_csv": cell_csv,
            })

        if idx < 5 or idx % 20 == 0:
            log(
                f"[STA MASK] idx={idx}, t={t:.0f}, "
                f"dynamic_channel_cells={int(np.sum(channel_mask_t))}, "
                f"station_cells={int(station_ids_arr.size)}"
            )

        # Station aggregation using the station window only.
        if station_ids_arr.size > 0:
            sid = station_ids_arr

            rain_station_mean = float(np.nanmean(rain[sid])) if sid.size > 0 else np.nan
            rain_station_max = float(np.nanmax(rain[sid])) if sid.size > 0 else np.nan

            Q_station_raw_m3_s = float(np.sum(np.maximum(Qflow_raw_m3_s[sid], 0.0)))
            Q_exchange_station_m3_s = float(np.sum(Qexchange_m3_s[sid]))
            Q_station_m3_s_hgs = float(np.sum(np.maximum(Qflow_m3_s[sid], 0.0)))
            if q_peak.size > 0:
                Q_station_peak_m3_s = float(np.nanmax(np.maximum(q_peak[sid], 0.0)))
            else:
                Q_station_peak_m3_s = np.nan

            Q_station_m3_s = Q_station_m3_s_hgs
            if Q_station_m3_s <= 0.0 and np.isfinite(Q_station_peak_m3_s) and Q_station_peak_m3_s > 0.0:
                Q_station_m3_s = Q_station_peak_m3_s * 1000.0

            Q_station_raw_m3_day = Q_station_raw_m3_s * DT_SEC
            Q_exchange_station_m3_day = Q_exchange_station_m3_s * DT_SEC
            Q_station_m3_day_raw_calc = Q_station_m3_s * DT_SEC
            if np.isfinite(Q_MIN) and Q_MIN > 0.0:
                Q_station_m3_day = max(float(Q_station_m3_day_raw_calc), float(Q_MIN))
                Q_station_m3_s = Q_station_m3_day / DT_SEC
            else:
                Q_station_m3_day = Q_station_m3_day_raw_calc
            delta_Q_station_m3_day = Q_station_m3_day - Q_station_raw_m3_day
            exchange_effect_expected_m3_day = -Q_exchange_station_m3_day
            if Q_station_raw_m3_day > 0.0:
                exchange_fraction_of_raw = delta_Q_station_m3_day / Q_station_raw_m3_day
            else:
                exchange_fraction_of_raw = np.nan

            if np.isfinite(station_area_m2) and station_area_m2 > 0.0:
                q_station_m_s = Q_station_m3_s / station_area_m2
            else:
                q_station_m_s = np.nan

            # hillslope+SDR delivered sediment in station window, kept as diagnostic only
            L_station_t_day = float(np.sum(delivered[sid]))

            # station SSC from river-routing concentration field
            weights = np.maximum(Qflow_m3_s[sid], 0.0)

            # q_peak diagnostic.
            if q_peak.size > 0:
                old_qpeak_station_m3_s = float(np.nanmax(q_peak[sid]))
            else:
                old_qpeak_station_m3_s = np.nan

            if np.isfinite(station_area_m2) and station_area_m2 > 0.0 and np.isfinite(old_qpeak_station_m3_s):
                old_qpeak_station_m_s = old_qpeak_station_m3_s / station_area_m2
            else:
                old_qpeak_station_m_s = np.nan

            # Station SSC from the river-routing concentration field.

            conc_station = np.asarray(conc_i[sid], dtype=float)
            conc_max_station = np.asarray(conc_max[sid], dtype=float)
            weights = np.asarray(np.maximum(Qflow_m3_s[sid], 0.0), dtype=float)

            # Effective concentration, limited by transport capacity.
            conc_eff_station = np.minimum(conc_station, conc_max_station)

            valid = (
                np.isfinite(conc_eff_station)
                & np.isfinite(weights)
                & (weights > 0)
            )

            if np.any(valid):
                SSC_event_mgL = float(np.average(conc_eff_station[valid], weights=weights[valid]) * 1e6)
            else:
                SSC_event_mgL = 0.0

            SSC_event_saturated_mgL = apply_event_saturation(SSC_event_mgL)
            SSC_base_component_mgL = float(C_base_mgL) if USE_BASE_CONC else 0.0

            if USE_RESUSPENSION and np.isfinite(Q_station_m3_day):
                Q_excess_m3_day = max(float(Q_station_m3_day) - float(Qcrit_m3_day), 0.0)
                SSC_resus_component_mgL = float(resus_k_mgL) * (Q_excess_m3_day ** float(resus_b))
                SSC_resus_component_mgL = min(
                    max(SSC_resus_component_mgL, 0.0),
                    max(float(C_resus_cap_mgL), 0.0)
                )
            else:
                Q_excess_m3_day = 0.0
                SSC_resus_component_mgL = 0.0

            SSC_event_component_mgL = (
                float(MUSLE_SSC_WEIGHT) * SSC_event_saturated_mgL
                if USE_MUSLE_EVENT else 0.0
            )

            SSC_station_mgL = (
                SSC_base_component_mgL
                + SSC_resus_component_mgL
                + SSC_event_component_mgL
            )
            # Station SSC component diagnostics.
            if np.any(valid):
                raw_conc_qw_mgL = float(np.average(conc_station[valid], weights=weights[valid]) * 1e6)
                eff_conc_qw_mgL = float(np.average(conc_eff_station[valid], weights=weights[valid]) * 1e6)
                max_conc_qw_mgL = float(np.average(conc_max_station[valid], weights=weights[valid]) * 1e6)

                raw_conc_max_mgL = float(np.nanmax(conc_station[valid]) * 1e6)
                eff_conc_max_mgL = float(np.nanmax(conc_eff_station[valid]) * 1e6)
                max_conc_max_mgL = float(np.nanmax(conc_max_station[valid]) * 1e6)

                Q_weight_sum_m3_s = float(np.nansum(weights[valid]))
            else:
                raw_conc_qw_mgL = np.nan
                eff_conc_qw_mgL = np.nan
                max_conc_qw_mgL = np.nan
                raw_conc_max_mgL = np.nan
                eff_conc_max_mgL = np.nan
                max_conc_max_mgL = np.nan
                Q_weight_sum_m3_s = np.nan
            if np.any(valid):
                # Sediment mass implied by effective concentration and flow volume.
                V_flow_valid_m3_day = weights[valid] * DT_SEC
                V_flow_sum_m3_day = float(np.nansum(V_flow_valid_m3_day))

                # Concentration using only flow volume as denominator.
                if V_flow_sum_m3_day > 0:
                    SSC_from_L_over_Q_mgL = float((L_station_t_day * 1e9) / (V_flow_sum_m3_day * 1000.0))
                else:
                    SSC_from_L_over_Q_mgL = np.nan
            else:
                V_flow_sum_m3_day = np.nan
                SSC_from_L_over_Q_mgL = np.nan

            if ENABLE_NUTRIENT_MODULE:
                hgs_nutrient_kg_m3 = cal_hgs_nutrient_cell_kg_m3(t)
                q_cell_m3_day = np.maximum(Qflow_m3_s[sid], 0.0) * DT_SEC
                if float(np.nansum(q_cell_m3_day)) <= 0.0 and Q_station_m3_day > 0.0:
                    q_peak_station = np.maximum(q_peak[sid], 0.0) if q_peak.size > 0 else np.zeros_like(q_cell_m3_day)
                    q_peak_sum = float(np.nansum(q_peak_station))
                    if q_peak_sum > 0.0:
                        q_cell_m3_day = Q_station_m3_day * q_peak_station / q_peak_sum
                    else:
                        q_cell_m3_day = np.full(sid.size, Q_station_m3_day / max(sid.size, 1), dtype=float)
                dissolved_n_load_kg_day = float(
                    np.nansum(hgs_nutrient_kg_m3[sid] * q_cell_m3_day)
                )

                if Q_station_m3_day > 0.0:
                    hgs_dissolved_n_mgL = (
                        dissolved_n_load_kg_day / Q_station_m3_day * 1000.0
                    )
                else:
                    hgs_dissolved_n_mgL = np.nan

                nutrient_er = nutrient_enrichment_ratio(SSC_station_mgL)
                sediment_bound_n_load_kg_day = (
                    max(L_station_t_day, 0.0)
                    * float(SOIL_N_CONC_KG_PER_TON)
                    * nutrient_er
                )
                if Q_station_m3_day > 0.0:
                    sediment_bound_n_mgL = (
                        sediment_bound_n_load_kg_day / Q_station_m3_day * 1000.0
                    )
                else:
                    sediment_bound_n_mgL = np.nan

                total_n_load_kg_day = (
                    dissolved_n_load_kg_day + sediment_bound_n_load_kg_day
                )
                if np.isfinite(hgs_dissolved_n_mgL) and np.isfinite(sediment_bound_n_mgL):
                    total_n_mgL = hgs_dissolved_n_mgL + sediment_bound_n_mgL
                else:
                    total_n_mgL = np.nan
            else:
                hgs_dissolved_n_mgL = np.nan
                dissolved_n_load_kg_day = np.nan
                sediment_bound_n_mgL = np.nan
                sediment_bound_n_load_kg_day = np.nan
                total_n_mgL = np.nan
                total_n_load_kg_day = np.nan
                nutrient_er = np.nan

            if idx < 9999:
                log(
                    f"[STA DIAG] idx={idx}, t={t:.0f}, n_cells={sid.size}, "
                    f"Q_raw_station_m3_s={Q_station_raw_m3_s:.3e}, "
                    f"Q_exchange_station_m3_s={Q_exchange_station_m3_s:.3e}, "
                    f"Q_eff_station_m3_s={Q_station_m3_s:.3e}, "
                    f"Q_eff_station_m3_day={Q_station_m3_day:.3e}, "
                    f"L_station_t_day={L_station_t_day:.3e}, "
                    f"SSC={SSC_station_mgL:.3f} mg/L, "
                    f"event={SSC_event_mgL:.3f}, "
                    f"event_sat={SSC_event_saturated_mgL:.3f}, "
                    f"base={SSC_base_component_mgL:.3f}, "
                    f"resus={SSC_resus_component_mgL:.3f}, "
                    f"event_w={SSC_event_component_mgL:.3f}, "
                    f"raw_qw={raw_conc_qw_mgL:.3f}, "
                    f"eff_qw={eff_conc_qw_mgL:.3f}, "
                    f"max_qw={max_conc_qw_mgL:.3f}, "
                    f"raw_max={raw_conc_max_mgL:.3f}, "
                    f"eff_max={eff_conc_max_mgL:.3f}, "
                    f"concmax_max={max_conc_max_mgL:.3f}, "
                    f"Q_weight_sum={Q_weight_sum_m3_s:.3e}"
                )
        else:
            rain_station_mean = np.nan
            rain_station_max = np.nan
            Q_station_raw_m3_s = np.nan
            Q_exchange_station_m3_s = np.nan
            Q_station_m3_s = np.nan
            Q_station_raw_m3_day = np.nan
            Q_exchange_station_m3_day = np.nan
            Q_station_m3_day = np.nan
            delta_Q_station_m3_day = np.nan
            exchange_effect_expected_m3_day = np.nan
            exchange_fraction_of_raw = np.nan
            q_station_m_s = np.nan
            old_qpeak_station_m3_s = np.nan
            old_qpeak_station_m_s = np.nan
            L_station_t_day = np.nan
            SSC_station_mgL = np.nan
            SSC_event_mgL = np.nan
            SSC_event_saturated_mgL = np.nan
            SSC_base_component_mgL = np.nan
            SSC_resus_component_mgL = np.nan
            SSC_event_component_mgL = np.nan
            Q_excess_m3_day = np.nan
            raw_conc_qw_mgL = np.nan
            eff_conc_qw_mgL = np.nan
            max_conc_qw_mgL = np.nan
            raw_conc_max_mgL = np.nan
            eff_conc_max_mgL = np.nan
            max_conc_max_mgL = np.nan
            Q_weight_sum_m3_s = np.nan
            V_flow_sum_m3_day = np.nan
            SSC_from_L_over_Q_mgL = np.nan
            hgs_dissolved_n_mgL = np.nan
            dissolved_n_load_kg_day = np.nan
            sediment_bound_n_mgL = np.nan
            sediment_bound_n_load_kg_day = np.nan
            total_n_mgL = np.nan
            total_n_load_kg_day = np.nan
            nutrient_er = np.nan

        station_ts.append({
            "time_sec": float(t),
            "step_index": int(idx),
            "n_cells": int(len(station_ids)),
            "A_station_m2": float(station_area_m2) if np.isfinite(station_area_m2) else np.nan,
            "rain_station_mean": float(rain_station_mean) if np.isfinite(rain_station_mean) else np.nan,
            "rain_station_max": float(rain_station_max) if np.isfinite(rain_station_max) else np.nan,
            "Q_station_raw_m3_s": float(Q_station_raw_m3_s) if np.isfinite(Q_station_raw_m3_s) else np.nan,
            "Q_exchange_station_m3_s": float(Q_exchange_station_m3_s) if np.isfinite(Q_exchange_station_m3_s) else np.nan,
            "Q_station_m3_s": float(Q_station_m3_s) if np.isfinite(Q_station_m3_s) else np.nan,
            "Q_station_raw_m3_day": float(Q_station_raw_m3_day) if np.isfinite(Q_station_raw_m3_day) else np.nan,
            "Q_exchange_station_m3_day": float(Q_exchange_station_m3_day) if np.isfinite(Q_exchange_station_m3_day) else np.nan,
            "Q_station_m3_day": float(Q_station_m3_day) if np.isfinite(Q_station_m3_day) else np.nan,
            "delta_Q_station_m3_day": float(delta_Q_station_m3_day) if np.isfinite(delta_Q_station_m3_day) else np.nan,
            "exchange_effect_expected_m3_day": float(exchange_effect_expected_m3_day) if np.isfinite(exchange_effect_expected_m3_day) else np.nan,
            "exchange_fraction_of_raw": float(exchange_fraction_of_raw) if np.isfinite(exchange_fraction_of_raw) else np.nan,
            "q_station_m_s": float(q_station_m_s) if np.isfinite(q_station_m_s) else np.nan,
            "old_qpeak_station_m3_s": float(old_qpeak_station_m3_s) if np.isfinite(old_qpeak_station_m3_s) else np.nan,
            "old_qpeak_station_m_s": float(old_qpeak_station_m_s) if np.isfinite(old_qpeak_station_m_s) else np.nan,
            "L_station_t_day": float(L_station_t_day) if np.isfinite(L_station_t_day) else np.nan,
            "SSC_station_mgL": float(SSC_station_mgL) if np.isfinite(SSC_station_mgL) else np.nan,
            "SSC_event_mgL": float(SSC_event_mgL) if np.isfinite(SSC_event_mgL) else np.nan,
            "SSC_event_saturated_mgL": float(SSC_event_saturated_mgL) if np.isfinite(SSC_event_saturated_mgL) else np.nan,
            "SSC_base_component_mgL": float(SSC_base_component_mgL) if np.isfinite(SSC_base_component_mgL) else np.nan,
            "SSC_resus_component_mgL": float(SSC_resus_component_mgL) if np.isfinite(SSC_resus_component_mgL) else np.nan,
            "SSC_event_component_mgL": float(SSC_event_component_mgL) if np.isfinite(SSC_event_component_mgL) else np.nan,
            "Q_excess_m3_day": float(Q_excess_m3_day) if np.isfinite(Q_excess_m3_day) else np.nan,
            "raw_conc_qw_mgL": float(raw_conc_qw_mgL) if np.isfinite(raw_conc_qw_mgL) else np.nan,
            "eff_conc_qw_mgL": float(eff_conc_qw_mgL) if np.isfinite(eff_conc_qw_mgL) else np.nan,
            "max_conc_qw_mgL": float(max_conc_qw_mgL) if np.isfinite(max_conc_qw_mgL) else np.nan,
            "raw_conc_max_mgL": float(raw_conc_max_mgL) if np.isfinite(raw_conc_max_mgL) else np.nan,
            "eff_conc_max_mgL": float(eff_conc_max_mgL) if np.isfinite(eff_conc_max_mgL) else np.nan,
            "max_conc_max_mgL": float(max_conc_max_mgL) if np.isfinite(max_conc_max_mgL) else np.nan,
            "Q_weight_sum_m3_s": float(Q_weight_sum_m3_s) if np.isfinite(Q_weight_sum_m3_s) else np.nan,
            "HGS_dissolved_N_mgL": float(hgs_dissolved_n_mgL) if np.isfinite(hgs_dissolved_n_mgL) else np.nan,
            "HGS_dissolved_N_kg_day": float(dissolved_n_load_kg_day) if np.isfinite(dissolved_n_load_kg_day) else np.nan,
            "sediment_bound_N_mgL": float(sediment_bound_n_mgL) if np.isfinite(sediment_bound_n_mgL) else np.nan,
            "sediment_bound_N_kg_day": float(sediment_bound_n_load_kg_day) if np.isfinite(sediment_bound_n_load_kg_day) else np.nan,
            "total_N_mgL": float(total_n_mgL) if np.isfinite(total_n_mgL) else np.nan,
            "total_N_kg_day": float(total_n_load_kg_day) if np.isfinite(total_n_load_kg_day) else np.nan,
            "nutrient_enrichment_ratio": float(nutrient_er) if np.isfinite(nutrient_er) else np.nan,
        })

    # ------------------ 8) write outputs ------------------
    df_station = pd.DataFrame(station_ts)
    station_csv = os.path.join(out_dir, "station_SSC_riverConcEff.csv")
    df_station.to_csv(station_csv, index=False)

    if ENABLE_NUTRIENT_MODULE:
        nutrient_columns = [
            "time_sec",
            "step_index",
            "n_cells",
            "rain_station_mean",
            "Q_station_m3_day",
            "L_station_t_day",
            "SSC_station_mgL",
            "HGS_dissolved_N_mgL",
            "HGS_dissolved_N_kg_day",
            "sediment_bound_N_mgL",
            "sediment_bound_N_kg_day",
            "total_N_mgL",
            "total_N_kg_day",
            "nutrient_enrichment_ratio",
        ]
        nutrient_csv = os.path.join(out_dir, "station_nutrient_total.csv")
        df_station[nutrient_columns].to_csv(nutrient_csv, index=False)
        log(f"[OUT] station nutrient output written to: {nutrient_csv}")

    if WRITE_CELL_SSC_OUTPUTS and cell_output_records:
        cell_index_csv = os.path.join(out_dir, "cell_SSC_outputs", "cell_SSC_output_index.csv")
        pd.DataFrame(cell_output_records).to_csv(cell_index_csv, index=False)
        log(f"[OUT] cell SSC output index written to: {cell_index_csv}")
    if not pest_mode:

        log(f"[MAP] delivered_series timesteps = {len(delivered_series)}")

        if plot_sediment_map_mode:
            log("[MAP] Plotting sediment maps...")
            for t, sed_arr in delivered_series.items():
                plot_shapes(t, out_dir, sed_arr)

        if plot_exchange_flux_map_mode:
            log("[MAP] Plotting exchange flux maps...")
            out_exchange_dir = os.path.join(out_dir, "exchange_maps")
            os.makedirs(out_exchange_dir, exist_ok=True)

            for t in delivered_series.keys():
                out_png = os.path.join(out_exchange_dir, f"exchange_map_{int(t)}.png")
                plot_exchange_map(t, out_png=out_png, dpi=300, mask_zero=True)

    log(f"[OUT] station SSC written to: {station_csv}")


    df_events = build_station_event_qpeak_table(
        df_station,
        q_col="q_station_m_s",
        Q_col="Q_station_m3_s",
        old_qpeak_col="old_qpeak_station_m3_s",
        old_qpeak_specific_col="old_qpeak_station_m_s",
        threshold=1e-8,
        min_gap_steps=1,
        min_event_steps=1
    )

    out_events = os.path.join(out_dir, "station_event_qpeak.csv")
    df_events.to_csv(out_events, index=False)
    log(f"[OUT] station event qpeak written to: {out_events}")

    if pest_mode:
        pest_txt = write_pest_txt_from_station_df(df_station, out_dir)
        log(f"[OUT] pest txt written to: {pest_txt}")

    return


def _gui_config_path():
    return os.path.join(APP_DIR, "Model_Config.txt")


def _gui_literal(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def _gui_load_config_values(config_path):
    ns = {}
    with open(config_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, config_path, "exec"), {}, ns)
    skip = {"__builtins__"}
    return {k: v for k, v in ns.items() if k not in skip and not k.startswith("__")}


def _gui_save_selected_values(config_path, updates):
    assign_re = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\s*(#.*)?)$")
    written = set()
    out_lines = []
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            m = assign_re.match(line.rstrip("\n"))
            if m and m.group(2) in updates:
                name = m.group(2)
                comment = m.group(5) or ""
                out_lines.append(f"{m.group(1)}{name}{m.group(3)}{_gui_literal(updates[name])}{comment}\n")
                written.add(name)
            else:
                out_lines.append(line)
    missing = [name for name in updates if name not in written]
    if missing:
        out_lines.append("\n# Added by GUI\n")
        for name in missing:
            out_lines.append(f"{name} = {_gui_literal(updates[name])}\n")
    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)


SWITCH_LABELS = {
    "timestamp_output_dir": "Create Timestamped Output Folder",
    "pest_mode": "Calibration / PEST Mode",
    "write_cell_ssc_outputs": "Write Cell-by-Cell SSC Diagnostics",
    "enable_daily_loop": "Enable Daily Sediment Loop",
    "plot_sediment_map_mode": "Export Sediment Maps",
    "plot_exchange_flux_map_mode": "Export Exchange-Flux Maps",
    "enable_nutrient_module": "Enable Nutrient Module",
    "nutrient_use_proxy_from_olf": "Read Nutrient From HGS OLF Output",
    "use_post_routing_conc": "Use Routed River Concentration",
    "use_dynamic_channel_mask": "Use Dynamic Channel Mask",
    "use_resuspension": "Enable Resuspension",
    "use_musle_event": "Enable Rainfall / Event Sediment",
}

PRIMARY_SWITCHES = [
    "enable_nutrient_module",
    "plot_sediment_map_mode",
    "plot_exchange_flux_map_mode",
    "pest_mode",
]

MORE_SWITCHES = [
    "timestamp_output_dir",
    "write_cell_ssc_outputs",
    "enable_daily_loop",
    "nutrient_use_proxy_from_olf",
    "use_post_routing_conc",
    "use_dynamic_channel_mask",
    "use_resuspension",
    "use_musle_event",
]

VISIBLE_SWITCHES = PRIMARY_SWITCHES + MORE_SWITCHES

GUI_DEFAULT_SWITCH_VALUES = {
    "enable_nutrient_module": True,
    "plot_sediment_map_mode": True,
    "plot_exchange_flux_map_mode": True,
    "pest_mode": False,
    "timestamp_output_dir": True,
    "write_cell_ssc_outputs": True,
    "enable_daily_loop": True,
    "nutrient_use_proxy_from_olf": True,
    "use_post_routing_conc": True,
    "use_dynamic_channel_mask": True,
    "use_resuspension": True,
    "use_musle_event": True,
}


def _gui_switch_label(name):
    return SWITCH_LABELS.get(name, name.replace("_", " ").title())


def _gui_bool_order(config_path, values):
    visible = [name for name in VISIBLE_SWITCHES if isinstance(values.get(name), bool)]
    extras = [
        name for name in sorted(values)
        if isinstance(values.get(name), bool) and name not in visible and name in SWITCH_LABELS
    ]
    return visible + extras


def launch_gui():
    import queue
    import subprocess
    import sys
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("HGS Sediment-Nutrient Module")
    root.geometry("1180x720")
    root.minsize(1040, 700)
    root.configure(bg="#f4f6f8")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("App.TFrame", background="#f4f6f8")
    style.configure("Panel.TLabelframe", background="#f4f6f8", padding=12)
    style.configure("Panel.TLabelframe.Label", background="#f4f6f8", foreground="#1f2933", font=("Segoe UI", 10, "bold"))
    style.configure("Title.TLabel", background="#f4f6f8", foreground="#111827", font=("Segoe UI", 18, "bold"))
    style.configure("Subtitle.TLabel", background="#f4f6f8", foreground="#5b6773", font=("Segoe UI", 9))
    style.configure("Status.TLabel", background="#f4f6f8", foreground="#374151", font=("Segoe UI", 9))
    style.configure("Field.TLabel", background="#f4f6f8", foreground="#1f2933", font=("Segoe UI", 9))
    style.configure("Switch.TCheckbutton", background="#f4f6f8", foreground="#111827", font=("Segoe UI", 9))
    style.map("Switch.TCheckbutton", background=[("active", "#f4f6f8")], foreground=[("active", "#111827")])
    style.configure("TButton", padding=(10, 5))
    style.configure("Run.TButton", padding=(14, 6), font=("Segoe UI", 9, "bold"))

    config_var = tk.StringVar(value=_gui_config_path())
    olf_var = tk.StringVar()
    chan_var = tk.StringVar()
    out_var = tk.StringVar()
    update_config_var = tk.BooleanVar(value=True)
    status_var = tk.StringVar(value="HGS Sediment-Nutrient Module: choose one input/config file. Numeric parameters stay inside that file.")
    bool_vars = {}
    bool_names = []
    process_ref = {"process": None}
    terminal_queue = queue.Queue()
    terminal_visible = tk.BooleanVar(value=False)
    terminal_button_text = tk.StringVar(value="Show Terminal")
    more_switches_visible = tk.BooleanVar(value=False)
    more_button_text = tk.StringVar(value="More")

    def append_terminal(text):
        terminal_text.configure(state="normal")
        terminal_text.insert("end", text)
        terminal_text.see("end")
        terminal_text.configure(state="disabled")

    def pump_terminal():
        try:
            while True:
                append_terminal(terminal_queue.get_nowait())
        except queue.Empty:
            pass
        root.after(100, pump_terminal)

    def load_from_config(show_message=False):
        cfg_path = config_var.get().strip()
        try:
            values = _gui_load_config_values(cfg_path)
        except Exception as exc:
            messagebox.showerror("Config error", str(exc))
            return
        olf_var.set(str(values.get("olf_path", "")))
        chan_var.set(str(values.get("chan_path", "")))
        out_var.set(str(values.get("out_dir", "")))
        rebuild_switches(cfg_path, values)
        status_var.set(f"Loaded input file: {cfg_path}")
        terminal_queue.put(f"[GUI] Loaded input file: {cfg_path}\n")
        if show_message:
            messagebox.showinfo("Loaded", "Input file loaded.")

    def handle_switch_change(name):
        map_keys = ("plot_sediment_map_mode", "plot_exchange_flux_map_mode")
        if name in map_keys and bool_vars.get(name) is not None and bool_vars[name].get():
            if bool_vars.get("pest_mode") is not None and bool_vars["pest_mode"].get():
                bool_vars["pest_mode"].set(False)
                terminal_queue.put("[GUI] Map export selected; Calibration / PEST mode was turned off.\n")
        elif name == "pest_mode" and bool_vars.get("pest_mode") is not None and bool_vars["pest_mode"].get():
            changed = False
            for key in map_keys:
                if bool_vars.get(key) is not None and bool_vars[key].get():
                    bool_vars[key].set(False)
                    changed = True
            if changed:
                terminal_queue.put("[GUI] Calibration / PEST mode selected; map exports were turned off.\n")

    def rebuild_switches(cfg_path, values):
        nonlocal bool_names
        for child in switches_inner.winfo_children():
            child.destroy()
        for child in more_switches_inner.winfo_children():
            child.destroy()
        bool_vars.clear()
        bool_names = _gui_bool_order(cfg_path, values)
        if not bool_names:
            ttk.Label(switches_inner, text="No boolean switches found in input file.").grid(row=0, column=0, sticky="w")
            return

        def add_switch(parent, idx, name, cols):
            var = tk.BooleanVar(value=bool(values.get(name)))
            bool_vars[name] = var
            row, col = divmod(idx, cols)
            cb = ttk.Checkbutton(
                parent,
                text=_gui_switch_label(name),
                variable=var,
                style="Switch.TCheckbutton",
                command=lambda n=name: handle_switch_change(n),
            )
            cb.grid(row=row, column=col, sticky="w", padx=12, pady=4)

        primary = [name for name in PRIMARY_SWITCHES if name in bool_names]
        more = [name for name in bool_names if name not in primary]
        for idx, name in enumerate(primary):
            add_switch(switches_inner, idx, name, 2)
        for idx, name in enumerate(more):
            add_switch(more_switches_inner, idx, name, 2)

    def browse_config():
        picked = filedialog.askopenfilename(
            title="Choose model input/config file",
            filetypes=[("Text config", "*.txt"), ("Python-style config", "*.py"), ("All files", "*.*")],
            initialdir=os.path.dirname(config_var.get()) if config_var.get() else None,
        )
        if picked:
            config_var.set(picked)
            load_from_config(show_message=False)

    def browse_file(var, title):
        picked = filedialog.askopenfilename(title=title)
        if picked:
            var.set(picked)

    def browse_dir(var, title):
        picked = filedialog.askdirectory(title=title)
        if picked:
            var.set(picked)

    def open_input_file():
        cfg_path = config_var.get().strip()
        if not os.path.exists(cfg_path):
            messagebox.showerror("Missing file", cfg_path)
            return
        os.startfile(cfg_path)

    def selected_updates():
        updates = {
            "olf_path": olf_var.get().strip(),
            "chan_path": chan_var.get().strip(),
            "out_dir": out_var.get().strip(),
        }
        for name in bool_names:
            if name in bool_vars:
                updates[name] = bool(bool_vars[name].get())
        return updates

    def save_selected(show_message=True):
        cfg_path = config_var.get().strip()
        try:
            _gui_save_selected_values(cfg_path, selected_updates())
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))
            return False
        status_var.set(f"Saved selected GUI values to: {cfg_path}")
        terminal_queue.put(f"[GUI] Saved selected values to: {cfg_path}\n")
        if show_message:
            messagebox.showinfo("Saved", "Selected file/path/switch values were saved to the input file.")
        return True

    def reader_thread(proc):
        try:
            for line in proc.stdout:
                terminal_queue.put(line)
        except Exception as exc:
            terminal_queue.put(f"[GUI] Log reader error: {exc}\n")
        code = proc.wait()
        terminal_queue.put(f"\n[GUI] Process finished with exit code {code}\n")
        process_ref["process"] = None

    def run_model(close_gui=False):
        cfg_path = config_var.get().strip()
        if update_config_var.get() and not save_selected(show_message=False):
            return
        if process_ref["process"] is not None:
            messagebox.showwarning("Already running", "A model process is already running.")
            return
        if not terminal_visible.get():
            show_terminal()
        env = os.environ.copy()
        env["GWSWI_CONFIG_PATH"] = cfg_path
        env["PYTHONUNBUFFERED"] = "1"
        if getattr(sys, "frozen", False):
            command = [sys.executable, "--nogui"]
            command_text = f"{sys.executable} --nogui"
            run_cwd = APP_DIR
        else:
            script_path = os.path.abspath(__file__)
            command = [sys.executable, script_path, "--nogui"]
            command_text = f"{sys.executable} {script_path} --nogui"
            run_cwd = APP_DIR
        terminal_queue.put(f"[GUI] Starting model with input file: {cfg_path}\n")
        terminal_queue.put(f"[GUI] Command: {command_text}\n\n")
        try:
            proc = subprocess.Popen(
                command,
                cwd=run_cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            messagebox.showerror("Run error", str(exc))
            return
        process_ref["process"] = proc
        threading.Thread(target=reader_thread, args=(proc,), daemon=True).start()
        status_var.set("Model started with selected input file.")
        if close_gui:
            root.destroy()

    def stop_model():
        proc = process_ref["process"]
        if proc is None:
            terminal_queue.put("[GUI] No running process to stop.\n")
            return
        proc.terminate()
        terminal_queue.put("[GUI] Stop requested.\n")

    def clear_terminal():
        terminal_text.configure(state="normal")
        terminal_text.delete("1.0", "end")
        terminal_text.configure(state="disabled")

    def update_window_size():
        height = 720
        if more_switches_visible.get():
            height += 150
        if terminal_visible.get():
            height += 140
        root.geometry(f"1180x{height}")

    def show_terminal():
        terminal_frame.grid()
        terminal_text.configure(height=18)
        terminal_visible.set(True)
        terminal_button_text.set("Hide Terminal")
        outer.rowconfigure(6, weight=1)
        update_window_size()

    def hide_terminal():
        terminal_frame.grid_remove()
        terminal_visible.set(False)
        terminal_button_text.set("Show Terminal")
        outer.rowconfigure(6, weight=0)
        update_window_size()

    def toggle_terminal():
        if terminal_visible.get():
            hide_terminal()
        else:
            show_terminal()

    def show_more_switches():
        more_switches_frame.grid()
        more_switches_visible.set(True)
        more_button_text.set("Less")
        update_window_size()

    def hide_more_switches():
        more_switches_frame.grid_remove()
        more_switches_visible.set(False)
        more_button_text.set("More")
        update_window_size()

    def toggle_more_switches():
        if more_switches_visible.get():
            hide_more_switches()
        else:
            show_more_switches()

    outer = ttk.Frame(root, padding=(18, 16, 18, 14), style="App.TFrame")
    outer.pack(fill="both", expand=True)

    title_label = ttk.Label(outer, text="HGS Sediment-Nutrient Module", style="Title.TLabel")
    title_label.grid(row=0, column=0, columnspan=3, sticky="w")
    subtitle = ttk.Label(
        outer,
        text="Configure HGS surface output, sediment routing, and nutrient post-processing from one input file.",
        style="Subtitle.TLabel",
    )
    subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 14))

    files_frame = ttk.LabelFrame(outer, text="Input Files and Output Location", style="Panel.TLabelframe")
    files_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 12))

    def add_file_row(parent, row, label, variable, browse_command):
        ttk.Label(parent, text=label, style="Field.TLabel", width=18).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(10, 8), pady=5)
        ttk.Button(parent, text="Browse", width=12, command=browse_command).grid(row=row, column=2, sticky="e", pady=5)

    add_file_row(files_frame, 0, "Input/Config File", config_var, browse_config)
    add_file_row(files_frame, 1, "HGS OLF File", olf_var, lambda: browse_file(olf_var, "Choose OLF file"))
    add_file_row(files_frame, 2, "HGS Channel File", chan_var, lambda: browse_file(chan_var, "Choose CHAN file"))
    add_file_row(files_frame, 3, "Output Folder", out_var, lambda: browse_dir(out_var, "Choose output folder"))
    files_frame.columnconfigure(1, weight=1)

    switches = ttk.LabelFrame(outer, text="Model Switches", style="Panel.TLabelframe")
    switches.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 12))
    switches.columnconfigure(0, weight=1)
    switches.columnconfigure(1, weight=0)
    switches_inner = ttk.Frame(switches, style="App.TFrame")
    switches_inner.grid(row=0, column=0, sticky="ew")
    switches_inner.columnconfigure(0, weight=1, minsize=520)
    switches_inner.columnconfigure(1, weight=1, minsize=520)
    ttk.Button(switches, textvariable=more_button_text, width=10, command=toggle_more_switches).grid(
        row=0, column=1, sticky="ne", padx=(12, 4), pady=(0, 0)
    )
    ttk.Label(
        switches,
        text="Note: Calibration / PEST mode only writes text/table outputs; maps are generated only when PEST mode is off.",
        style="Subtitle.TLabel",
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
    more_switches_frame = ttk.Frame(switches, style="App.TFrame")
    more_switches_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    more_switches_inner = ttk.Frame(more_switches_frame, style="App.TFrame")
    more_switches_inner.pack(fill="x", expand=True)
    more_switches_inner.columnconfigure(0, weight=1, minsize=520)
    more_switches_inner.columnconfigure(1, weight=1, minsize=520)

    control_frame = ttk.LabelFrame(outer, text="Run Control", style="Panel.TLabelframe")
    control_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 10))
    ttk.Checkbutton(
        control_frame,
        text="Save File Paths and Switches to the Input File Before Running",
        variable=update_config_var,
        style="Switch.TCheckbutton",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

    button_bar = ttk.Frame(control_frame, style="App.TFrame")
    button_bar.grid(row=1, column=0, columnspan=3, sticky="w")
    ttk.Button(button_bar, text="Run Model", width=16, style="Run.TButton", command=lambda: run_model(False)).pack(side="left", padx=(0, 8))
    ttk.Button(button_bar, text="Stop Model", width=14, command=stop_model).pack(side="left", padx=(0, 8))
    ttk.Button(button_bar, textvariable=terminal_button_text, width=16, command=toggle_terminal).pack(side="left", padx=(0, 8))
    ttk.Button(button_bar, text="Clear Terminal", width=16, command=clear_terminal).pack(side="left", padx=(0, 8))

    ttk.Label(outer, textvariable=status_var, style="Status.TLabel").grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 8))

    terminal_frame = ttk.LabelFrame(outer, text="Terminal Log", style="Panel.TLabelframe")
    terminal_frame.grid(row=6, column=0, columnspan=3, sticky="nsew")
    terminal_scroll = ttk.Scrollbar(terminal_frame, orient="vertical")
    terminal_text = tk.Text(
        terminal_frame,
        height=18,
        bg="#0f1720",
        fg="#d7dee8",
        insertbackground="#d7dee8",
        selectbackground="#23486b",
        font=("Consolas", 9),
        relief="flat",
        padx=10,
        pady=8,
        wrap="word",
        yscrollcommand=terminal_scroll.set,
        state="disabled",
    )
    terminal_scroll.configure(command=terminal_text.yview)
    terminal_text.pack(side="left", fill="both", expand=True)
    terminal_scroll.pack(side="right", fill="y")

    outer.columnconfigure(1, weight=1)
    outer.rowconfigure(6, weight=0)
    terminal_frame.grid_remove()
    more_switches_frame.grid_remove()

    pump_terminal()
    if config_var.get().strip():
        load_from_config(show_message=False)
    else:
        rebuild_switches("", GUI_DEFAULT_SWITCH_VALUES)
        status_var.set("No input file selected. Click Browse to choose a config file.")
        terminal_queue.put("[GUI] No input file selected. Click Browse to choose a config file.\n")
    root.mainloop()


if __name__ == "__main__":
    import sys
    if not ENABLE_GUI or "--nogui" in sys.argv or "--headless" in sys.argv:
        main()
    else:
        launch_gui()
