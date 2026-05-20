"""
questionnaire_utils.py — Krushi Mitra v2.0
===========================================
Converts farmer questionnaire answers + district environmental data
into [N, P, K, temperature, humidity, pH, rainfall] for ML prediction.

Key design principle:
  Rainfall feedback options display district-calibrated mm values so
  "Very low" in drought-prone Nuapada (~1000 mm/yr) means different
  actual mm than "Very low" in coastal Kendrapara (~1600 mm/yr).

Data sources:
  - final_environment_dataset.csv   → temperature, humidity per district
  - district_wise_rainfall_normal.csv → annual rainfall baseline per district
"""

import os
import csv

_BASE_DIR = os.path.dirname(__file__)
ENV_DATA_PATH      = os.path.join(_BASE_DIR, '..', 'data', 'final_environment_dataset.csv')
RAINFALL_NORM_PATH = os.path.join(_BASE_DIR, '..', 'data', 'district_wise_rainfall_normal.csv')

# ── Module-level cache ────────────────────────────────────────────────────────
_env_cache      = None   # district (lower) → row dict
_rainfall_cache = None   # district (lower) → annual mm float


# ═══════════════════════════════════════════════════════════════════════════════
#  QUESTIONNAIRE MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Nitrogen — leaf colour ────────────────────────────────────────────────────
N_LEAF_MAP = {
    "Yellowish":     20,    # severe N deficiency (chlorosis)
    "Light green":   50,    # mild deficiency
    "Healthy green": 80,    # adequate
    "Dark green":   120,    # high / excess N
}

# ── Phosphorus — early growth ─────────────────────────────────────────────────
P_GROWTH_BASE = {
    "Weak growth":         15,
    "Okay growth":         35,
    "Good growth":         60,
    "Very healthy growth": 80,
}

# ── Phosphorus — flowering delay (confirmatory) ───────────────────────────────
P_FLOWER_MOD = {
    "Yes":       -10,   # delayed = confirms low P
    "Sometimes":   0,
    "No":        +10,   # no delay = P adequate
}

# ── Potassium — drought response ──────────────────────────────────────────────
K_DROUGHT_BASE = {
    "Plants dry quickly":    20,
    "Plants become weak":    45,
    "Plants manage somehow": 80,
    "Plants stay healthy":  120,
}

# ── Potassium — tip-burn (confirmatory) ──────────────────────────────────────
K_TIPBURN_MOD = {
    "Yes, very often": -15,   # tip burn = classic K deficiency symptom
    "Sometimes":         0,
    "Rarely":          +10,
    "Never noticed":   +20,
}

# ── Rainfall — fraction of district monthly average ───────────────────────────
RAINFALL_FEEDBACK_FACTOR = {
    "Very low":        0.35,
    "Less than usual": 0.65,
    "Normal":          1.00,
    "Heavy":           1.45,
}

# ── pH — soil-type defaults ───────────────────────────────────────────────────
SOIL_PH_DEFAULTS = {
    "Red Soil":      5.8,
    "Black Soil":    7.2,
    "Alluvial Soil": 6.8,
    "Laterite Soil": 5.5,
    "Sandy Soil":    6.4,
    "Loamy Soil":    6.7,
    "Yellow Soil":   6.0,
    "Arid Soil":     8.0,
    "Mountain Soil": 6.0,
}

# ── Odisha district fallback (used when CSV unavailable) ─────────────────────
_ODISHA_DISTRICTS = [
    "Angul", "Balasore", "Bargarh", "Boudh", "Bhubaneswar",
    "Cuttack", "Deogarh", "Dhenkanal", "Gajapati", "Ganjam",
    "Jagatsinghpur", "Jajpur", "Jharsuguda", "Kalahandi",
    "Kandhamal", "Kendrapara", "Keonjhar", "Khordha", "Koraput",
    "Malkangiri", "Mayurbhanj", "Nabarangpur", "Nayagarh",
    "Nuapada", "Puri", "Rayagada", "Sambalpur", "Subarnapur",
    "Sundargarh",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def _find_col(headers, *keywords):
    """Return the first header whose name contains any of the keywords (case-insensitive)."""
    for kw in keywords:
        for h in headers:
            if kw.lower() in h.lower():
                return h
    return None


def _load_env_data():
    """
    Load final_environment_dataset.csv.
    Returns dict: lowercase_district → raw row dict.
    """
    global _env_cache
    if _env_cache is not None:
        return _env_cache

    data = {}
    if not os.path.exists(ENV_DATA_PATH):
        print(f"[questionnaire_utils] ENV file not found: {ENV_DATA_PATH}")
        _env_cache = data
        return data

    with open(ENV_DATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        dist_col = _find_col(headers, 'district')
        if not dist_col:
            print("[questionnaire_utils] No 'district' column found in ENV CSV.")
            _env_cache = data
            return data
        for row in reader:
            key = row[dist_col].strip().lower()
            data[key] = dict(row)

    _env_cache = data
    return data


def _load_rainfall_normals():
    """
    Load district_wise_rainfall_normal.csv.
    Returns dict: lowercase_district → annual_rainfall_mm (float).
    """
    global _rainfall_cache
    if _rainfall_cache is not None:
        return _rainfall_cache

    data = {}
    if not os.path.exists(RAINFALL_NORM_PATH):
        print(f"[questionnaire_utils] Rainfall normals file not found: {RAINFALL_NORM_PATH}")
        _rainfall_cache = data
        return data

    with open(RAINFALL_NORM_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        dist_col   = _find_col(headers, 'district')
        annual_col = _find_col(headers, 'annual', 'total', 'jan-dec', 'yearly')

        if not dist_col:
            print("[questionnaire_utils] No 'district' column in rainfall CSV.")
            _rainfall_cache = data
            return data

        for row in reader:
            district = row[dist_col].strip().lower()
            if annual_col:
                try:
                    data[district] = float(row[annual_col])
                except (ValueError, TypeError):
                    pass
            else:
                # If no annual column, sum all numeric columns (Jan–Dec)
                total = 0.0
                for col in headers:
                    if col == dist_col:
                        continue
                    try:
                        total += float(row[col])
                    except (ValueError, TypeError):
                        pass
                if total > 0:
                    data[district] = total

    _rainfall_cache = data
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — Location helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_states_and_districts() -> dict:
    """
    Build {state: [districts...]} from the environment CSV.
    Falls back to hardcoded Odisha list if CSV is unavailable or lacks a state column.

    Returns:
        dict mapping state name → sorted list of district names.
    """
    data = _load_env_data()

    if data:
        first_row = next(iter(data.values()))
        headers   = list(first_row.keys())
        state_col = _find_col(headers, 'state')
        dist_col  = _find_col(headers, 'district')

        if state_col and dist_col:
            result: dict[str, list] = {}
            for row in data.values():
                state    = row[state_col].strip()
                district = row[dist_col].strip()
                result.setdefault(state, [])
                if district not in result[state]:
                    result[state].append(district)
            # Sort districts within each state
            return {s: sorted(d) for s, d in sorted(result.items())}

    # Fallback
    return {"Odisha": sorted(_ODISHA_DISTRICTS)}


def get_district_env_data(district: str) -> dict:
    """
    Return {temperature, humidity, annual_rainfall} for a district.
    Falls back to Odisha average if the district is not found in the CSV.

    Args:
        district: e.g. "Cuttack"

    Returns:
        dict with keys: temperature (°C), humidity (%), annual_rainfall (mm)
    """
    data = _load_env_data()
    key  = district.strip().lower()
    row  = data.get(key)

    if row:
        headers  = list(row.keys())
        temp_col = _find_col(headers, 'temp')
        hum_col  = _find_col(headers, 'humid')
        rain_col = _find_col(headers, 'rain', 'rainfall', 'precipitation')

        try:
            return {
                "temperature":     float(row[temp_col])  if temp_col  else 27.0,
                "humidity":        float(row[hum_col])   if hum_col   else 65.0,
                "annual_rainfall": float(row[rain_col])  if rain_col  else 1400.0,
            }
        except (ValueError, TypeError):
            pass

    # Fallback: Odisha average
    return {"temperature": 27.0, "humidity": 65.0, "annual_rainfall": 1400.0}


def get_district_annual_rainfall(district: str) -> float:
    """
    Annual rainfall (mm) for a district.
    Uses district_wise_rainfall_normal.csv first, then falls back to
    final_environment_dataset.csv, then a 1400 mm Odisha average.

    Args:
        district: e.g. "Cuttack"

    Returns:
        Annual rainfall in mm as float.
    """
    normals = _load_rainfall_normals()
    key     = district.strip().lower()

    if key in normals:
        return normals[key]

    # Try env dataset
    env = get_district_env_data(district)
    return env.get("annual_rainfall", 1400.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — Estimation
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_rainfall_mm(feedback: str, district: str) -> float:
    """
    Convert rainfall feedback → estimated monthly rainfall (mm),
    calibrated to the district's average annual rainfall.

    E.g. for Koraput (annual ~1500 mm):
        monthly avg ≈ 125 mm
        "Very low"  → 0.35 × 125 ≈  44 mm
        "Normal"    → 1.00 × 125 = 125 mm
        "Heavy"     → 1.45 × 125 ≈ 181 mm

    For Nuapada (annual ~1000 mm):
        monthly avg ≈  83 mm
        "Very low"  → 0.35 ×  83 ≈  29 mm
        "Normal"    → 1.00 ×  83 =  83 mm

    Args:
        feedback: one of RAINFALL_FEEDBACK_FACTOR keys
        district: district name

    Returns:
        Monthly rainfall estimate in mm, clamped to [20, 300].
    """
    annual   = get_district_annual_rainfall(district)
    monthly  = annual / 12.0
    factor   = RAINFALL_FEEDBACK_FACTOR.get(feedback, 1.0)
    result   = round(monthly * factor, 1)
    return max(20.0, min(result, 300.0))


def get_rainfall_option_labels(district: str) -> dict:
    """
    Build dynamic option labels for the rainfall question,
    showing actual mm estimates for a given district.

    Returns:
        dict: {option_key: display_label}

    Example for Cuttack (~1200 mm/yr, monthly avg ~100 mm):
        "Very low"        → "Very low  (~35 mm/month)"
        "Less than usual" → "Less than usual  (~65 mm/month)"
        "Normal"          → "Normal  (~100 mm/month)"
        "Heavy"           → "Heavy  (~145 mm/month)"
    """
    annual  = get_district_annual_rainfall(district)
    monthly = annual / 12.0
    labels  = {}
    for opt, factor in RAINFALL_FEEDBACK_FACTOR.items():
        mm = round(monthly * factor)
        labels[opt] = f"{opt}  (~{mm} mm/month)"
    return labels


def estimate_params(
    soil_type: str,
    district:  str,
    ans_N:     str,
    ans_P1:    str,
    ans_P2:    str,
    ans_K1:    str,
    ans_K2:    str,
    ans_rain:  str,
) -> dict:
    """
    Master estimation function. Converts questionnaire answers + soil type +
    district data into the 7-feature vector expected by the ML model.

    Args:
        soil_type : detected soil, e.g. "Red Soil"
        district  : selected district, e.g. "Cuttack"
        ans_N     : leaf colour answer
        ans_P1    : early crop growth answer
        ans_P2    : flowering delay answer
        ans_K1    : drought response answer
        ans_K2    : leaf tip-burn answer
        ans_rain  : recent rainfall feedback

    Returns:
        dict with keys: N, P, K, ph, temperature, humidity, rainfall
    """
    # ── N ──────────────────────────────────────────────────────────────────────
    N = float(N_LEAF_MAP.get(ans_N, 60))

    # ── P ──────────────────────────────────────────────────────────────────────
    P_base = float(P_GROWTH_BASE.get(ans_P1, 40))
    P_mod  = float(P_FLOWER_MOD.get(ans_P2, 0))
    P      = max(5.0, min(P_base + P_mod, 145.0))

    # ── K ──────────────────────────────────────────────────────────────────────
    K_base = float(K_DROUGHT_BASE.get(ans_K1, 60))
    K_mod  = float(K_TIPBURN_MOD.get(ans_K2, 0))
    K      = max(5.0, min(K_base + K_mod, 205.0))

    # ── Environmental data (from district CSV) ─────────────────────────────────
    env         = get_district_env_data(district)
    temperature = env["temperature"]
    humidity    = env["humidity"]

    # ── pH (soil-type default) ─────────────────────────────────────────────────
    ph = SOIL_PH_DEFAULTS.get(soil_type, 6.5)

    # ── Rainfall (district-calibrated feedback) ────────────────────────────────
    rainfall = estimate_rainfall_mm(ans_rain, district)

    # ── Humidity correction for extreme rainfall deviation ─────────────────────
    annual  = get_district_annual_rainfall(district)
    monthly = annual / 12.0
    if rainfall < monthly * 0.50:
        humidity = max(humidity - 15.0, 20.0)
    elif rainfall > monthly * 1.30:
        humidity = min(humidity + 10.0, 95.0)

    return {
        "N":           round(N, 1),
        "P":           round(P, 1),
        "K":           round(K, 1),
        "ph":          round(ph, 2),
        "temperature": round(temperature, 1),
        "humidity":    round(humidity, 1),
        "rainfall":    round(rainfall, 1),
    }


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== States & Districts ===")
    sd = get_states_and_districts()
    for state, districts in sd.items():
        print(f"{state}: {len(districts)} districts → {districts[:5]}...")

    print("\n=== District ENV Data ===")
    for d in ["Cuttack", "Koraput", "Nuapada"]:
        print(f"{d}: {get_district_env_data(d)}")

    print("\n=== Annual Rainfall ===")
    for d in ["Cuttack", "Koraput", "Nuapada"]:
        print(f"{d}: {get_district_annual_rainfall(d):.0f} mm/yr")

    print("\n=== Rainfall Labels (Cuttack) ===")
    labels = get_rainfall_option_labels("Cuttack")
    for k, v in labels.items():
        print(f"  {v}")

    print("\n=== Full Param Estimation ===")
    params = estimate_params(
        soil_type = "Red Soil",
        district  = "Cuttack",
        ans_N     = "Light green",
        ans_P1    = "Okay growth",
        ans_P2    = "Sometimes",
        ans_K1    = "Plants become weak",
        ans_K2    = "Sometimes",
        ans_rain  = "Normal",
    )
    for k, v in params.items():
        print(f"  {k}: {v}")
