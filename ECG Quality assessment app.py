import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# ECG QUALITY ASSESSMENT USING AI
# Streamlit Web App
# ============================================================

st.set_page_config(
    page_title="ECG Quality Assessment using AI",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ ECG Machine Quality Assessment using AI")
st.caption(
    "This prototype evaluates ECG signal quality as Good, "
    "Acceptable, or Poor. It is NOT a disease-diagnosis system."
)

# ------------------------------------------------------------
# 1. ECG filtering
# ------------------------------------------------------------
def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0):
    nyquist = 0.5 * fs
    highcut = min(highcut, nyquist - 1.0)

    if highcut <= lowcut:
        return signal.copy()

    b, a = butter(
        4,
        [lowcut / nyquist, highcut / nyquist],
        btype="band"
    )

    padlen = min(len(signal) - 1, 3 * max(len(a), len(b)))
    if padlen <= 1:
        return signal.copy()

    return filtfilt(b, a, signal, padlen=padlen)


# ------------------------------------------------------------
# 2. Feature extraction
# ------------------------------------------------------------
def extract_features(signal, fs):
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 100:
        raise ValueError("ECG signal is too short. Please provide at least 100 samples.")

    x = x - np.median(x)

    # Filtered signal
    filtered = bandpass_filter(x, fs)

    # Time-domain features
    mean_val = np.mean(x)
    std_val = np.std(x) + 1e-8
    rms_val = np.sqrt(np.mean(x ** 2))
    p2p_val = np.ptp(x)
    peak_abs = np.max(np.abs(x)) + 1e-8
    crest_factor = peak_abs / rms_val if rms_val > 1e-8 else 0.0

    # Difference / roughness feature
    diff_std = np.std(np.diff(x))
    roughness = diff_std / std_val

    # Very large outlier/saturation proportion
    z = np.abs((x - np.mean(x)) / std_val)
    outlier_ratio = np.mean(z > 5)

    # Frequency-domain features
    freqs, power = welch(x, fs=fs, nperseg=min(1024, len(x)))

    total_mask = (freqs >= 0.5) & (freqs <= min(40.0, fs / 2 - 1))
    baseline_mask = freqs < 0.5
    high_mask = freqs > 40.0

    total_power = np.sum(power[total_mask]) + 1e-12
    baseline_power = np.sum(power[baseline_mask])
    high_power = np.sum(power[high_mask])

    baseline_ratio = baseline_power / total_power
    high_frequency_ratio = high_power / total_power

    # Noise estimate from raw - filtered
    residual = x - filtered
    noise_ratio = np.std(residual) / (np.std(filtered) + 1e-8)

    # SNR estimate in dB
    snr_db = 20 * np.log10((np.std(filtered) + 1e-8) /
                           (np.std(residual) + 1e-8))

    return {
        "mean": mean_val,
        "std": std_val,
        "rms": rms_val,
        "peak_to_peak": p2p_val,
        "crest_factor": crest_factor,
        "roughness": roughness,
        "outlier_ratio": outlier_ratio,
        "baseline_ratio": baseline_ratio,
        "high_frequency_ratio": high_frequency_ratio,
        "noise_ratio": noise_ratio,
        "snr_db": snr_db
    }


FEATURE_NAMES = [
    "mean",
    "std",
    "rms",
    "peak_to_peak",
    "crest_factor",
    "roughness",
    "outlier_ratio",
    "baseline_ratio",
    "high_frequency_ratio",
    "noise_ratio",
    "snr_db"
]


# ------------------------------------------------------------
# 3. Synthetic ECG generator for an academic demo model
# ------------------------------------------------------------
def gaussian(t, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


def generate_clean_ecg(fs=250, duration=10, heart_rate=72, seed=None):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fs)
    beat_period = 60.0 / heart_rate

    signal = np.zeros_like(t)

    beat_times = np.arange(-1.0, duration + 1.0, beat_period)

    for beat in beat_times:
        # Simple P-QRS-T morphology
        signal += gaussian(t, beat - 0.20, 0.045, 0.12)   # P
        signal += gaussian(t, beat - 0.045, 0.012, -0.15) # Q
        signal += gaussian(t, beat,        0.010, 1.00)   # R
        signal += gaussian(t, beat + 0.040, 0.014, -0.25) # S
        signal += gaussian(t, beat + 0.28, 0.070, 0.35)   # T

    # Tiny physiological-looking variation
    signal *= 1 + 0.02 * np.sin(2 * np.pi * 0.15 * t)
    signal += 0.005 * rng.normal(size=len(t))

    return t, signal


def add_noise(signal, fs, noise_level, seed=None):
    rng = np.random.default_rng(seed)
    t = np.arange(len(signal)) / fs

    # White/muscle-like noise
    noise = noise_level * rng.normal(size=len(signal))

    # Baseline wandering
    baseline = (
        0.10 * noise_level * np.sin(2 * np.pi * 0.25 * t)
        + 0.05 * noise_level * np.sin(2 * np.pi * 0.35 * t)
    )

    # Power-line-like interference
    powerline = 0.08 * noise_level * np.sin(2 * np.pi * 50 * t)

    return signal + noise + baseline + powerline


@st.cache_resource
def train_ai_model():
    X = []
    y = []

    labels = {
        "Good": np.linspace(0.005, 0.02, 120),
        "Acceptable": np.linspace(0.025, 0.06, 120),
        "Poor": np.linspace(0.07, 0.20, 120)
    }

    sample_id = 0

    for label, levels in labels.items():
        for noise_level in levels:
            hr = np.random.default_rng(sample_id).uniform(60, 90)

            _, clean = generate_clean_ecg(
                fs=250,
                duration=8,
                heart_rate=hr,
                seed=sample_id
            )

            noisy = add_noise(
                clean,
                fs=250,
                noise_level=float(noise_level),
                seed=sample_id
            )

            features = extract_features(noisy, 250)
            X.append([features[name] for name in FEATURE_NAMES])
            y.append(label)

            sample_id += 1

    X = np.asarray(X)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Good", "Acceptable", "Poor"]
    )

    return model, accuracy, cm


model, model_accuracy, model_cm = train_ai_model()


# ------------------------------------------------------------
# 4. Heart-rate estimation
# ------------------------------------------------------------
def estimate_heart_rate(signal, fs):
    filtered = bandpass_filter(signal, fs)

    distance = max(int(0.35 * fs), 1)
    prominence = max(0.30 * np.std(filtered), 0.01)

    peaks, _ = find_peaks(
        filtered,
        distance=distance,
        prominence=prominence
    )

    if len(peaks) < 2:
        return None, peaks

    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.30) & (rr < 2.0)]

    if len(rr) == 0:
        return None, peaks

    heart_rate = 60.0 / np.median(rr)
    return heart_rate, peaks


def quality_score(label, probabilities):
    base = {
        "Good": 95,
        "Acceptable": 72,
        "Poor": 35
    }[label]

    confidence = max(probabilities.values())

    # Confidence is only used to make the displayed score reflect
    # the model's certainty; it is not a clinical measurement.
    score = base * (0.75 + 0.25 * confidence)

    return float(np.clip(score, 0, 100))


# ------------------------------------------------------------
# 5. Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("ECG Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload ECG CSV",
    type=["csv"]
)

fs = st.sidebar.number_input(
    "Sampling frequency (Hz)",
    min_value=50,
    max_value=2000,
    value=250,
    step=10
)

st.sidebar.markdown(
    "**CSV format:** one numeric column containing ECG samples. "
    "If multiple numeric columns are present, you can select one."
)


# ------------------------------------------------------------
# 6. Load data
# ------------------------------------------------------------
ecg_source = "Demo ECG"

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_columns:
            st.error("No numeric ECG column was found in the CSV.")
            st.stop()

        if len(numeric_columns) == 1:
            selected_column = numeric_columns[0]
        else:
            selected_column = st.sidebar.selectbox(
                "Select ECG signal column",
                numeric_columns
            )

        signal = df[selected_column].dropna().to_numpy(dtype=float)
        ecg_source = f"Uploaded: {selected_column}"

    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()
else:
    duration = 10
    demo_hr = 72

    demo_t, clean_demo = generate_clean_ecg(
        fs=int(fs),
        duration=duration,
        heart_rate=demo_hr,
        seed=123
    )

    demo_type = st.sidebar.selectbox(
        "Demo signal quality",
        ["Good", "Acceptable", "Poor"]
    )

    level_map = {
        "Good": 0.012,
        "Acceptable": 0.045,
        "Poor": 0.12
    }

    signal = add_noise(
        clean_demo,
        fs=int(fs),
        noise_level=level_map[demo_type],
        seed=456
    )

# Limit extremely large files for browser performance
max_samples = int(fs * 60)

if len(signal) > max_samples:
    st.warning(
        "The file is longer than 60 seconds, so only the first 60 seconds "
        "are analyzed in this prototype."
    )
    signal = signal[:max_samples]

if len(signal) < 100:
    st.error("The ECG signal must contain at least 100 samples.")
    st.stop()


# ------------------------------------------------------------
# 7. Analysis
# ------------------------------------------------------------
features = extract_features(signal, fs)
feature_vector = np.array([[features[name] for name in FEATURE_NAMES]])

prediction = model.predict(feature_vector)[0]
probability_array = model.predict_proba(feature_vector)[0]
probabilities = dict(zip(model.classes_, probability_array))

score = quality_score(prediction, probabilities)
heart_rate, peaks = estimate_heart_rate(signal, fs)

filtered_signal = bandpass_filter(signal, fs)
time = np.arange(len(signal)) / fs


# ------------------------------------------------------------
# 8. Results
# ------------------------------------------------------------
st.subheader("ECG Quality Assessment")

col1, col2, col3, col4 = st.columns(4)

col1.metric("AI Quality", prediction)
col2.metric("Quality Score", f"{score:.1f}/100")
col3.metric(
    "Estimated Heart Rate",
    f"{heart_rate:.1f} BPM" if heart_rate is not None else "Not detected"
)
col4.metric("Estimated SNR", f"{features['snr_db']:.1f} dB")


if prediction == "Good":
    st.success(
        "The AI classifies this ECG recording as GOOD QUALITY. "
        "The signal appears suitable for further analysis from a signal-quality perspective."
    )
elif prediction == "Acceptable":
    st.warning(
        "The AI classifies this ECG recording as ACCEPTABLE QUALITY. "
        "Some artifacts are present, but the signal may still be usable."
    )
else:
    st.error(
        "The AI classifies this ECG recording as POOR QUALITY. "
        "Check electrode placement, patient movement, and electrical interference, "
        "then consider repeating the recording."
    )

st.progress(int(score))


# ------------------------------------------------------------
# 9. ECG waveform plots
# ------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.markdown("### Original ECG")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(time, signal, linewidth=1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Raw ECG Signal")
    ax1.grid(alpha=0.25)
    st.pyplot(fig1, clear_figure=True)

with right:
    st.markdown("### Filtered ECG")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(time, filtered_signal, linewidth=1)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("0.5–40 Hz Band-pass Filtered ECG")
    ax2.grid(alpha=0.25)
    st.pyplot(fig2, clear_figure=True)


# ------------------------------------------------------------
# 10. Detected R-peaks
# ------------------------------------------------------------
st.markdown("### R-Peak Detection")

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(time, filtered_signal, linewidth=1)
if len(peaks) > 0:
    ax3.scatter(
        time[peaks],
        filtered_signal[peaks],
        s=35,
        zorder=3
    )
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Amplitude")
ax3.set_title("Detected R-Peaks")
ax3.grid(alpha=0.25)
st.pyplot(fig3, clear_figure=True)


# ------------------------------------------------------------
# 11. Features
# ------------------------------------------------------------
st.markdown("### Extracted Signal-Quality Features")

feature_table = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Value": [features[name] for name in FEATURE_NAMES]
})

st.dataframe(feature_table, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# 12. AI probabilities
# ------------------------------------------------------------
st.markdown("### AI Classification Probabilities")

prob_table = pd.DataFrame({
    "Quality Class": list(probabilities.keys()),
    "Probability": [round(v * 100, 2) for v in probabilities.values()]
})

st.dataframe(prob_table, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# 13. Model information
# ------------------------------------------------------------
with st.expander("AI Model Information"):
    st.write(
        f"Random Forest model accuracy on the internally generated demo "
        f"test set: **{model_accuracy * 100:.2f}%**"
    )

    st.write(
        "The demo model is trained on synthetic ECG signals with controlled "
        "noise levels. For a research-quality project, replace this training "
        "set with a labeled ECG signal-quality dataset."
    )

    cm_df = pd.DataFrame(
        model_cm,
        index=["Actual Good", "Actual Acceptable", "Actual Poor"],
        columns=["Predicted Good", "Predicted Acceptable", "Predicted Poor"]
    )

    st.write("Confusion matrix:")
    st.dataframe(cm_df, use_container_width=True)


# ------------------------------------------------------------
# 14. Project disclaimer
# ------------------------------------------------------------
st.divider()
st.caption(
    "Academic prototype only. This software assesses signal quality and "
    "does not diagnose cardiac disease or replace a clinician."
)
