import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.title("AI Driven Ransomware Detection and Family Classification")



xgb_class = joblib.load("xgb_class.pkl")
xgb_family = joblib.load("xgb_family.pkl")
scaler_fam = joblib.load("scaler_fam.pkl")
le_fam = joblib.load("le_fam.pkl")

encoders = joblib.load("feature_encoders.pkl")

df = pd.read_csv("dataset.csv")

label_map = {0: "Benign", 1: "Ransomware"}

feature_order = [
    "DllCharacteristics",
    "files_malicious",
    "files_suspicious",
    "files_text",
    "SizeOfImage",
    "processes_malicious",
    "processes_monitored",
    "apis",
    "AddressOfEntryPoint",
    "OperatingSystemVersion",
    "rdata_SizeOfRawData",
    "address_of_ne_header",
    "rdata_VirtualSize",
    "total_procsses",
    "network_http"
]


if "detected" not in st.session_state:
    st.session_state.detected = False


st.header("Step 1: Detection")

det_features = [
    "network_http",
    "files_unknown",
    "total_processes",
    "network_dns",
    "files_suspicious",
    "registry_read",
    "processes_monitored",
    "registry_total",
    "files_malicious",
    "processes_malicious"
]

det_values = []

for f in det_features:
    val = st.number_input(f, value=0.0, key="det_" + f)
    det_values.append(val)

if st.button("Detect Ransomware"):

    det_array = np.array(det_values).reshape(1, -1)

    pred_det = xgb_class.predict(det_array)[0]

    if pred_det == 1:
        st.error("⚠ Ransomware Detected")
        st.session_state.detected = True
    else:
        st.success("✔ File is Benign")
        st.session_state.detected = False


if st.session_state.detected:

    st.header("Step 2: Family Classification")

    fam_inputs = {}


    fam_inputs["AddressOfEntryPoint"] = st.selectbox(
        "AddressOfEntryPoint",
        sorted(df["AddressOfEntryPoint"].unique())
    )

    fam_inputs["apis"] = st.number_input("apis", value=0.0)

    fam_inputs["processes_monitored"] = st.number_input(
        "processes_monitored", value=0.0
    )

    fam_inputs["processes_malicious"] = st.number_input(
        "processes_malicious", value=0.0
    )

    fam_inputs["SizeOfImage"] = st.selectbox(
        "SizeOfImage",
        sorted(df["SizeOfImage"].unique())
    )

    fam_inputs["files_text"] = st.number_input("files_text", value=0.0)

    fam_inputs["files_suspicious"] = st.number_input(
        "files_suspicious", value=0.0
    )

    fam_inputs["files_malicious"] = st.number_input(
        "files_malicious", value=0.0
    )

    fam_inputs["DllCharacteristics"] = st.selectbox(
        "DllCharacteristics",
        sorted(df["DllCharacteristics"].unique())
    )

    fam_inputs["OperatingSystemVersion"] = st.selectbox(
        "OperatingSystemVersion",
        sorted(df["OperatingSystemVersion"].unique())
    )

    fam_inputs["rdata_SizeOfRawData"] = st.selectbox(
        "rdata_SizeOfRawData",
        sorted(df["rdata_SizeOfRawData"].unique())
    )

    fam_inputs["address_of_ne_header"] = st.selectbox(
        "address_of_ne_header",
        sorted(df["address_of_ne_header"].unique())
    )

    fam_inputs["rdata_VirtualSize"] = st.selectbox(
        "rdata_VirtualSize",
        sorted(df["rdata_VirtualSize"].unique())
    )

    fam_inputs["total_procsses"] = st.number_input(
    "total_procsses", value=0.0
    )

    fam_inputs["network_http"] = st.number_input(
        "network_http", value=0.0
    )

    if st.button("Classify Family"):

        fam_df = pd.DataFrame([fam_inputs])

        fam_df = fam_df[feature_order]

        for col in fam_df.columns:
            if col in encoders:
                try:
                    fam_df[col] = encoders[col].transform(fam_df[col])
                except Exception as e:
                    st.error(f"Encoding error in {col}: {e}")
                    fam_df[col] = 0

        fam_array = fam_df.values

        fam_scaled = scaler_fam.transform(fam_array)

        pred = xgb_family.predict(fam_scaled)[0]

        family = le_fam.inverse_transform([pred])[0]

        st.success(f"Predicted Ransomware Family: {family}")