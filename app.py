import streamlit as st
import numpy as np
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="icon.png",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
}

.block-container {
    max-width: 1150px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

html,
body,
[class*="css"],
.stApp,
.stMarkdown,
.stText,
label,
button,
input,
textarea {
    font-family:
        "JetBrains Mono",
        "Cascadia Code",
        "Fira Code",
        "Consolas",
        monospace !important;
}

h1 {
    font-family:
        "JetBrains Mono",
        "Cascadia Code",
        "Fira Code",
        monospace !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: #f0f6fc !important;
    letter-spacing: -1px;
}

h2,
h3 {
    font-family:
        "JetBrains Mono",
        "Cascadia Code",
        "Fira Code",
        monospace !important;
    color: #f0f6fc !important;
    font-weight: 600 !important;
}

p {
    color: #8b949e !important;
    line-height: 1.7;
}

section[data-testid="stSidebar"] {
    background-color: #080c12;
    border-right: 1px solid #21262d;
}

section[data-testid="stSidebar"] * {
    font-family:
        "JetBrains Mono",
        "Cascadia Code",
        "Fira Code",
        "Consolas",
        monospace !important;
}

button[data-baseweb="tab"] {
    font-family:
        "JetBrains Mono",
        "Cascadia Code",
        monospace !important;
    color: #8b949e !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #f0f6fc !important;
}

.stSlider label,
.stSlider label p,
.stSlider label div,
.stSlider [data-testid="stMarkdownContainer"],
.stSlider [data-testid="stMarkdownContainer"] p {
    color: #0fffff !important;
    font-family: "JetBrains Mono", "Cascadia Code", "Fira Code", "Consolas", monospace !important;
    font-weight: 500 !important;
}
}

.stSlider [data-testid="stMarkdownContainer"] {
    color: #8b949e !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background-color: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 8px;
    font-family:
        "JetBrains Mono",
        "Cascadia Code",
        monospace !important;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #21262d;
    border-color: #58a6ff;
    color: #ffffff;
}

button[kind="primary"] {
    background-color: #238636 !important;
    border: 1px solid #2ea043 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

button[kind="primary"]:hover {
    background-color: #2ea043 !important;
}

input {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    font-family:
        "JetBrains Mono",
        "Cascadia Code",
        monospace !important;
}

div[data-testid="stExpander"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
}

div[data-testid="stAlert"] {
    border-radius: 8px;
    font-family:
        "JetBrains Mono",
        "Cascadia Code",
        monospace !important;
}

hr {
    border-color: #21262d !important;
}

.stCaption {
    color: #6e7681 !important;
    font-family:
        "JetBrains Mono",
        "Cascadia Code",
        monospace !important;
}

code {
    font-family:
        "JetBrains Mono",
        "Cascadia Code",
        "Fira Code",
        monospace !important;
    background-color: #161b22 !important;
    color: #79c0ff !important;
}

div[data-testid="stSuccess"] {
    background-color: #0d2818;
    border: 1px solid #238636;
    border-radius: 10px;
}

div[data-testid="stError"] {
    background-color: #2d1117;
    border: 1px solid #da3633;
    border-radius: 10px;
}

div[data-testid="stInfo"] {
    background-color: #0c1929;
    border: 1px solid #1f6feb;
    border-radius: 10px;
}

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #0d1117;
}

::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #484f58;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 8])

with col1:
    st.image("icon.png", width=70)

with col2:
    st.title("Breast Cancer Prediction Assistant")

st.markdown(
    """
    This application uses a **machine learning model** to classify
    a cell-sample measurement as **Benign** or **Malignant**.

    **Please enter values exactly as shown in your laboratory/pathology report.**

    Do not estimate the values yourself.
    """
)

with st.sidebar:
    st.header("ℹ️ How to use")

    st.markdown(
        """
        ### Steps

        **1.** Keep your laboratory/pathology report ready.

        **2.** Enter the measurements in the sections below.

        **3.** Use the sliders to enter the reported values.

        **4.** Click **Predict**.

        **5.** Discuss the actual laboratory/pathology results
        with a qualified healthcare professional.

        ---
        """
    )

    st.info(
        "The model result is an educational prediction and "
        "is NOT a medical diagnosis."
    )

    st.divider()

    st.header("🤖 Model")

    st.write("Support Vector Classifier")
    st.write("Kernel: RBF")

    st.divider()

    st.caption(
        "Built using Python, Scikit-learn and Streamlit."
    )

base_features = [
    (
        "radius",
        "Radius",
        "Average distance from the center to the edge of the cell.",
        0.0,
        35.0,
        0.1
    ),
    (
        "texture",
        "Texture",
        "Variation in grayscale values of the cell.",
        0.0,
        40.0,
        0.1
    ),
    (
        "perimeter",
        "Perimeter",
        "Length of the cell boundary.",
        0.0,
        200.0,
        0.1
    ),
    (
        "area",
        "Area",
        "Area covered by the cell.",
        0.0,
        3000.0,
        1.0
    ),
    (
        "smoothness",
        "Smoothness",
        "Variation in the smoothness of the cell boundary.",
        0.0,
        0.3,
        0.001
    ),
    (
        "compactness",
        "Compactness",
        "How compact the cell shape is.",
        0.0,
        0.5,
        0.001
    ),
    (
        "concavity",
        "Concavity",
        "Severity of indentations in the cell boundary.",
        0.0,
        0.5,
        0.001
    ),
    (
        "concave_points",
        "Concave Points",
        "Number/severity of concave portions of the boundary.",
        0.0,
        0.3,
        0.001
    ),
    (
        "symmetry",
        "Symmetry",
        "How symmetrical the cell shape is.",
        0.0,
        0.5,
        0.001
    ),
    (
        "fractal_dimension",
        "Fractal Dimension",
        "Complexity of the cell boundary.",
        0.0,
        0.2,
        0.001
    )
]

groups = [
    (
        "📊 Average Measurements",
        "_mean",
        "Average measurements calculated from the cell sample."
    ),
    (
        "📈 Measurement Variation",
        "_se",
        "Standard Error (SE) describing variation in the measurements."
    ),
    (
        "🔬 Largest Measurements",
        "_worst",
        "Largest or most extreme measurements found in the sample."
    )
]

inputs = {}

with st.form("prediction_form"):
    tabs = st.tabs(
        [group[0] for group in groups]
    )

    for tab, (group_name, suffix, description) in zip(
        tabs,
        groups
    ):
        with tab:
            st.subheader(group_name)
            st.caption(description)
            st.write("")

            cols = st.columns(2)

            for i, (
                feat_key,
                label,
                help_text,
                min_value,
                max_value,
                step
            ) in enumerate(base_features):

                full_key = f"{feat_key}{suffix}"

                with cols[i % 2]:
                    inputs[full_key] = st.slider(
                        label,
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=float(min_value),
                        step=float(step),
                        help=help_text,
                        key=full_key
                    )

    st.write("")

    submitted = st.form_submit_button(
        "🔍 Predict",
        type="primary",
        use_container_width=True
    )

if submitted:
    ordered_columns = (
        [
            f"{feat_key}_mean"
            for feat_key, *_ in base_features
        ]
        +
        [
            f"{feat_key}_se"
            for feat_key, *_ in base_features
        ]
        +
        [
            f"{feat_key}_worst"
            for feat_key, *_ in base_features
        ]
    )

    if all(
        value == 0.0
        for value in inputs.values()
    ):
        st.warning(
            "⚠️ Please enter the measurements from your report "
            "before predicting."
        )
    else:
        data = np.array(
            [
                [
                    inputs[column]
                    for column in ordered_columns
                ]
            ]
        )

        try:
            data_scaled = scaler.transform(data)
            prediction = model.predict(data_scaled)

            st.divider()
            st.subheader("Prediction Result")

            if prediction[0] == 1:
                st.error(
                    "🔴 Model Prediction: MALIGNANT"
                )

                st.write(
                    """
                    This machine-learning model classified this
                    sample as **malignant**.

                    This is just for my self educational model prediction.
                    """
                )
            else:
                st.success(
                    "🟢 Model Prediction: BENIGN"
                )

                st.write(
                    """
                    This machine-learning model classified this
                    sample as **benign**.

                    This is just for my self educational model prediction.
                    """
                )

        except Exception as e:
            st.error(
                f"Prediction failed: {e}"
            )

st.divider()

st.caption(
    """
    Made with love by Arpan
    """
)