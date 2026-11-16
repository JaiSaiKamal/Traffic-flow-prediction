from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATASET_PATH = APP_DIR / "Metro_Interstate_Traffic_Volume.csv"
MODEL_FEATURES = ["hour", "day_of_week", "traffic_volume", "average_speed"]
FLOW_FEATURES = [
    "hour",
    "day_of_week",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
]

st.set_page_config(
    page_title="Traffic Flow | Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #eaf1f8;
        --muted: #9aacc0;
        --line: rgba(159, 185, 210, .16);
        --panel: rgba(17, 30, 45, .78);
        --teal: #57e0c1;
        --blue: #70a9ff;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, rgba(32, 111, 130, .22), transparent 34%),
            radial-gradient(ellipse at 5% 30%, rgba(46, 83, 145, .16), transparent 32%),
            #08111b;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    .block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
    h1 { letter-spacing: -.045em; }
    h2, h3 { letter-spacing: -.025em; }
    p, label, .stCaption { color: var(--muted); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 20px;
    }
    [data-testid="stMetricLabel"] { color: #a9bacb; }
    [data-testid="stMetricValue"] { color: var(--ink); }
    div[data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid var(--line); }
    button[data-baseweb="tab"] {
        color: var(--muted);
        border-radius: 10px 10px 0 0;
        padding: 12px 18px;
    }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--teal); }
    div[data-baseweb="tab-highlight"] { background-color: var(--teal); }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 38px 42px;
        margin: 8px 0 26px;
        border: 1px solid rgba(126, 202, 221, .24);
        border-radius: 24px;
        background:
            radial-gradient(circle at 88% 30%, rgba(87, 224, 193, .18), transparent 23%),
            linear-gradient(115deg, rgba(20, 42, 61, .97), rgba(11, 25, 39, .94));
        box-shadow: 0 24px 70px rgba(0, 0, 0, .22);
    }
    .hero:after {
        content: '〰';
        position: absolute;
        right: 6%;
        top: 10%;
        color: rgba(87, 224, 193, .12);
        font-size: 150px;
        line-height: 1;
        transform: rotate(-15deg);
    }
    .eyebrow {
        color: var(--teal);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .18em;
        text-transform: uppercase;
    }
    .hero h1 {
        position: relative;
        z-index: 1;
        color: #f3f8fc;
        font-size: clamp(2.2rem, 5vw, 3.8rem);
        line-height: 1.05;
        margin: 14px 0 12px;
    }
    .hero p {
        position: relative;
        z-index: 1;
        max-width: 700px;
        color: #afc0d0;
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 0;
    }
    .section-kicker {
        color: var(--teal);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .feature-card, .info-card {
        height: 100%;
        padding: 20px;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--panel);
    }
    .feature-card .icon { font-size: 22px; margin-bottom: 14px; }
    .feature-card h3, .info-card h3 { color: #eaf1f8; font-size: 1rem; margin: 0 0 8px; }
    .feature-card p, .info-card p { color: var(--muted); font-size: .9rem; line-height: 1.55; margin: 0; }
    .feature-card .unit {
        display: inline-block;
        color: #82e4d0;
        background: rgba(87, 224, 193, .09);
        border: 1px solid rgba(87, 224, 193, .17);
        border-radius: 99px;
        padding: 3px 9px;
        margin-top: 14px;
        font-size: .72rem;
    }
    .note {
        padding: 14px 17px;
        border-left: 3px solid var(--teal);
        border-radius: 0 10px 10px 0;
        background: rgba(87, 224, 193, .07);
        color: #b7c8d7;
        line-height: 1.55;
    }
    .stButton > button {
        color: #061611;
        border: 0;
        border-radius: 10px;
        background: linear-gradient(100deg, #57e0c1, #8ae5c5);
        font-weight: 700;
        padding: .65rem 1.2rem;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover {
        color: #061611;
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(87, 224, 193, .2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    kmeans = joblib.load(APP_DIR / "kmeans_traffic_model.pkl")
    scaler = joblib.load(APP_DIR / "scaler.pkl")
    mapping = joblib.load(APP_DIR / "cluster_mapping.pkl")
    return kmeans, scaler, mapping


@st.cache_resource
def load_flow_analysis():
    return joblib.load(APP_DIR / "flow_analysis.pkl")


@st.cache_data
def load_dataset():
    data = pd.read_csv(DATASET_PATH)
    data["date_time"] = pd.to_datetime(data["date_time"])
    return data


kmeans, scaler, cluster_to_state = load_models()
flow_analysis = load_flow_analysis()
dataset = load_dataset()

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Traffic intelligence · Project dashboard</div>
      <h1>Read the road<br>before it slows.</h1>
      <p>Explore hourly interstate traffic patterns, forecast flow volume, compare
      classification models, and classify road conditions with a four-state model.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

overview_tab, predictor_tab, dataset_tab = st.tabs(
    ["✦  Project overview", "⌁  Live predictor", "▤  Dataset & features"]
)

with overview_tab:
    st.markdown('<div class="section-kicker">At a glance</div>', unsafe_allow_html=True)
    st.subheader("A clearer picture of traffic flow")
    date_range = (
        f"{dataset['date_time'].min():%b %Y} — "
        f"{dataset['date_time'].max():%b %Y}"
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Hourly observations", f"{len(dataset):,}")
    metric_columns[1].metric("Coverage", date_range)
    metric_columns[2].metric(
        "Average volume", f"{dataset['traffic_volume'].mean():,.0f} veh/hr"
    )
    metric_columns[3].metric("Traffic states", "4 clusters")

    st.markdown("")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown('<div class="section-kicker">Project overview</div>', unsafe_allow_html=True)
        st.markdown(
            """
            ### From raw observations to road states
            This project turns hourly interstate volume records into an at-a-glance
            traffic state. A **K-Means** model groups observations with similar
            conditions; the resulting clusters are labeled from free flowing to
            gridlock using their average speed.

            The input variables are standardized before clustering so differences in
            units do not dominate the model. The predictor uses the same four inputs
            and preprocessing pipeline as training.
            """
        )
        st.markdown(
            """
            <div class="note"><strong>How to read it:</strong> this is an exploratory
            traffic-state classifier, not a live sensor feed or travel-time forecast.
            Average speed is estimated during training because it is not present in
            the source CSV.</div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-kicker">Model pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-card">
              <h3>01 &nbsp; Prepare</h3><p>Extract hour and weekday from each timestamp.</p>
              <hr style="border-color:rgba(159,185,210,.13)">
              <h3>02 &nbsp; Scale</h3><p>Standardize all four model inputs.</p>
              <hr style="border-color:rgba(159,185,210,.13)">
              <h3>03 &nbsp; Cluster</h3><p>Group observations into four K-Means clusters.</p>
              <hr style="border-color:rgba(159,185,210,.13)">
              <h3>04 &nbsp; Label</h3><p>Rank clusters by average speed and map them to traffic states.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown('<div class="section-kicker">Model evaluation</div>', unsafe_allow_html=True)
    st.subheader("Forecasting & classifier benchmarks")
    score_left, score_right = st.columns([1.1, 1], gap="large")
    with score_left:
        regression_metrics = flow_analysis["regression_metrics"]
        regression_columns = st.columns(3)
        regression_columns[0].metric(
            "Volume MAE", f"{regression_metrics['MAE']:,.0f} veh/hr"
        )
        regression_columns[1].metric(
            "Volume RMSE", f"{regression_metrics['RMSE']:,.0f} veh/hr"
        )
        regression_columns[2].metric("Volume R²", f"{regression_metrics['R2']:.2f}")
        st.caption("Random Forest regression · lower MAE/RMSE is better; higher R² is better.")
    with score_right:
        comparison = pd.DataFrame(flow_analysis["classification_metrics"])
        for metric in ["Accuracy", "Macro F1"]:
            comparison[metric] = comparison[metric].map(lambda value: f"{value:.1%}")
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.caption(
            "The models classify traffic-volume bands; macro F1 weights all four "
            "bands equally."
        )

    importance_left, importance_right = st.columns([1.15, 1], gap="large")
    with importance_left:
        st.markdown('<div class="section-kicker">Ensemble insight</div>', unsafe_allow_html=True)
        st.subheader("Random Forest feature importance")
        importance = flow_analysis["feature_importances"].set_index("Feature")
        st.bar_chart(importance, horizontal=True, color="#70a9ff")
    with importance_right:
        st.markdown(
            f"""
            <div class="feature-card">
              <div class="icon">◈</div>
              <h3>Chronological holdout</h3>
              <p>Models train on the first {flow_analysis['train_rows']:,} records and
              are evaluated on the next {flow_analysis['test_rows']:,} records,
              from {flow_analysis['test_start'][:10]} through
              {flow_analysis['test_end'][:10]}.</p>
              <p style="margin-top:12px">Naive Bayes and Logistic Regression are
              compared with the Random Forest classifier on the same traffic-volume
              bands. The chart ranks the Random Forest classifier's strongest
              signals.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    chart_column, chart_note = st.columns([1.35, 1], gap="large")
    with chart_column:
        st.markdown('<div class="section-kicker">Daily rhythm</div>', unsafe_allow_html=True)
        st.subheader("Average traffic volume by hour")
        hourly_profile = (
            dataset.assign(hour=dataset["date_time"].dt.hour)
            .groupby("hour", as_index=False)["traffic_volume"]
            .mean()
            .rename(columns={"hour": "Hour", "traffic_volume": "Average volume"})
            .set_index("Hour")
        )
        st.bar_chart(hourly_profile, color="#57e0c1")
    with chart_note:
        st.markdown('<div class="section-kicker">What the model sees</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="feature-card">
              <div class="icon">◉</div>
              <h3>Four signals, one traffic picture</h3>
              <p>Time-of-day and weekday describe when a reading occurs.
              Traffic volume and estimated speed describe the road conditions.
              Together, they help distinguish a busy road from a slow one.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with predictor_tab:
    st.markdown('<div class="section-kicker">Interactive tool</div>', unsafe_allow_html=True)
    st.subheader("Forecast hourly traffic")
    st.caption(
        "Estimate traffic volume from time and weather, then compare three models "
        "for the expected volume band."
    )
    day_options = [
        (0, "Monday"),
        (1, "Tuesday"),
        (2, "Wednesday"),
        (3, "Thursday"),
        (4, "Friday"),
        (5, "Saturday"),
        (6, "Sunday"),
    ]
    with st.form("flow_forecast_form"):
        time_column, weather_column = st.columns(2, gap="large")
        with time_column:
            forecast_day = st.selectbox(
                "Day of the week",
                day_options,
                format_func=lambda value: value[1],
            )
            forecast_hour = st.slider(
                "Hour of day", min_value=0, max_value=23, value=9
            )
        with weather_column:
            forecast_temp = st.number_input(
                "Temperature (K)",
                min_value=0.0,
                value=float(dataset["temp"].median()),
                step=0.5,
            )
            forecast_rain = st.number_input(
                "Rainfall in previous hour (mm)",
                min_value=0.0,
                value=float(dataset["rain_1h"].median()),
                step=0.1,
            )
            forecast_snow = st.number_input(
                "Snowfall in previous hour (mm)",
                min_value=0.0,
                value=float(dataset["snow_1h"].median()),
                step=0.1,
            )
            forecast_clouds = st.slider(
                "Cloud cover (%)",
                min_value=0,
                max_value=100,
                value=int(dataset["clouds_all"].median()),
            )
        forecast_submitted = st.form_submit_button(
            "Estimate hourly flow", type="primary"
        )

    if forecast_submitted:
        forecast_input = pd.DataFrame(
            [
                [
                    forecast_hour,
                    forecast_day[0],
                    forecast_temp,
                    forecast_rain,
                    forecast_snow,
                    forecast_clouds,
                ]
            ],
            columns=FLOW_FEATURES,
        )
        estimated_volume = flow_analysis["regressor"].predict(forecast_input)[0]
        st.markdown("#### Continuous volume estimate")
        st.metric("Predicted traffic volume", f"{estimated_volume:,.0f} vehicles/hour")

        st.markdown("#### Predicted traffic-volume band")
        model_columns = st.columns(len(flow_analysis["classifiers"]))
        for column, (model_name, model) in zip(
            model_columns, flow_analysis["classifiers"].items()
        ):
            volume_band = model.predict(forecast_input)[0]
            with column:
                st.metric(model_name, volume_band)
        thresholds = flow_analysis["volume_thresholds"]
        st.caption(
            "Volume bands are defined by training-set quartiles: "
            f"Low ≤ {thresholds[0]:,.0f}, Moderate ≤ {thresholds[1]:,.0f}, "
            f"High ≤ {thresholds[2]:,.0f}, and Very high above "
            f"{thresholds[2]:,.0f} vehicles/hour."
        )

    with st.expander("Original K-Means traffic-state classifier"):
        st.caption("Classify an observed traffic scenario using volume and speed.")
        cluster_left, cluster_right = st.columns(2, gap="large")
        with cluster_left:
            cluster_day = st.selectbox(
                "Observed day of the week",
                day_options,
                format_func=lambda value: value[1],
                key="cluster_day",
            )
            cluster_hour = st.slider(
                "Observed hour of day",
                min_value=0,
                max_value=23,
                value=9,
                key="cluster_hour",
            )
        with cluster_right:
            cluster_volume = st.number_input(
                "Observed traffic volume (vehicles/hour)",
                min_value=0,
                max_value=10000,
                value=1000,
                step=100,
                key="cluster_volume",
            )
            cluster_speed = st.slider(
                "Observed average speed (km/h)",
                min_value=0,
                max_value=120,
                value=15,
                key="cluster_speed",
            )
        if st.button("Classify traffic state"):
            cluster_input = pd.DataFrame(
                [[cluster_hour, cluster_day[0], cluster_volume, cluster_speed]],
                columns=MODEL_FEATURES,
            )
            scaled_input = scaler.transform(cluster_input)
            cluster_id = kmeans.predict(scaled_input)[0]
            predicted_state = cluster_to_state[cluster_id]
            st.success(f"Predicted flow state: {predicted_state}")

with dataset_tab:
    st.markdown('<div class="section-kicker">Data explorer</div>', unsafe_allow_html=True)
    st.subheader("Inside the dataset")
    st.write(
        "The source file contains hourly Metro Interstate traffic observations, "
        "weather conditions, and timestamps. K-Means uses four state-model inputs; "
        "regression and traffic-band classifiers use six time and weather features."
    )

    dataset_metrics = st.columns(3)
    dataset_metrics[0].metric("Source fields", f"{dataset.shape[1]}")
    dataset_metrics[1].metric(
        "Weather categories", f"{dataset['weather_main'].nunique():,}"
    )
    dataset_metrics[2].metric(
        "Peak recorded volume", f"{dataset['traffic_volume'].max():,} veh/hr"
    )

    st.markdown('<div class="section-kicker">K-Means state model inputs</div>', unsafe_allow_html=True)
    feature_columns = st.columns(4, gap="medium")
    model_feature_cards = [
        ("◷", "Hour", "Hour extracted from the observation timestamp.", "0–23"),
        ("▦", "Day of week", "Weekday encoded from Monday (0) to Sunday (6).", "0–6"),
        ("↗", "Traffic volume", "Vehicles recorded per hour in the source dataset.", "vehicles/hr"),
        ("⌁", "Average speed", "Estimated speed used by the training pipeline.", "km/h · synthetic"),
    ]
    for column, (icon, name, description, unit) in zip(feature_columns, model_feature_cards):
        with column:
            st.markdown(
                f"""
                <div class="feature-card">
                  <div class="icon">{icon}</div>
                  <h3>{name}</h3>
                  <p>{description}</p>
                  <span class="unit">{unit}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown(
        '<div class="section-kicker">Regression & supervised classifier inputs</div>',
        unsafe_allow_html=True,
    )
    flow_feature_cards = [
        ("◷", "Hour", "Hour extracted from the observation timestamp.", "0–23"),
        ("▦", "Day of week", "Weekday encoded from Monday (0) to Sunday (6).", "0–6"),
        ("°", "Temperature", "Observed ambient temperature.", "Kelvin"),
        ("☂", "Rainfall", "Precipitation during the previous hour.", "mm"),
        ("❄", "Snowfall", "Snowfall during the previous hour.", "mm"),
        ("☁", "Cloud cover", "Observed cloud coverage.", "percent"),
    ]
    for start in range(0, len(flow_feature_cards), 3):
        feature_columns = st.columns(3, gap="medium")
        for column, (icon, name, description, unit) in zip(
            feature_columns, flow_feature_cards[start : start + 3]
        ):
            with column:
                st.markdown(
                  f"""
                  <div class="feature-card">
                    <div class="icon">{icon}</div>
                    <h3>{name}</h3>
                    <p>{description}</p>
                    <span class="unit">{unit}</span>
                  </div>
                  """,
                  unsafe_allow_html=True,
                )

    st.markdown("")
    st.markdown(
        f"""
        <div class="note"><strong>Target definitions:</strong> regression estimates
        traffic_volume in vehicles per hour. Naive Bayes, Logistic Regression, and
        Random Forest classify four traffic-volume bands using training-set quartiles
        ({flow_analysis['volume_thresholds'][0]:,.0f}, 
        {flow_analysis['volume_thresholds'][1]:,.0f}, 
        {flow_analysis['volume_thresholds'][2]:,.0f} vehicles/hour). The separate
        K-Means model still uses average_speed, which is not observed in the CSV and
        is estimated during training.</div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View the source dataset fields"):
        source_fields = pd.DataFrame(
            [
                ("holiday", "Holiday name or None", "Raw"),
                ("temp", "Temperature in Kelvin", "Raw + flow model input"),
                ("rain_1h", "Rainfall in the previous hour (mm)", "Raw + flow model input"),
                ("snow_1h", "Snowfall in the previous hour (mm)", "Raw + flow model input"),
                ("clouds_all", "Cloud cover (%)", "Raw + flow model input"),
                ("weather_main", "Broad weather category", "Raw"),
                ("weather_description", "Detailed weather description", "Raw"),
                ("date_time", "Timestamp of the hourly observation", "Raw"),
                ("traffic_volume", "Vehicles recorded per hour", "Raw + target / K-Means input"),
                ("hour", "Hour derived from date_time", "Engineered + model input"),
                ("day_of_week", "Weekday derived from date_time", "Engineered + model input"),
                ("average_speed", "Rule-based speed estimate (km/h)", "Synthetic + model input"),
            ],
            columns=["Field", "Description", "Role"],
        )
        st.dataframe(source_fields, hide_index=True, width="stretch")

    st.markdown('<div class="section-kicker">Sample records</div>', unsafe_allow_html=True)
    preview_columns = [
        "date_time",
        "traffic_volume",
        "temp",
        "rain_1h",
        "clouds_all",
        "weather_main",
    ]
    st.dataframe(
        dataset[preview_columns].head(12),
        hide_index=True,
        width="stretch",
    )
