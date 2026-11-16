# Traffic Flow Prediction

An interactive Streamlit dashboard for exploring Metro Interstate traffic data,
forecasting hourly traffic volume, comparing volume-band classifiers, and
classifying traffic states with a K-Means model.

The app loads the included CSV dataset and trained model files from this
directory. The average-speed feature in the K-Means model is synthetically
estimated during training; the app is an exploratory dashboard, not a live
sensor feed or travel-time forecast.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

To retrain the models and update the saved model files:

```bash
python train.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder's files to a GitHub repository.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app
   from that repository.
3. Select the branch containing the app and set the main file path to `app.py`.
4. Deploy. Streamlit installs the dependencies from `requirements.txt`.

The CSV and `.pkl` model files are required at runtime and should remain in the
repository alongside `app.py`.
