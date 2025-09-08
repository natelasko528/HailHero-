# Hail Hero MVP - README

This document provides instructions for running the Hail Hero MVP.

## Running the MVP

1.  **Install Dependencies**

    ```bash
    pip install -r specs/001-hail-hero-hail/prototypes/requirements.txt
    pip install flask
    ```

2.  **Generate Leads**

    This script will fetch real data from NOAA if you have an `NCEI_TOKEN` environment variable set. Otherwise, it will generate synthetic data.

    ```bash
    python src/mvp/runner.py
    ```

3.  **Run the Web App**

    ```bash
    export FLASK_APP=src/mvp/app.py
    flask run
    ```

    You can then view the leads at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

4.  **Submit an Inspection (Example)**

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{ "lead_id": "<LEAD_ID>", "notes": "Roof looks damaged.", "photos": ["photo1.jpg"] }' http://127.0.0.1:5000/api/inspection
    ```
