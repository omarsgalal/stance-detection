# How to run

1. Add working directory to python module search path
    ```bash
    export PYTHONPATH="${PYTHONPATH}:${pwd}"
    ```
2. Download models
    ```bash
    ./scripts/download_models.sh
    ```
    If permission denied, run the following first:
    ```bash
    chmod +rwx scripts/download_models.sh
    ```