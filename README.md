# How to run

1. Download models

    Change the permission of download_models.sh to be executable Then run the script:
    ```bash
    chmod +rwx scripts/download_models.sh
    ./scripts/download_models.sh
    ```
    
2. Run inference script to get predictions:
    ```bash
    python src/inference.py
    ```