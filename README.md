# How to run

The code was tested on Google Colab. Here are the steps to reproduce the results:

1. Install requirements
    ```bash
    pip install -r requirements.txt
    ```

1. Download models
    ```bash
    python src/download_models.py
    ```
    
2. Run inference script to get predictions:
    ```bash
    python src/inference.py --dataset_path "datasets/Mawqif_AllTargets_Blind Test.csv" --output_path "predictions.csv"
    ```

