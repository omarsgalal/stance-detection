import pandas as pd

def save_output(df, save_path='predictions.tsv'):
    final_df = pd.DataFrame()
    final_df["ID"] = df["ID"]
    final_df["Target"] = df["target"]
    final_df["Tweet"] = df["text"]
    final_df["Stance"] = df["stance"]
    final_df["Stance"] = final_df["Stance"].apply(lambda x: x if x is not None and x != "Neutral" else "NONE")
    final_df["Stance"] = final_df["Stance"].apply(lambda x: x.upper())
    final_df.to_csv(save_path, index=False, sep="\t", encoding='utf8')