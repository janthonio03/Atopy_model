import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

df = pd.read_csv('/Users/saymyname/Downloads/AD_0212/atopy_severe.csv')

severe_drug_codes = [1278, 1280, 1284, 1285, 1287, 1288, 1289, 1291, 1292, 1293]

def prep_and_coinflip_baseline(df):
    severe_event = df[df['DATA'].isin(severe_drug_codes)].groupby('ID')['AGE'].min().reset_index()
    severe_event.columns = ['ID', 'severe_age']

    all_patients = pd.DataFrame({'ID': df['ID'].unique()})
    label_df = pd.merge(all_patients, severe_event, on='ID', how='left')

    label_df['target'] = label_df['severe_age'].notna().astype(int)

    train_ids, test_ids = train_test_split(label_df['ID'].unique(), test_size=0.2, random_state=42)
    test_labels = label_df[label_df['ID'].isin(test_ids)]['target'].values

    y_pred_coin = np.random.choice([0, 1], size=len(test_labels), p=[0.5, 0.5])

    accuracy = (test_labels == y_pred_coin).mean()
    f1 = f1_score(test_labels, y_pred_coin)
    precision = precision_score(test_labels, y_pred_coin)
    recall = recall_score(test_labels, y_pred_coin)
    
    print(f"--- [50:50 Random Baseline 결과] ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    return label_df, test_labels

final_label_df, y_true = prep_and_coinflip_baseline(df)