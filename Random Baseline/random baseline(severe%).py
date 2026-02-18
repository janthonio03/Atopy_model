import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

df = pd.read_csv('/Users/saymyname/Downloads/AD_0212/atopy_severe.csv')

severe_drug_codes = [1278, 1280, 1284, 1285, 1287, 1288, 1289, 1291, 1292, 1293]

def prep_and_weighted_baseline_with_metrics(df):
    severe_event = df[df['DATA'].isin(severe_drug_codes)].groupby('ID')['AGE'].min().reset_index()
    severe_event.columns = ['ID', 'severe_age']

    all_patients = pd.DataFrame({'ID': df['ID'].unique()})

    label_df = pd.merge(all_patients, severe_event, on='ID', how='left')

    label_df['target'] = label_df['severe_age'].notna().astype(int)

    train_ids, test_ids = train_test_split(label_df['ID'].unique(), test_size=0.2, random_state=42)
    test_labels = label_df[label_df['ID'].isin(test_ids)]['target'].values

    p = test_labels.mean() 
    y_pred_weighted = np.random.choice([0, 1], size=len(test_labels), p=[1-p, p])

    accuracy = (test_labels == y_pred_weighted).mean()
    f1 = f1_score(test_labels, y_pred_weighted)
    precision = precision_score(test_labels, y_pred_weighted)
    recall = recall_score(test_labels, y_pred_weighted)
    
    print(f"--- [Severe% Baseline 상세 결과] ---")
    print(f"Percentage of Severe: {p*100:.2f}%")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-score:  {f1:.4f} (모델의 종합 성능 지표)")
    print("-" * 40)
    
    return label_df, test_labels

final_label_df, y_true = prep_and_weighted_baseline_with_metrics(df)
