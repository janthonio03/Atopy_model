import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# 1. 파일 로드
df = pd.read_csv('/Users/saymyname/Downloads/AD_0212/atopy_severe.csv')

# 약물 코드 설정
severe_drug_codes = [1278, 1280, 1284, 1285, 1287, 1288, 1289, 1291, 1292, 1293]

def prep_and_coinflip_baseline(df):
    # 1. 환자별 최초 중증 진단일 찾기
    severe_event = df[df['DATA'].isin(severe_drug_codes)].groupby('ID')['AGE'].min().reset_index()
    severe_event.columns = ['ID', 'severe_age']
    
    # 2. 전체 환자 리스트
    all_patients = pd.DataFrame({'ID': df['ID'].unique()})
    label_df = pd.merge(all_patients, severe_event, on='ID', how='left')
    
    # 3. Target 설정
    label_df['target'] = label_df['severe_age'].notna().astype(int)

    # 4. Train/Test Split
    train_ids, test_ids = train_test_split(label_df['ID'].unique(), test_size=0.2, random_state=42)
    test_labels = label_df[label_df['ID'].isin(test_ids)]['target'].values
    
    # 5. 50:50 Coin Flip Baseline (확률 지정 없음 = 균등 확률)
    y_pred_coin = np.random.choice([0, 1], size=len(test_labels), p=[0.5, 0.5])
    
    # 6. 성능 지표 계산
    accuracy = (test_labels == y_pred_coin).mean()
    f1 = f1_score(test_labels, y_pred_coin)
    precision = precision_score(test_labels, y_pred_coin)
    recall = recall_score(test_labels, y_pred_coin)
    
    print(f"--- [50:50 Coin Flip Baseline 결과] ---")
    print(f"테스트셋 내 중증 환자 수: {test_labels.sum()}명 / 전체: {len(test_labels)}명")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f} (정확도)")
    print(f"Precision: {precision:.4f} (중증이라 예측한 것 중 실제 비율)")
    print(f"Recall:    {recall:.4f} (실제 중증 환자 중 찾아낸 비율)")
    print(f"F1-score:  {f1:.4f} (모델의 종합 성능 지표)")
    print("-" * 40)
    
    print("\n[Classification Report]")
    print(classification_report(test_labels, y_pred_coin, target_names=['Normal(0)', 'Severe(1)']))
    
    return label_df, test_labels

# 실행
final_label_df, y_true = prep_and_coinflip_baseline(df)