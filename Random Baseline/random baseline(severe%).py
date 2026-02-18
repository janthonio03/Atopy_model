import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# 1. 파일 로드
df = pd.read_csv('/Users/saymyname/Downloads/AD_0212/atopy_severe.csv')

# 약물 코드 설정 (데이터셋에 존재하는 것으로 확인된 코드들)
severe_drug_codes = [1278, 1280, 1284, 1285, 1287, 1288, 1289, 1291, 1292, 1293]

def prep_and_weighted_baseline_with_metrics(df):
    # 1. 환자별 최초 중증 진단일 찾기
    severe_event = df[df['DATA'].isin(severe_drug_codes)].groupby('ID')['AGE'].min().reset_index()
    severe_event.columns = ['ID', 'severe_age']
    
    # 2. 전체 환자 리스트 추출
    all_patients = pd.DataFrame({'ID': df['ID'].unique()})
    
    # 3. 정보 병합
    label_df = pd.merge(all_patients, severe_event, on='ID', how='left')
    
    # 4. Target 설정 (중증 기록이 있으면 1, 없으면 0)
    label_df['target'] = label_df['severe_age'].notna().astype(int)

    # 5. Train/Test Split (환자 단위로 8:2 분할)
    train_ids, test_ids = train_test_split(label_df['ID'].unique(), test_size=0.2, random_state=42)
    test_labels = label_df[label_df['ID'].isin(test_ids)]['target'].values
    
    # 6. Weighted Random Baseline (테스트셋의 실제 중증 비율 p 사용)
    p = test_labels.mean() 
    y_pred_weighted = np.random.choice([0, 1], size=len(test_labels), p=[1-p, p])
    
    # 7. 성능 지표 계산
    accuracy = (test_labels == y_pred_weighted).mean()
    f1 = f1_score(test_labels, y_pred_weighted)
    precision = precision_score(test_labels, y_pred_weighted)
    recall = recall_score(test_labels, y_pred_weighted)
    
    print(f"--- [비율 기반 Baseline 상세 결과] ---")
    print(f"테스트셋 내 중증 환자 수: {test_labels.sum()}명 / 전체: {len(test_labels)}명")
    print(f"실제 중증 비율 (p): {p*100:.2f}%")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (중증이라 예측한 것 중 실제 비율)")
    print(f"Recall:    {recall:.4f} (실제 중증 환자 중 찾아낸 비율)")
    print(f"F1-score:  {f1:.4f} (모델의 종합 성능 지표)")
    print("-" * 40)
    
    # 클래스별 상세 보고서 (0과 1 각각의 성적)
    print("\n[Classification Report]")
    print(classification_report(test_labels, y_pred_weighted, target_names=['Normal(0)', 'Severe(1)']))
    
    return label_df, test_labels

# 실행
final_label_df, y_true = prep_and_weighted_baseline_with_metrics(df)
