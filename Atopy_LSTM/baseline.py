import pandas as pd
import numpy as np

df = pd.read_csv('/Users/saymyname/Library/Mobile Documents/com~apple~CloudDocs/LDI LAB/Atopy/atopy_PatientData1/atopy_train.csv') # 데이터 파일명 입력

real_prescriptions = df[df['DOSE'] > 0].copy()

real_prescriptions['DOSE_log'] = np.log1p(real_prescriptions['DOSE'])
real_prescriptions['DAY_log'] = np.log1p(real_prescriptions['DAY'])

base_dose_mse = ((real_prescriptions['DOSE_log'] - real_prescriptions['DOSE_log'].mean())**2).mean()
base_dur_mse = ((real_prescriptions['DAY_log'] - real_prescriptions['DAY_log'].mean())**2).mean()

print(f"--- 실제 처방(777건) 전용 Baseline ---")
print(f"Dose Baseline MSE (Log): {base_dose_mse:.4f}")
print(f"Duration Baseline MSE (Log): {base_dur_mse:.4f}")
