import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input

df = pd.read_csv('/Users/saymyname/Downloads/AD_0212/atopy_severe.csv')

severe_drug_codes = [1278, 1280, 1284, 1285, 1287, 1288, 1289, 1291, 1292, 1293]

def run_lstm_baseline(df):
    severe_event = df[df['DATA'].isin(severe_drug_codes)].groupby('ID')['AGE'].min().reset_index()
    severe_event.columns = ['ID', 'severe_age']
    
    all_patients = pd.DataFrame({'ID': df['ID'].unique()})
    label_df = pd.merge(all_patients, severe_event, on='ID', how='left')
    label_df['target'] = label_df['severe_age'].notna().astype(int)

    feature_df = df[~df['DATA'].isin(severe_drug_codes)].copy()

    unique_codes = feature_df['DATA'].unique()
    code_to_int = {code: i+1 for i, code in enumerate(unique_codes)}
    feature_df['DATA_ENCODED'] = feature_df['DATA'].map(code_to_int)

    feature_df = feature_df.sort_values(by=['ID', 'AGE'])
    sequences = feature_df.groupby('ID')['DATA_ENCODED'].apply(list).reset_index()

    final_df = pd.merge(label_df, sequences, on='ID', how='left')
    final_df['DATA_ENCODED'] = final_df['DATA_ENCODED'].apply(lambda x: x if isinstance(x, list) else [])

    MAX_LEN = 50
    
    X = pad_sequences(final_df['DATA_ENCODED'], maxlen=MAX_LEN, padding='pre')
    y = final_df['target'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}

    VOCAB_SIZE = len(code_to_int) + 1
    
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=VOCAB_SIZE, output_dim=32), 
        LSTM(32, return_sequences=False),              
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')             
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')

    model.fit(X_train, y_train, epochs=5, batch_size=64, class_weight=class_weight_dict, verbose=1)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = (y_test == y_pred).mean()
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-score:  {f1:.4f}")

run_lstm_baseline(df)