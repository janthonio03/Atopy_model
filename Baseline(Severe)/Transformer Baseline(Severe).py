import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def run_transformer_baseline():
    df = pd.read_csv('/Users/saymyname/Downloads/AD_0212/atopy_severe.csv')
    severe_drug_codes = [1278, 1280, 1284, 1285, 1287, 1288, 1289, 1291, 1292, 1293]

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
    VOCAB_SIZE = len(code_to_int) + 1
    
    X = pad_sequences(final_df['DATA_ENCODED'], maxlen=MAX_LEN, padding='pre')
    y = final_df['target'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}

    EMBED_DIM = 32 
    NUM_HEADS = 2 
    FF_DIM = 32 

    inputs = layers.Input(shape=(MAX_LEN,))
    embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)
    x = embedding_layer(inputs)
    
    transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)
    x = transformer_block(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    
    model.fit(X_train, y_train, epochs=5, batch_size=64, class_weight=class_weight_dict, verbose=1)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    print(f"Accuracy:  {(y_test == y_pred).mean():.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")

run_transformer_baseline()