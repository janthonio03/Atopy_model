import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current Device: {device}")

class AtopySequenceDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.sort_values(['ID', 'AGE']).reset_index(drop=True)

        df['delta_age_log'] = np.log1p(df.groupby('ID')['AGE'].diff().fillna(0))
        df['DOSE_log'] = np.log1p(df['DOSE'])
        df['DAY_log'] = np.log1p(df['DAY']) 
        
        self.df = df
        self.target_indices = self.df[self.df['DOSE'] > 0].index.tolist()
        self.id_list = self.df['ID'].to_dict()

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, idx):
        target_idx = self.target_indices[idx]
        patient_id = self.id_list[target_idx]
        start_idx = (self.df['ID'] == patient_id).idxmax() 
        history = self.df.iloc[start_idx : target_idx + 1]
        
        tokens = torch.LongTensor(history['DATA'].values)
        times = torch.FloatTensor(history['delta_age_log'].values)
        target = torch.FloatTensor([history.iloc[-1]['DOSE_log'], history.iloc[-1]['DAY_log']])
        return tokens, times, target

def collate_fn(batch):
    tokens, times, targets = zip(*batch)
    tokens_pad = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
    times_pad = nn.utils.rnn.pad_sequence(times, batch_first=True, padding_value=0)
    return tokens_pad, times_pad, torch.stack(targets)

class AtopyMultiHeadLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, dropout_rate=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + 1, hidden_dim, batch_first=True, num_layers=2, dropout=dropout_rate)

        self.dose_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, tokens, times):
        x = torch.cat([self.embedding(tokens), times.unsqueeze(-1)], dim=-1)
        _, (hn, _) = self.lstm(x)
        last_hidden = hn[-1] 
        return self.dose_head(last_hidden), self.duration_head(last_hidden)

TRAIN_PATH = 'atopy_train.csv' 
VAL_PATH = 'atopy_val.csv'

train_ds = AtopySequenceDataset(TRAIN_PATH)
val_ds = AtopySequenceDataset(VAL_PATH)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

model = AtopyMultiHeadLSTM(vocab_size=2000).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
model.apply(init_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
criterion = nn.MSELoss()

BASELINE_DOSE = 0.1223
BASELINE_DUR = 0.4019

best_val_loss = float('inf')

for epoch in range(20):
    model.train()
    train_dose_loss, train_dur_loss = 0, 0
    
    for tokens, times, targets in train_loader:
        tokens, times, targets = tokens.to(device), times.to(device), targets.to(device)
        optimizer.zero_grad()

        dose_p, dur_p = model(tokens, times)
        loss_dose = criterion(dose_p.squeeze(), targets[:, 0])
        loss_dur = criterion(dur_p.squeeze(), targets[:, 1])

        total_loss = loss_dose + loss_dur
        total_loss.backward()
        optimizer.step()
        
        train_dose_loss += loss_dose.item()
        train_dur_loss += loss_dur.item()

    model.eval()
    val_dose_loss, val_dur_loss = 0, 0
    with torch.no_grad():
        for tokens, times, targets in val_loader:
            tokens, times, targets = tokens.to(device), times.to(device), targets.to(device)
            dose_p, dur_p = model(tokens, times)
            val_dose_loss += criterion(dose_p.squeeze(), targets[:, 0]).item()
            val_dur_loss += criterion(dur_p.squeeze(), targets[:, 1]).item()

    n_train, n_val = len(train_loader), len(val_loader)
    avg_val_dose = val_dose_loss / n_val
    avg_val_dur = val_dur_loss / n_val
    avg_val_total = avg_val_dose + avg_val_dur
 
    scheduler.step(avg_val_total)

    if avg_val_total < best_val_loss:
        best_val_loss = avg_val_total
        torch.save(model.state_dict(), 'best_atopy_model.pth')
        save_msg = " [BEST SAVED]"
    else:
        save_msg = ""

    print(f"Epoch {epoch+1:2d}")
    print(f"  [Train] Dose: {train_dose_loss/n_train:.4f}, Dur: {train_dur_loss/n_train:.4f}")
    print(f"  [Val  ] Dose: {avg_val_dose:.4f} (Base: {BASELINE_DOSE}), Dur: {avg_val_dur:.4f} (Base: {BASELINE_DUR}){save_msg}")
    print("-" * 65)
