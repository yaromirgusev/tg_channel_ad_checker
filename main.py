from data_getting import login_to_telegram, get_posts, posts_to_excel
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, fc_out_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn1 = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, fc_out_size)
        self.rnn2 = nn.RNN(fc_out_size, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        rnn_out1, _ = self.rnn1(x)
        out1 = self.fc1(rnn_out1[:, -1, :]).unsqueeze(1)
        rnn_out2, _ = self.rnn2(out1)
        out2 = self.fc2(rnn_out2[:, -1, :]).flatten()
        return self.sigmoid(out2)

def load_model(model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_name)
    model.to(device)
    return model

def inference(text, model):
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=False,
        return_tensors='pt',
        truncation=True
    )

    input_id = encoding['input_ids'].to(device)

    with torch.no_grad():
        model.eval() 
        output = model(input_id)
        prediction_prob = output.item()

    prediction = ''
    if round(prediction_prob) == 1:
        prediction = 'Реклама'
    else:
        prediction = 'Не реклама'

    if prediction == 'Реклама':
        return 1
    return 0

def collect_data(channel_url, amount_of_posts, type_of_url, file_name):
    login_to_telegram()
    z = get_posts(channel_url, amount_of_posts, type_of_url)
    posts_to_excel(z, file_name)

# collect_data('https://web.telegram.org/a/#-1001763496063', 150, 1, 'data.xlsx')

def get_amount_of_ad(model, file_name):
    df = pd.read_excel(file_name)
    data = np.array(df['posts'])

    total = len(data)
    preds = []
    for i in data:
        preds.append(inference(i, model))

    amount = np.array(preds).sum() / total
    return f'В последних {total} постах на канале {round(amount*100)}% рекламы ({np.array(preds).sum()} из {total})'

model = load_model('model.pth')

print(get_amount_of_ad(model, 'data.xlsx'))



