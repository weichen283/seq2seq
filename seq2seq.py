import torch.optim as optim
import warnings
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
from tqdm import tqdm
import spacy
from model import *
warnings.filterwarnings('ignore')

RETRAIN = True
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                      batch_size=BATCH_SIZE, device=device)
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
EMB_DIM = 256
HID_DIM = 512  # each conv. layer has 2 * hid_dim filters
ENC_LAYERS = 10  # number of conv. blocks in encoder
DEC_LAYERS = 10  # number of conv. blocks in decoder
ENC_KERNEL_SIZE = 3  # must be odd!
DEC_KERNEL_SIZE = 3  # can be even or odd
ENC_DROPOUT = 0.25
DEC_DROPOUT = 0.25
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)

model = Seq2Seq(enc, dec).to(device)
if RETRAIN:
    model.load_state_dict(torch.load('cnn.pth'))
    print('weights loaded')


optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

N_EPOCHS = 10
CLIP = 0.1

best_valid_loss = float('inf')

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    train_loss = 0
    train_bar = tqdm(train_iterator)
    for i, data in enumerate(train_bar):
        src = data.src
        trg = data.trg
        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        train_loss += loss.item()

        train_bar.desc = '\r[%d/%d] Train Loss: %.3f' % (epoch, N_EPOCHS, loss)
    print('\rTrain Loss: %.3f' % (train_loss / len(train_iterator)))

    val_bar = tqdm(valid_iterator)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_bar):
            src = data.src
            trg = data.trg
            output, _ = model(src, trg[:, :-1])
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
            loss = criterion(output, trg)
            val_loss += loss.item()
            val_bar.desc = '\r[%d/%d] Val Loss: %.3f' % (epoch, N_EPOCHS, loss)
    print('\rVal Loss: %.3f' % (val_loss / len(valid_iterator)))

    if (val_loss / len(valid_iterator)) < best_valid_loss:
        best_valid_loss = val_loss / len(valid_iterator)
        torch.save(model.state_dict(), 'cnn.pth')
