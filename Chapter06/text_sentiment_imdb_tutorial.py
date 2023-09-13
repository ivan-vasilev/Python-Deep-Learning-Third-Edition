import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB

from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(
    yield_tokens(IMDB(split='train')),
    specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(int(_label) - 1)
        processed_text = torch.tensor(vocab(tokenizer(_text)), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets


class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super().__init__()

        # Embedding field
        self.embedding = torch.nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            sparse=False)

        # LSTM cell
        self.rnn = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size)

        # Fully connected output
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, text_sequence, offsets):
        # Extract embedding vectors
        embeddings = self.embedding(text_sequence, offsets)

        packed_output, _ = self.rnn(embeddings)

        return self.fc(packed_output)


VOCAB_SIZE = len(vocab)
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 64

model = LSTMModel(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, 2)


def train_model(model, cost_function, optimizer, data_loader):
    # send the model to the GPU
    model.to(device)

    # set model to training mode
    model.train()

    current_loss = 0.0
    current_acc = 0

    # iterate over the training data
    for i, (labels, inputs, offsets) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        offsets = offsets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs, offsets)
            _, predictions = torch.max(outputs, 1)
            loss = cost_function(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

        # statistics
        current_loss += loss.item() * labels.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


def test_model(model, cost_function, data_loader):
    # send the model to the GPU
    model.to(device)

    # set model in evaluation mode
    model.eval()

    current_loss = 0.0
    current_acc = 0

    # iterate over  the validation data
    for i, (labels, inputs, offsets) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        offsets = offsets.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs, offsets)
            _, predictions = torch.max(outputs, 1)
            loss = cost_function(outputs, labels)

        # statistics
        current_loss += loss.item() * labels.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))

    return total_loss, total_acc


# Hyperparameters
EPOCHS = 5  # epoch
BATCH_SIZE = 64  # batch size for training

cost_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())

from torchtext.data.functional import to_map_style_dataset

train_iter, test_iter = IMDB()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=True, collate_fn=collate_batch)

for epoch in range(EPOCHS):
    print(f'Epoch: {epoch + 1}')
    train_model(model, cost_fn, optim, train_dataloader)
    test_model(model, cost_fn, test_dataloader)
