import torch # type: ignore
from torch import nn, optim, Tensor, LongTensor, FloatTensor # type: ignore
from torch.utils.data import DataLoader # type: ignore
from collections import Counter
from torch import tensor # type: ignore

from transformers import AutoTokenizer
from datasets import load_dataset

from tqdm import tqdm # type: ignore
from sklearn.metrics import f1_score

from word2vec import Word2Vec
from model import MyGRULanguageModel
from config import *



if __name__ == "__main__":
    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # load Word2Vec checkpoint and get trained embeddings
    word2vec = Word2Vec(vocab_size, d_model, window_size, method)
    checkpoint = torch.load("word2vec.pt")
    word2vec.load_state_dict(checkpoint)
    embeddings = word2vec.embeddings_weight()

    # load dataset
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    train_labels = dataset["train"]["label"]
    label_counts = Counter(train_labels)
    num_classes = len(label_counts)

    # class weights (inverse frequency)
    weights = [1.0 / (label_counts[i] ** 0.5) for i in range(num_classes)]
    class_weights = torch.tensor(weights).to(device)

    # declare model, criterion and optimizer
    model = MyGRULanguageModel(d_model, hidden_size, num_classes, embeddings).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 🔥 클래스 가중치 적용
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # make dataloaders
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=True)

    # train
    for epoch in tqdm(range(num_epochs)):
        loss_sum = 0
        for data in train_loader:
            optimizer.zero_grad()
            input_ids = tokenizer(data["verse_text"], padding=True, return_tensors="pt")\
                .input_ids.to(device)
            labels = data["label"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        preds = []
        labels = []
        with torch.no_grad():
            for data in validation_loader:
                input_ids = tokenizer(data["verse_text"], padding=True, return_tensors="pt")\
                    .input_ids.to(device)
                logits = model(input_ids)
                labels += data["label"].tolist()
                preds += logits.argmax(-1).cpu().tolist()

        macro = f1_score(labels, preds, average='macro')
        micro = f1_score(labels, preds, average='micro')
        print(f"loss: {loss_sum/len(train_loader):.6f} | macro: {macro:.6f} | micro: {micro:.6f}")

    # save model checkpoint
    torch.save(model.cpu().state_dict(), "checkpoint.pt")