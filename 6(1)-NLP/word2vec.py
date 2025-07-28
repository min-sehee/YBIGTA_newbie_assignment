import torch # type: ignore
from torch import nn, Tensor, LongTensor # type: ignore
from torch.optim import Adam # type: ignore

from transformers import PreTrainedTokenizer

from typing import Literal


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method


    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)


        if self.method == "cbow":

            cbow_sample: list[tuple[list[int], int]] = []

            for text in corpus:
                token_ids: list[int] = tokenizer(text, add_special_tokens=False)["input_ids"]
                
                for i in range(len(token_ids)):
                    cbow_context = []

                    for j in range(-self.window_size, self.window_size + 1):
                        if j == 0 or i + j < 0 or i + j >= len(token_ids):
                            continue
                        cbow_context.append(token_ids[i + j])

                    if len(cbow_context) == 0:
                        continue

                    cbow_center = token_ids[i]
                    cbow_sample.append((cbow_context, cbow_center))

            self._train_cbow(cbow_sample, criterion, optimizer, num_epochs)

        elif self.method == "skipgram":

            skipgram_sample: list[tuple[int, int]] = []

            for text in corpus:
                token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                
                for i in range(len(token_ids)):
                    skipgram_center: int = token_ids[i]

                    for j in range(-self.window_size, self.window_size + 1):
                        if j == 0 or i + j < 0 or i + j >= len(token_ids):
                            continue
                        skipgram_context: int = token_ids[i + j]
                        skipgram_sample.append((skipgram_center, skipgram_context))

            print("Number of samples:", len(skipgram_sample))
            self._train_skipgram(skipgram_sample, criterion, optimizer, num_epochs)


    def _train_cbow(
        self,
        sample: list[tuple[list[int], int]],
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        num_epochs: int
    ) -> None:
        
        for epoch in range(num_epochs):
            total_loss = 0.0

            for context_ids, center_id in sample:
                context_tensor = torch.tensor(context_ids, device=self.embeddings.weight.device)  
                center_tensor = torch.tensor([center_id], device=self.embeddings.weight.device) 

                context_embeds = self.embeddings(context_tensor)          
                avg_embed = context_embeds.mean(dim=0, keepdim=True)     

                logits = self.weight(avg_embed)                          

                loss = criterion(logits, center_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")


    def _train_skipgram(
        self,
        sample: list[tuple[int, int]],
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        num_epochs: int
    ) -> None:

        for epoch in range(num_epochs):
            total_loss = 0.0

            for i, (context_id, center_id) in enumerate(sample):

                center_tensor = torch.tensor([center_id], device=self.embeddings.weight.device)   
                context_tensor = torch.tensor([context_id], device=self.embeddings.weight.device) 

                center_embed = self.embeddings(center_tensor)

                logits = self.weight(center_embed)      

                loss = criterion(logits, context_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if i % 1000 == 0:
                    print(f"Epoch {epoch+1}, Step {i}, partial loss: {loss.item():.4f}")

            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")