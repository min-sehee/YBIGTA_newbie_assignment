import torch # type: ignore
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # update gate z
        self.linear_xz = nn.Linear(input_size, hidden_size)
        self.linear_hz = nn.Linear(hidden_size, hidden_size)

        # reset gate r
        self.linear_xr = nn.Linear(input_size, hidden_size)
        self.linear_hr = nn.Linear(hidden_size, hidden_size)

        # candidate hidden state
        self.linear_xh = nn.Linear(input_size, hidden_size)
        self.linear_hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        z = torch.sigmoid(self.linear_xz(x) + self.linear_hz(h))
        r = torch.sigmoid(self.linear_xr(x) + self.linear_hr(h))
        h_tilde = torch.tanh(self.linear_xh(x) + self.linear_hh(r * h))
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        self.to_d_model = nn.Linear(hidden_size, input_size)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, seq_len, _ = inputs.size()
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            h = self.cell(x_t, h)

        h_projected = self.to_d_model(h)       

        return h_projected