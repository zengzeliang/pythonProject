import torch
from my_torch_crf import CRF
num_tags = 5
model = CRF(num_tags)

if __name__ == '__main__':
    seq_length = 3
    batch_size = 2

    emissions = torch.randn(seq_length, batch_size, num_tags)

    tags = torch.tensor([[0, 1], [2, 4], [3, 1]], dtype=torch.long)

    ans = model(emissions, tags)

    de = model.decode(emissions)

    print(de)

    print(ans)
