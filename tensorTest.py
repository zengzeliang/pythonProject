import torch

if __name__ == '__main__':
    a = torch.ones(2, 3)

    b = torch.ones(10, 3)

    c = torch.cat([a, b], dim=0)

    for epoch in range(10):
        print(epoch)

    a = torch.tensor([[0.0, 0.1, 0.1], [1.0, 1.5, 1.4], [2.1, 2.3, 2.4], [3.9, 3.9, 3.8]])


