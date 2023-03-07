import torch

def main():
    x = torch.tensor([[0.5,0.4,0.5], [1,1,1], [0,0,0]])
    y = torch.tensor([[0.5,0.4,0.5], [1,1,1], [0,0,0]])
    print(x)
    x[1][1] = 5
    print(x)

    z = x + y
    print(z)
    print(z.max().item())
    print(z.device)
    c = z.to("cuda")
    print(c.device)


if __name__ == "__main__":
    main()
