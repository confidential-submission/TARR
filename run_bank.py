"""
Reproduce Table 1 / Table 3 results for the Bank Marketing dataset.

Sensitive attribute: age  (one experiment).

Usage:
    python run_bank.py --method tarr
    python run_bank.py --method erm
"""

import argparse

import torch
from torch.utils.data import DataLoader

from tarr.data import load_bank
from tarr.eval import accuracy, consistency_score
from tarr.model import SimpleMLP
from train import erm_train, tarr_train


def run_one(method: str, args) -> dict:
    device = args.device
    train_ds, val_ds, test_ds, n_features, sensitive_idx = load_bank(
        sensitive='age', seed=args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, num_workers=0)

    model = SimpleMLP(in_features=n_features)

    if method == 'tarr':
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        tarr_train(
            model, train_loader,
            sensitive_idx=sensitive_idx,
            eps_p=args.eps_p,
            eps_f=args.eps_f,
            optimizer=opt,
            device=device,
            n_epochs=args.epochs,
        )
    elif method == 'erm':
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        erm_train(model, train_loader, optimizer=opt, device=device,
                  n_epochs=args.epochs)
    else:
        raise ValueError(f"Unknown method '{method}'")

    acc = accuracy(model, test_loader, device)
    cns = consistency_score(model, test_loader, sensitive_idx, device)
    val_cns = consistency_score(model, val_loader, sensitive_idx, device)

    print(f"Bank (age) | method={method:4s} | "
          f"acc={acc:.2f}%  CNS={cns:.2f}%  val_CNS={val_cns:.2f}%")
    return {'method': method, 'acc': acc, 'cns': cns}


def main():
    parser = argparse.ArgumentParser(description='TARR — Bank Marketing dataset')
    parser.add_argument('--method', choices=['tarr', 'erm'], default='tarr')
    parser.add_argument('--eps_p', type=float, default=1.0)
    parser.add_argument('--eps_f', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()

    run_one(args.method, args)
    if args.method == 'tarr':
        run_one('erm', args)


if __name__ == '__main__':
    main()
