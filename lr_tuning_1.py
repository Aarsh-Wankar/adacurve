import os
import torch
import train_pipeline

adacurve_lrs = [0.1, 0.01, 0.001]
adahessian_lrs = [0.15, 0.1, 0.01]
adam_lrs = [1e-2, 1e-3, 3e-4]
adamw_lrs = [1e-2, 1e-3, 3e-4]

model = "resnet18"
dataset = "cifar10"

# Finding best adacurve lrs
adacurve_vals = []
for lr in adacurve_lrs:
    val_acc, test_acc = train_pipeline.main([
        "--dataset", dataset,
        "--model", model,
        "--optimizer", "adacurve",
        "--lr", str(lr),
        "--epochs", "20",
        "--batch-size", "64",
        "--log-dir", "./lr_tuning_results",
        "--seed", "42",
        "--scheduler", "cosine",
        "--device", "cuda:0"
    ])
    adacurve_vals.append(val_acc)
    print(f"Learning rate: {lr}, Validation accuracy: {val_acc}")
best_adacurve_lr = adacurve_lrs[adacurve_vals.index(max(adacurve_vals))]
print("Best Adacurve learning rate:", best_adacurve_lr)

# Finding best Adahessian lrs
adahessian_vals = []
for lr in adahessian_lrs:
    val_acc, test_acc = train_pipeline.main([
        "--dataset", dataset,
        "--model", model,
        "--optimizer", "adahessian",
        "--lr", str(lr),
        "--epochs", "20",
        "--batch-size", "64",
        "--log-dir", "./lr_tuning_results",
        "--seed", "42",
        "--scheduler", "cosine",
        "--device", "cuda:0"
    ])
    adahessian_vals.append(val_acc)
    print(f"Learning rate: {lr}, Validation accuracy: {val_acc}")
best_adahessian_lr = adahessian_lrs[adahessian_vals.index(max(adahessian_vals))]
print("Best Adahessian learning rate:", best_adahessian_lr)