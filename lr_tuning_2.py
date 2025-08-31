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
with open("adam_adamw_best_lrs.txt", "w") as f:
    adam_vals = []
    for lr in adam_lrs:
        val_acc, test_acc = train_pipeline.main([
            "--dataset", dataset,
            "--model", model,
            "--optimizer", "adam",
            "--lr", str(lr),
            "--epochs", "20",
            "--batch-size", "64",
            "--log-dir", "./lr_tuning_results",
            "--seed", "42",
            "--scheduler", "cosine",
            "--device", "cuda:1"
        ])
        adam_vals.append(val_acc)
        print(f"Learning rate: {lr}, Validation accuracy: {val_acc}\n")
        f.write(f"Adam Learning rate: {lr}, Validation accuracy: {val_acc}\n")
    best_adam_lr = adam_lrs[adam_vals.index(max(adam_vals))]
    print("Best Adam learning rate:", best_adam_lr)
    f.write(f"Best Adam learning rate: {best_adam_lr}\n")

    adamw_vals = []
    for lr in adamw_lrs:
        val_acc, test_acc = train_pipeline.main([
            "--dataset", dataset,
            "--model", model,
            "--optimizer", "adamw",
            "--lr", str(lr),
            "--epochs", "20",
            "--batch-size", "64",
            "--log-dir", "./lr_tuning_results",
            "--seed", "42",
            "--scheduler", "cosine",
            "--device", "cuda:1"
        ])
        adamw_vals.append(val_acc)
        print(f"Learning rate: {lr}, Validation accuracy: {val_acc}")
        f.write(f"AdamW Learning rate: {lr}, Validation accuracy: {val_acc}\n")
    best_adamw_lr = adamw_lrs[adamw_vals.index(max(adamw_vals))]
    print("Best AdamW learning rate:", best_adamw_lr)
    f.write(f"Best AdamW learning rate: {best_adamw_lr}\n")