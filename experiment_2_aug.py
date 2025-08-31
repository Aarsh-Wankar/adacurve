import os
import torch
import train_pipeline
import csv

adam_filename = "./results_adam_aug.csv"
adamw_filename = "./results_adamw_aug.csv"

adacurve_lr = 0.01
adahessian_lr = 0.15
adam_lr = 1e-3
adamw_lr = 1e-3

model = "resnet18"
dataset = "cifar10"


with open(adam_filename, "w", newline="") as adam_f, open(adamw_filename, "w", newline="") as adamw_f:
    adam_csv_writer = csv.writer(adam_f)
    adam_csv_writer.writerow(["Seed", "adam Val Acc", "adam Test Acc"])

    adamw_csv_writer = csv.writer(adamw_f)
    adamw_csv_writer.writerow(["Seed", "Adamw Val Acc", "Adamw Test Acc"])

    for seed in range(0, 5):
        adam_min_val_acc, adam_test_acc = train_pipeline.main([
            "--dataset", dataset,
            "--model", model,
            "--optimizer", "adam",
            "--lr", str(adam_lr),
            "--epochs", "40",
            "--batch-size", "64",
            "--log-dir", "./experiment_results_aug",
            "--seed", str(seed),
            "--scheduler", "cosine",
            "--device", "cuda:1",
            "--augmentation", "basic"
        ])


        adam_csv_writer.writerow([seed, adam_min_val_acc, adam_test_acc])
        print(f"Seed: {seed}, adam Val Acc: {adam_min_val_acc}, adam Test Acc: {adam_test_acc}")
        adam_f.flush()

        adamw_min_val_acc, adamw_test_acc = train_pipeline.main([
            "--dataset", dataset,
            "--model", model,
            "--optimizer", "adamw",
            "--lr", str(adamw_lr),
            "--epochs", "40",
            "--batch-size", "64",
            "--log-dir", "./experiment_results_aug",
            "--seed", str(seed),
            "--scheduler", "cosine",
            "--device", "cuda:1",
            "--augmentation", "basic"
        ])


        adamw_csv_writer.writerow([seed, adamw_min_val_acc, adamw_test_acc])
        print(f"Seed: {seed}, Adamw Val Acc: {adamw_min_val_acc}, Adamw Test Acc: {adamw_test_acc}")
        adamw_f.flush()