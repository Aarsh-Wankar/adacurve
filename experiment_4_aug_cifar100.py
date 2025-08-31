import os
import torch
import train_pipeline
import csv


adacurve_lr = 0.01
adahessian_lr = 0.15
adam_lr = 1e-3
adamw_lr = 1e-3

model = "resnet18"
dataset = "cifar100"

adacurve_filename = "./results_adacurve_aug_cifar100.csv"
adahessian_filename = "./results_adahessian_aug_cifar100.csv"

with open(adahessian_filename, "w", newline="") as adahessian_f:
    # adacurve_csv_writer = csv.writer(adacurve_f)
    # adacurve_csv_writer.writerow(["Seed", "Adacurve Val Acc", "Adacurve Test Acc"])

    adahessian_csv_writer = csv.writer(adahessian_f)
    adahessian_csv_writer.writerow(["Seed", "Adahessian Val Acc", "Adahessian Test Acc"])

    for seed in range(0, 5):
        # adacurve_min_val_acc, adacurve_test_acc = train_pipeline.main([
        #     "--dataset", dataset,
        #     "--model", model,
        #     "--optimizer", "adacurve",
        #     "--lr", str(adacurve_lr),
        #     "--epochs", "20",
        #     "--batch-size", "64",
        #     "--log-dir", "./experiment_results",
        #     "--seed", str(seed),
        #     "--scheduler", "cosine",
        #     "--device", "cuda:0"
        # ])


        # adacurve_csv_writer.writerow([seed, adacurve_min_val_acc, adacurve_test_acc])
        # print(f"Seed: {seed}, Adacurve Val Acc: {adacurve_min_val_acc}, Adacurve Test Acc: {adacurve_test_acc}")

        adahessian_min_val_acc, adahessian_test_acc = train_pipeline.main([
            "--dataset", dataset,
            "--model", model,
            "--optimizer", "adahessian",
            "--lr", str(adahessian_lr),
            "--epochs", "50",
            "--batch-size", "64",
            "--log-dir", "./experiment_results_aug_cifar100",
            "--seed", str(seed),
            "--scheduler", "cosine",
            "--device", "cuda:1",
            "--augmentation", "basic"
        ])


        adahessian_csv_writer.writerow([seed, adahessian_min_val_acc, adahessian_test_acc])
        adahessian_f.flush()
        print(f"Seed: {seed}, Adahessian Val Acc: {adahessian_min_val_acc}, Adahessian Test Acc: {adahessian_test_acc}")
