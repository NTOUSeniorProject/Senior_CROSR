"""GPU-only: python -m valid.train_ntu_cross_validation --help"""

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from constants import project_path
from common.models.ST_CROSR import ST_CROSR
from common.ntu_normalize import get_valid_mask
from training.center_loss import CenterLoss
from valid.cross_validation_config import CV_CONFIG, SUBJECT_GROUPS
from valid.ntu_cross_validation import (
    FoldDataset, calibrate, combined_scores, describe_records, normalize_samples,
    scan_samples, split_records, subject_split,
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def loader(records, config, shuffle=False):
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(
        FoldDataset(records, config["known_actions"], config["max_frames"], config["num_nodes"]),
        batch_size=config["batch_size"], shuffle=shuffle, num_workers=config["num_workers"],
        pin_memory=True, worker_init_fn=seed_worker, generator=generator,
    )


def prepare(inputs, config):
    inputs = inputs.cuda(non_blocking=True)
    mask = get_valid_mask(inputs)
    return normalize_samples(inputs, config["center_joint_idx"]), mask


@torch.no_grad()
def validation_accuracy(model, batches, config):
    model.eval()
    correct, total = 0, 0
    for inputs, labels, _ in batches:
        inputs, _ = prepare(inputs, config)
        logits, _, _, _ = model(inputs)
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        total += len(labels)
    return correct / total


@torch.no_grad()
def compute_centroids(model, batches, config):
    model.eval()
    sums = torch.zeros(len(config["known_actions"]), config["feat_dim"], device="cuda")
    counts = torch.zeros(len(sums), device="cuda")
    for inputs, labels, _ in batches:
        inputs, _ = prepare(inputs, config)
        _, _, features, _ = model(inputs)
        labels = labels.cuda(non_blocking=True)
        sums.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    if (counts == 0).any():
        raise ValueError("訓練集缺少類別，無法建立特徵中心")
    return F.normalize(sums / counts[:, None], dim=1)


@torch.no_grad()
def collect_scores(model, batches, centroids, config):
    model.eval()
    output = {key: [] for key in ("labels", "unknown", "predictions", "distance", "mse")}
    for inputs, labels, unknown in batches:
        inputs, mask = prepare(inputs, config)
        logits, reconstructed, features, _ = model(inputs)
        mse = (((reconstructed - inputs) ** 2) * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) + 1e-6)
        distance = 1 - (F.normalize(features, dim=1) @ centroids.T).max(1).values
        values = (labels, unknown, logits.argmax(1), distance, mse)
        for key, value in zip(output, values):
            output[key].extend(value.cpu().tolist())
    result = {key: np.asarray(value) for key, value in output.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("模型輸出包含 NaN/Inf")
    return result


def evaluate(scores, calibration, class_count):
    combined = combined_scores(scores["distance"], scores["mse"], calibration)
    predicted_unknown = combined > calibration["threshold"]
    known = scores["unknown"] == 0
    actual = scores["unknown"]
    return {
        "known_accuracy": float(accuracy_score(scores["labels"][known], scores["predictions"][known])),
        "known_macro_f1": float(f1_score(scores["labels"][known], scores["predictions"][known],
                                        labels=list(range(class_count)), average="macro", zero_division=0)),
        "unknown_auroc": float(roc_auc_score(actual, combined)),
        "distance_auroc": float(roc_auc_score(actual, scores["distance"])),
        "mse_auroc": float(roc_auc_score(actual, scores["mse"])),
        "unknown_precision": float(precision_score(actual, predicted_unknown, zero_division=0)),
        "unknown_recall": float(recall_score(actual, predicted_unknown, zero_division=0)),
        "unknown_f1": float(f1_score(actual, predicted_unknown, zero_division=0)),
        "test_known_fpr": float(predicted_unknown[known].mean()),
        "confusion_matrix_known_unknown": confusion_matrix(actual, predicted_unknown, labels=[0, 1]).tolist(),
    }, combined


def run_fold(fold, splits, config, directory):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    train = loader(splits["train"], config, shuffle=True)
    val = loader(splits["val"], config)
    write_json(directory / "manifest.json", {
        "fold": fold, "subjects": subject_split(fold), "config": config,
        "counts": {key: describe_records(rows, config["known_actions"]) for key, rows in splits.items()},
        "samples": splits,
    })
    model = ST_CROSR(len(config["known_actions"]), config["num_nodes"], config["max_frames"]).cuda()
    center = CenterLoss(len(config["known_actions"]), config["feat_dim"], use_gpu=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_model"], weight_decay=config["weight_decay"])
    center_optimizer = torch.optim.SGD(center.parameters(), lr=config["lr_center"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    best_accuracy = -1.0
    for epoch in range(config["num_epochs"]):
        model.train()
        totals = np.zeros(4)
        count = 0
        reconstruction_weight = config["max_lambda_recon"] * min(1.0, (epoch + 1) / config["warmup_epochs"])
        for batch_index, (inputs, labels, _) in enumerate(train):
            inputs, mask = prepare(inputs, config)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            center_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits, reconstructed, features, memory_weights = model(inputs)
                classification = F.cross_entropy(logits, labels)
                reconstruction = (((reconstructed.float() - inputs) ** 2) * mask).sum() / (mask.sum() + 1e-6)
                center_loss = center(features.float(), labels)
                sparse = -memory_weights.max(1).values.mean()
                loss = (classification + reconstruction_weight * reconstruction
                        + config["lambda_center"] * center_loss + config["lambda_entropy"] * sparse)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Fold {fold} epoch {epoch + 1} loss 非有限值")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(center_optimizer)
            scaler.update()
            totals += np.array([loss.item(), classification.item(), reconstruction.item(), center_loss.item()]) * len(labels)
            count += len(labels)
            if batch_index % 20 == 0:
                print(f"Fold {fold:02d} epoch {epoch + 1}/{config['num_epochs']} batch {batch_index + 1}/{len(train)} loss={loss.item():.5f}", flush=True)
        scheduler.step()
        accuracy = validation_accuracy(model, val, config)
        improved = accuracy > best_accuracy
        best_accuracy = max(best_accuracy, accuracy)
        state = {
            "fold": fold, "epoch": epoch, "config": config,
            "known_actions": config["known_actions"], "num_classes": len(config["known_actions"]),
            "model_state_dict": model.state_dict(), "center_state_dict": center.state_dict(),
            "optimizer_model_state_dict": optimizer.state_dict(),
            "optimizer_center_state_dict": center_optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "best_val_accuracy": best_accuracy, "normalization": "per_sample",
        }
        torch.save(state, directory / "last.pth")
        if improved:
            torch.save(state, directory / "best_val.pth")
        history = dict(zip(("loss", "classification", "reconstruction", "center"), (totals / count).tolist()))
        history.update(epoch=epoch + 1, validation_accuracy=accuracy)
        with (directory / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(history, allow_nan=False) + "\n")
        print(f"Fold {fold:02d} epoch {epoch + 1}: validation accuracy={accuracy:.4%}", flush=True)
    best = torch.load(directory / "best_val.pth", map_location="cuda", weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    centroids = compute_centroids(model, loader(splits["train"], config), config)
    validation = collect_scores(model, val, centroids, config)
    calibration = calibrate(validation["distance"], validation["mse"], config["known_fpr"], config["cosine_weight"])
    write_json(directory / "calibration.json", calibration)
    torch.save({**calibration, "centroids_norm": centroids.cpu(), "normalization": "per_sample",
                "known_actions": config["known_actions"], "fold": fold}, directory / "radar_meta_params.pth")
    test = collect_scores(model, loader(splits["test"], config), centroids, config)
    metrics, combined = evaluate(test, calibration, len(config["known_actions"]))
    np.savez_compressed(directory / "test_predictions.npz", **test, combined=combined,
                        paths=np.array([r["path"] for r in splits["test"]]))
    result = {"fold": fold, "best_epoch": best["epoch"] + 1, "best_val_accuracy": best_accuracy,
              "seconds": time.perf_counter() - started,
              "test_samples": len(splits["test"]), "metrics": metrics}
    write_json(directory / "metrics.json", result)
    print(f"Fold {fold:02d} complete: {metrics}", flush=True)
    return result


def summarize(results, output):
    keys = [key for key, value in results[0]["metrics"].items() if isinstance(value, (int, float))]
    aggregates = {}
    for key in keys:
        values = [r["metrics"][key] for r in results]
        aggregates[key] = {"mean": float(np.mean(values)),
                           "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    write_json(output / "summary.json", {"completed_folds": [r["fold"] for r in results],
               "complete_10_fold": len(results) == 10, "aggregation": "unweighted fold mean; sample std ddof=1",
               "metrics": aggregates, "folds": results})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fold", "test_samples", *keys])
        writer.writeheader()
        for result in results:
            writer.writerow({"fold": result["fold"], "test_samples": result["test_samples"],
                             **{key: result["metrics"][key] for key in keys}})


def main():
    parser = argparse.ArgumentParser(description="GPU-only NTU60 subject-disjoint 10-fold training")
    parser.add_argument("--data-root", default=CV_CONFIG["data_root"])
    parser.add_argument("--output", help="新的輸出目錄；已存在時拒絕覆寫")
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 11), default=list(range(1, 11)))
    parser.add_argument("--epochs", type=int, default=CV_CONFIG["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=CV_CONFIG["batch_size"])
    parser.add_argument("--num-workers", type=int, default=CV_CONFIG["num_workers"])
    parser.add_argument("--seed", type=int, default=CV_CONFIG["seed"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true", help="只檢查分組並輸出清單，不訓練")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0 or not 0 <= args.seed < 2**32:
        parser.error("epochs>=1, batch-size>=2, num-workers>=0, 0<=seed<2**32")
    if len(set(args.folds)) != len(args.folds):
        parser.error("folds 不可重複")
    if not args.prepare_only:
        if not torch.cuda.is_available():
            parser.error("此程式僅支援 CUDA GPU，請使用支援 CUDA 的 PyTorch 環境")
        if not 0 <= args.gpu < torch.cuda.device_count():
            parser.error("GPU 編號不存在")
        torch.cuda.set_device(args.gpu)
        torch.backends.cudnn.benchmark = False
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}", flush=True)
    config = dict(CV_CONFIG, data_root=str(project_path(args.data_root).resolve()),
                  num_epochs=args.epochs, batch_size=args.batch_size,
                  num_workers=args.num_workers, seed=args.seed)
    config["known_actions"] = sorted(config["known_actions"])
    records = scan_samples(config["data_root"])
    missing = set(range(1, 41)) - {r["subject"] for r in records}
    if missing:
        raise ValueError(f"十折實驗缺少人物：{sorted(missing)}")
    splits = {fold: split_records(records, fold, config["known_actions"]) for fold in args.folds}
    output = project_path(args.output) if args.output else Path(config["output_root"]) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", {
        "protocol": "NTU60 custom subject-disjoint 10-fold; 8 train / 1 known-only val / 1 test",
        "groups": SUBJECT_GROUPS, "config": config, "selected_folds": args.folds,
        "normalization": "per_sample", "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(args.gpu) if not args.prepare_only else None,
        "inventory": describe_records(records, config["known_actions"]),
    })
    results = []
    for fold, rows in splits.items():
        if args.prepare_only:
            write_json(output / f"fold_{fold:02d}_manifest.json", {
                "subjects": subject_split(fold), "samples": rows,
                "counts": {key: describe_records(value, config["known_actions"]) for key, value in rows.items()},
            })
        else:
            result = run_fold(fold, rows, config, output / f"fold_{fold:02d}")
            results.append(result)
            summarize(results, output)
            torch.cuda.empty_cache()
    print(f"輸出：{output}", flush=True)


if __name__ == "__main__":
    main()
