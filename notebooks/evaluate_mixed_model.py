import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.model_0929 import DeepSets_varSets_forDiagnel, collate_fn, varSets_Datasets
from utils.formate_matrix_toMLData import formate_dataMatrix, matrix_trimer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate mixed model on discrete test data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Volumes/TRANSCEND/datas/discrete_test"),
        help="Directory containing formatted matrices (CSV/TXT).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("../model_weights/mixed_distribution/mixed_0929.pth"),
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="Computation device. 'auto' selects CUDA>MPS>CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation DataLoader.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to save prediction/results dataframe as CSV.",
    )
    return parser.parse_args()


def select_device(preference: str = "auto") -> torch.device:
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    if preference == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available.")
    if preference == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_data_files(directory: Path) -> Iterable[Path]:
    valid_ext = {".csv", ".txt"}
    ignored_prefixes = ("._", ".DS_Store", "Thumbs.db")
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith(ignored_prefixes):
            continue
        if entry.suffix.lower() not in valid_ext:
            continue
        if entry.is_file():
            yield entry


def load_datasets(data_dir: Path) -> Tuple[list, list, list]:
    test_states = []
    test_del_t = []
    test_targets = []
    formater = formate_dataMatrix()

    for file_path in iter_data_files(data_dir):
        print(f"Processing: {file_path}")
        try:
            with open(file_path, "rb") as f:
                all_matrix = np.loadtxt(f, delimiter=",")
        except Exception as exc:
            print(f"❌ Skipping file {file_path}: {exc}")
            continue

        tm = matrix_trimer(all_matrix)
        trm = tm.trim_transitionRateMatrix()
        data = tm.trim_data()
        output_vec = np.array(formater.GetOutputVector_byDiagonal(trm))

        state = np.stack([data[:, 0], data[:, 1]], axis=0)
        test_states.append(state)
        test_del_t.append(data[:, 2])
        test_targets.append(output_vec)

    return test_states, test_del_t, test_targets


class AllLifespanLoss(nn.Module):
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        epsilon = 1e-12
        y_pred_inverse = 1.0 / (outputs + epsilon)
        y_true_inverse = 1.0 / (targets + epsilon)
        loss_tensor = torch.abs(y_pred_inverse - y_true_inverse)[0]
        return loss_tensor, y_true_inverse[0], y_pred_inverse


def run_inference(
    dataloader: DataLoader, model: nn.Module, device: torch.device
) -> pd.DataFrame:
    criterion = AllLifespanLoss()
    true_lifespan = []
    pred_lifespan = []
    loss_values = []

    model.eval()
    dtype_long = torch.long
    dtype_float = torch.float32

    with torch.no_grad():
        for states, delta_t, targets, lengths in dataloader:
            states = states.to(device, non_blocking=True).to(dtype_long)
            delta_t = delta_t.to(device, non_blocking=True).to(dtype_float)
            targets = targets.to(device, non_blocking=True).to(dtype_float)
            lengths = lengths.to(device, non_blocking=True).to(dtype_long)

            outputs = model(states, delta_t, lengths)[0]
            loss_tensor, true_expect, pred_expect = criterion(outputs, targets)

            true_expect_cpu = true_expect.detach().cpu()
            pred_expect_cpu = pred_expect.detach().cpu()
            batch_loss = torch.abs(true_expect_cpu - pred_expect_cpu)

            true_lifespan.extend(true_expect_cpu.tolist())
            pred_lifespan.extend(pred_expect_cpu.tolist())
            loss_values.extend(batch_loss.tolist())

    return pd.DataFrame(
        {
            "true": true_lifespan,
            "pred": pred_lifespan,
            "loss": loss_values,
        }
    )


def summarise_results(df: pd.DataFrame) -> None:
    p95 = np.percentile(df["loss"], 95) if not df.empty else float("nan")
    mae = float(np.mean(df["loss"])) if not df.empty else float("nan")
    print(f"Samples evaluated: {len(df)}")
    print(f"MAE (absolute diff): {mae:.4f}")
    print(f"95th percentile error: {p95:.4f}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    print(f"Loading data from {data_dir}")
    states, del_t, targets = load_datasets(data_dir)
    dataset = varSets_Datasets(states, del_t, targets)

    use_cuda = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=use_cuda,
    )

    device = select_device(args.device)
    print(f"Using device: {device}")

    model = DeepSets_varSets_forDiagnel(device=device)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    df = run_inference(dataloader, model, device)
    summarise_results(df)

    if args.output_csv:
        output_path = args.output_csv.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
