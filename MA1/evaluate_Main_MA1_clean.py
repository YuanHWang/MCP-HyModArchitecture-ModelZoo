import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from MCPBRNN_lib_tools.Eval_Metric import KGE, NS  # noqa: E402
from MCPBRNN_lib_tools.MA_Zoo import (  # noqa: E402
    MCPBRNN_Generic_PETconstraint_Scaling,
)


SPLIT_NAMES = {
    -99999: "spinup",
    -1: "train",
    0: "selection",
    1: "testing",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained MA1 single-node MCP with PET-constrained scaling."
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--time_lag", type=int, default=0)
    parser.add_argument("--c_mean", type=float, default=412.9139085)
    parser.add_argument("--c_std", type=float, default=77.49943867)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return parser.parse_args()


def resolve_path(value, default_path):
    path = Path(value) if value else default_path
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def safe_value(value):
    return -99999 if np.isnan(value) else float(value)


def calculate_metrics(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()
    kge, corr, kge_a, kge_b, kgess = KGE(sim, obs)
    return {
        "NSE": float(NS(sim, obs)),
        "KGE": safe_value(kge),
        "KGE-A": float(kge_a),
        "KGE-B": float(kge_b),
        "Corr": safe_value(corr),
        "MSE": float(mean_squared_error(obs, sim)),
        "KGEss": safe_value(kgess),
    }


def build_time_axis(n_rows):
    spinup_dates = pd.date_range("1948-10-01", "1949-09-30", freq="D")
    simulation_dates = pd.date_range("1948-10-01", "1988-09-30", freq="D")

    time = []
    phase = []
    for cycle in range(1, 4):
        time.extend(spinup_dates.strftime("%Y-%m-%d"))
        phase.extend([f"spinup{cycle}"] * len(spinup_dates))

    time.extend(simulation_dates.strftime("%Y-%m-%d"))
    phase.extend(["simulation"] * len(simulation_dates))

    if len(time) != n_rows:
        raise ValueError(
            f"Expected {len(time)} daily records for 3 x WY1949 spinup + "
            f"WY1949-WY1988, but found {n_rows}."
        )
    return time, phase


def save_timeseries(path, time, phase, tensor, column_name):
    values = tensor.detach().cpu().numpy().reshape(-1)
    pd.DataFrame({"time": time, column_name: values, "phase": phase}).to_csv(
        path, index=False
    )


def load_data(data_dir, device):
    data = pd.read_csv(
        data_dir / "LeafRiverDaily_43YR.txt",
        header=None,
        sep=r"\s+",
        names=["P", "PET", "Q"],
    )
    flags = pd.read_csv(
        data_dir / "LeafRiverDaily_43YR_Flag.txt",
        header=None,
        sep=r"\s+",
        names=["Flag"],
    )["Flag"]

    if len(data) != len(flags):
        raise ValueError("Forcing/flow data and skill flags have different lengths.")

    x = torch.tensor(
        data[["P", "PET"]].to_numpy(), dtype=torch.float32, device=device
    ).unsqueeze(1)
    y = torch.tensor(data[["Q"]].to_numpy(), dtype=torch.float32, device=device)
    flag_t = torch.tensor(flags.to_numpy(), device=device)

    masks = {
        "train": flag_t.eq(-1).unsqueeze(1),
        "selection": flag_t.eq(0).unsqueeze(1),
        "testing": flag_t.eq(1).unsqueeze(1),
        "spinup": flag_t.eq(-99999).unsqueeze(1),
    }
    return data, flags, x, y, masks


def evaluate_splits(predictions, y, masks):
    rows = []
    for split in ("train", "selection", "testing", "spinup"):
        sim = torch.masked_select(predictions, masks[split]).unsqueeze(1)
        obs = torch.masked_select(y, masks[split]).unsqueeze(1)
        row = {"split": split, "n": int(sim.numel())}
        row.update(calculate_metrics(sim, obs))
        rows.append(row)
    return pd.DataFrame(rows)


class Model(nn.Module):
    """Thin wrapper retained so original checkpoint keys remain compatible."""

    def __init__(self, spin_len, train_time_len):
        super().__init__()
        self.MCPBRNNNode = MCPBRNN_Generic_PETconstraint_Scaling(
            input_size=1,
            hidden_size=1,
            gate_dim=1,
            spinLen=spin_len,
            traintimeLen=train_time_len,
            initial_forget_bias=0,
        )

    def forward(self, x, time_lag, c_mean, c_std):
        return self.MCPBRNNNode(x, 0, time_lag, c_mean, c_std)


def main():
    args = parse_args()
    if args.c_std == 0:
        raise ValueError("c_std must be non-zero.")

    device = resolve_device(args.device)
    checkpoint = resolve_path(args.checkpoint, SCRIPT_DIR / "model_epoch27.pt")
    data_dir = resolve_path(args.data_dir, PROJECT_DIR / "20220527-MDUPLEX-LeafRiver")
    output_dir = resolve_path(args.output_dir, SCRIPT_DIR / "evaluation_MA1")
    output_dir.mkdir(parents=True, exist_ok=True)

    spin_len = 1095 - args.time_lag
    train_time_len = 8400 - args.time_lag
    data, flags, x, y, masks = load_data(data_dir, device)

    model = Model(spin_len, train_time_len).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()

    with torch.no_grad():
        result = model(x, args.time_lag, args.c_mean, args.c_std)

    predictions = result[0]
    metrics = evaluate_splits(predictions, y, masks)

    time, phase = build_time_axis(x.shape[0])
    timeseries = pd.DataFrame(
        {
            "time": time,
            "Qsim": predictions.detach().cpu().numpy().reshape(-1),
            "phase": phase,
            "P": data["P"].to_numpy(),
            "PET": data["PET"].to_numpy(),
            "Q": data["Q"].to_numpy(),
            "Flag": flags.to_numpy(),
        }
    )
    timeseries["Split"] = timeseries["Flag"].map(SPLIT_NAMES)
    timeseries.to_csv(output_dir / "evaluation_timeseries.csv", index=False)
    metrics.to_csv(output_dir / "evaluation_metrics.csv", index=False)

    diagnostic_series = {
        "Qsim.csv": (result[0], "Qsim"),
        "storage.csv": (result[1], "storage"),
        "loss_unconstrained.csv": (result[2], "loss_unconstrained"),
        "loss_constrained.csv": (result[3], "loss_constrained"),
        "bypass.csv": (result[4], "bypass"),
        "gate_input.csv": (result[5], "gate_input"),
        "gate_output.csv": (result[6], "gate_output"),
        "gate_loss_unconstrained.csv": (result[7], "gate_loss_unconstrained"),
        "gate_loss_constrained.csv": (result[8], "gate_loss_constrained"),
        "gate_remember.csv": (result[9], "gate_remember"),
    }
    for filename, (tensor, column_name) in diagnostic_series.items():
        save_timeseries(output_dir / filename, time, phase, tensor, column_name)

    print(metrics.to_string(index=False))
    print(f"\nCheckpoint: {checkpoint}")
    print(f"Outputs:    {output_dir}")


if __name__ == "__main__":
    main()
