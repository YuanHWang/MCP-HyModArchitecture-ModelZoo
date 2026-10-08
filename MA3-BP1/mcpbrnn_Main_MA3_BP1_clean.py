import argparse
import copy
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
from MCPBRNN_lib_tools.Loss_Function import KGELoss  # noqa: E402
from MCPBRNN_lib_tools.MA_Zoo import (  # noqa: E402
    MCPBRNN_Generic_PETconstraint_Scaling_BYPASSM0,
    MCPBRNN_SW_Variant_Routing,
)


PARAMETER_COLUMNS = [
    "MCPBRNNNode.weight_r_yom",
    "MCPBRNNNode.weight_r_ylm",
    "MCPBRNNNode.weight_r_yfm",
    "MCPBRNNNode.bias_b0_yom",
    "MCPBRNNNode.weight_b1_yom",
    "MCPBRNNNode.bias_b0_ylm",
    "MCPBRNNNode.weight_b2_ylm",
    "MCPBRNNNode.theltaC",
    "MCPBRNNNode_SW.weight_r_yom",
    "MCPBRNNNode_SW.weight_r_yfm",
    "MCPBRNNNode_SW.bias_b0_yom",
    "MCPBRNNNode_SW.weight_b1_yom",
]
METRIC_NAMES = ["NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss"]
SPLITS = ("train", "selection", "testing", "spinup")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue training MA3-BP1: threshold-bypass main MCP followed by one routing MCP node."
    )
    parser.add_argument("--case_no", type=int, default=0)
    parser.add_argument("--epoch_no", type=int, default=3000)
    parser.add_argument("--time_lag", type=int, default=0)
    parser.add_argument("--seed_no", type=int, default=2925)

    parser.add_argument("--c_mean", type=float, default=412.9139085)
    parser.add_argument("--c_std", type=float, default=77.49943867)

    parser.add_argument("--sw_mean", type=float, default=1.673117957)
    parser.add_argument("--sw_std", type=float, default=3.592955064)

    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return parser.parse_args()


def resolve_path(value, default_path):
    path = Path(value) if value else default_path
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path

def resolve_checkpoint(value):
    if value:
        return resolve_path(value, SCRIPT_DIR)

    checkpoints = sorted(SCRIPT_DIR.glob("*.pt"))

    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one .pt checkpoint in {SCRIPT_DIR}, "
            f"but found {len(checkpoints)}."
        )

    return checkpoints[0]


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
        "mse": float(mean_squared_error(obs, sim)),
        "KGEss": safe_value(kgess),
    }


def calculate_lag_kge(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()
    n = obs.shape[0]

    scores = []
    for lag in (1, 2, 3):
        kge, *_ = KGE(sim[lag:n], obs[: n - lag])
        scores.append(safe_value(kge))
    return scores


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
            f"Expected {len(time)} records for 3 x WY1949 spinup + WY1949-WY1988, "
            f"but found {n_rows}."
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
        data[["P", "PET"]].to_numpy(),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)

    y = torch.tensor(
        data[["Q"]].to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    flag_t = torch.tensor(flags.to_numpy(), device=device)

    masks = {
        "train": flag_t.eq(-1).unsqueeze(1),
        "selection": flag_t.eq(0).unsqueeze(1),
        "testing": flag_t.eq(1).unsqueeze(1),
        "spinup": flag_t.eq(-99999).unsqueeze(1),
    }

    return x, y, masks


class Model(nn.Module):
    """MA3-BP1 wrapper retaining original checkpoint module names."""

    def __init__(self, spin_len, train_time_len):
        super().__init__()

        self.MCPBRNNNode = MCPBRNN_Generic_PETconstraint_Scaling_BYPASSM0(
            input_size=1,
            hidden_size=1,
            gate_dim=1,
            spinLen=spin_len,
            traintimeLen=train_time_len,
            initial_forget_bias=0,
        )

        self.MCPBRNNNode_SW = MCPBRNN_SW_Variant_Routing(
            input_size=1,
            hidden_size=1,
            gate_dim=1,
            spinLen=spin_len,
            traintimeLen=train_time_len,
            initial_forget_bias=0,
        )

    def forward(
        self,
        x,
        epoch,
        time_lag,
        c_mean,
        c_std,
        sw_mean,
        sw_std,
    ):
        main = self.MCPBRNNNode(
            x,
            epoch,
            time_lag,
            c_mean,
            c_std,
        )

        # main[0] already includes bypass, matching the original MA3-BP1.
        routing = self.MCPBRNNNode_SW(
            main[0],
            epoch,
            time_lag,
            sw_mean,
            sw_std,
        )

        return (
            routing[0],  # system discharge
            main[0],     # main-node discharge = MCP output + bypass
            routing[1],  # routing storage
            main[1],     # main storage
            main[2],     # unconstrained loss
            main[3],     # constrained loss
            main[4],     # bypass
            main[5],     # bypass/input gate
            routing[2],  # routing output gate
            main[6],     # main output gate
            main[7],     # main loss gate
            main[8],     # constrained main loss gate
            routing[3],  # routing remember gate
            main[9],     # main remember gate
        )


def main():
    args = parse_args()

    if args.epoch_no < 1:
        raise ValueError("epoch_no must be at least 1.")
    if args.c_std == 0:
        raise ValueError("c_std must be non-zero.")
    if args.sw_std == 0:
        raise ValueError("sw_std must be non-zero.")

    np.random.seed(args.seed_no)
    torch.manual_seed(args.seed_no)

    device = resolve_device(args.device)

    spin_len = 1095 - args.time_lag
    train_time_len = 8400 - args.time_lag

    checkpoint = resolve_checkpoint(args.checkpoint)

    data_dir = resolve_path(
        args.data_dir,
        PROJECT_DIR / "20220527-MDUPLEX-LeafRiver",
    )

    case_name = f"MA3-BP1_{args.case_no}"
    case_dir = resolve_path(args.output_dir, SCRIPT_DIR / case_name)
    case_dir.mkdir(parents=True, exist_ok=True)

    x, y, masks = load_data(data_dir, device)

    model = Model(spin_len, train_time_len).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device),
        strict=True,
    )

    loss_func = KGELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    learning_rates = {300: 0.0125, 600: 0.0125}

    history = []
    best_epoch = None
    best_kge = -np.inf
    best_state = None

    for epoch in range(1, args.epoch_no + 1):
        if epoch in learning_rates:
            for group in optimizer.param_groups:
                group["lr"] = learning_rates[epoch]

        model.train()
        optimizer.zero_grad()

        discharge = model(
            x,
            epoch,
            args.time_lag,
            args.c_mean,
            args.c_std,
            args.sw_mean,
            args.sw_std,
        )[0]

        sim_train = torch.masked_select(
            discharge,
            masks["train"],
        ).unsqueeze(1)

        obs_train = torch.masked_select(
            y,
            masks["train"],
        ).unsqueeze(1)

        loss = loss_func(sim_train, obs_train)
        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            result = model(
                x,
                epoch,
                args.time_lag,
                args.c_mean,
                args.c_std,
                args.sw_mean,
                args.sw_std,
            )

            discharge = result[0]

            split_tensors = {}
            row = {"epoch": epoch}

            for split in SPLITS:
                sim = torch.masked_select(
                    discharge,
                    masks[split],
                ).unsqueeze(1)

                obs = torch.masked_select(
                    y,
                    masks[split],
                ).unsqueeze(1)

                split_tensors[split] = (sim, obs)
                metrics = calculate_metrics(sim, obs)

                suffix = "" if split == "train" else f"_{split}"

                for metric_name in METRIC_NAMES:
                    row[f"{metric_name}{suffix}"] = metrics[metric_name]

        state = model.state_dict()

        for name in PARAMETER_COLUMNS:
            row[name] = state[name].detach().cpu().reshape(-1)[0].item()

        for lag, score in zip(
            (1, 2, 3),
            calculate_lag_kge(*split_tensors["train"]),
        ):
            row[f"KGEtimelag_{lag}"] = score

        history.append(row)

        if row["KGE_selection"] > best_kge:
            best_kge = row["KGE_selection"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch}: "
            f"KGE_selection = {row['KGE_selection']:.6f}"
        )

    summary_columns = (
        ["epoch"]
        + PARAMETER_COLUMNS
        + METRIC_NAMES
        + [f"{name}_selection" for name in METRIC_NAMES]
        + [f"{name}_testing" for name in METRIC_NAMES]
        + [f"{name}_spinup" for name in METRIC_NAMES]
        + ["KGEtimelag_1", "KGEtimelag_2", "KGEtimelag_3"]
    )

    pd.DataFrame(history)[summary_columns].to_csv(
        case_dir / f"IC_caseno_{args.case_no}_summary.csv",
        index=False,
    )

    model.load_state_dict(best_state, strict=True)
    model.eval()

    with torch.no_grad():
        result = model(
            x,
            best_epoch,
            args.time_lag,
            args.c_mean,
            args.c_std,
            args.sw_mean,
            args.sw_std,
        )

    time, phase = build_time_axis(x.shape[0])

    output_series = {
        f"Outhidden_{args.case_no}_summary.csv": (
            result[0],
            "discharge",
        ),
        f"Outhidden_Main_{args.case_no}_summary.csv": (
            result[1],
            "main_discharge",
        ),
        f"Outcell_sw_{args.case_no}_summary.csv": (
            result[2],
            "routing_storage",
        ),
        f"Outcell_{args.case_no}_summary.csv": (
            result[3],
            "main_storage",
        ),
        f"Outloss_{args.case_no}_summary.csv": (
            result[4],
            "loss_unconstrained",
        ),
        f"Outlossc_{args.case_no}_summary.csv": (
            result[5],
            "loss_constrained",
        ),
        f"Outbypass_{args.case_no}_summary.csv": (
            result[6],
            "bypass",
        ),
        f"Out_gatei_{args.case_no}_summary.csv": (
            result[7],
            "gate_input",
        ),
        f"Out_gateo_sw_{args.case_no}_summary.csv": (
            result[8],
            "routing_output_gate",
        ),
        f"Out_gateo_{args.case_no}_summary.csv": (
            result[9],
            "main_output_gate",
        ),
        f"Out_gatel_{args.case_no}_summary.csv": (
            result[10],
            "main_loss_gate",
        ),
        f"Out_gatelc_{args.case_no}_summary.csv": (
            result[11],
            "main_loss_gate_constrained",
        ),
        f"Out_gatef_sw_{args.case_no}_summary.csv": (
            result[12],
            "routing_remember_gate",
        ),
        f"Out_gatef_{args.case_no}_summary.csv": (
            result[13],
            "main_remember_gate",
        ),
    }

    for filename, (tensor, column_name) in output_series.items():
        save_timeseries(
            case_dir / filename,
            time,
            phase,
            tensor,
            column_name,
        )

    pd.DataFrame(
        [best_epoch],
        columns=["Best_Epoch"],
    ).to_csv(
        case_dir / f"Best_Epoch_Caseno{args.case_no}_summary.csv",
        index=False,
    )

    final_checkpoint = case_dir / f"best_model_epoch{best_epoch}.pt"
    torch.save(best_state, final_checkpoint)

    print(f"Best epoch = {best_epoch}")
    print(f"Best KGE_selection = {best_kge:.6f}")
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Saved best model to: {final_checkpoint}")


if __name__ == "__main__":
    main()
