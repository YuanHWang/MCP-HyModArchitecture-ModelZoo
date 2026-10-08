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
    MCPBRNN_Generic_PETconstraint_Three_VariantOutputGate,
    MCPBRNN_GWVariant_Routing,
    MCPBRNN_SW_Variant_Routing,
)


SPLIT_NAMES = {
    -99999: "spinup",
    -1: "train",
    0: "selection",
    1: "testing",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pretrained MA6 three-output main MCP with "
            "direct quick flow plus SW and GW routing."
        )
    )

    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--time_lag", type=int, default=0)

    parser.add_argument("--c_mean", type=float, default=412.9139085)
    parser.add_argument("--c_std", type=float, default=77.49943867)

    parser.add_argument("--sw_mean", type=float, default=3.051248249)
    parser.add_argument("--sw_std", type=float, default=3.365261828)

    parser.add_argument("--gw_mean", type=float, default=1.87176748)
    parser.add_argument("--gw_std", type=float, default=2.085571043)

    parser.add_argument(
        "--initial_gw_storage",
        type=float,
        default=1.6871063,
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
    )

    return parser.parse_args()


def resolve_path(value, default_path):
    path = Path(value) if value else default_path

    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()

    return path


def resolve_device(name):
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(name)


def safe_value(value):
    return -99999 if np.isnan(value) else float(value)


def calculate_metrics(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()

    kge, corr, kge_a, kge_b, kgess = KGE(
        sim,
        obs,
    )

    return {
        "NSE": float(NS(sim, obs)),
        "KGE": safe_value(kge),
        "KGE-A": float(kge_a),
        "KGE-B": float(kge_b),
        "Corr": safe_value(corr),
        "MSE": float(
            mean_squared_error(obs, sim)
        ),
        "KGEss": safe_value(kgess),
    }


def build_time_axis(n_rows):
    spinup_dates = pd.date_range(
        "1948-10-01",
        "1949-09-30",
        freq="D",
    )

    simulation_dates = pd.date_range(
        "1948-10-01",
        "1988-09-30",
        freq="D",
    )

    time = []
    phase = []

    for cycle in range(1, 4):
        time.extend(
            spinup_dates.strftime("%Y-%m-%d")
        )

        phase.extend(
            [f"spinup{cycle}"] * len(spinup_dates)
        )

    time.extend(
        simulation_dates.strftime("%Y-%m-%d")
    )

    phase.extend(
        ["simulation"] * len(simulation_dates)
    )

    if len(time) != n_rows:
        raise ValueError(
            f"Expected {len(time)} records for "
            "3 x WY1949 spinup + WY1949-WY1988, "
            f"but found {n_rows}."
        )

    return time, phase


def save_timeseries(
    path,
    time,
    phase,
    tensor,
    column_name,
):
    values = (
        tensor.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    pd.DataFrame(
        {
            "time": time,
            column_name: values,
            "phase": phase,
        }
    ).to_csv(
        path,
        index=False,
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
        raise ValueError(
            "Forcing/flow data and skill flags "
            "have different lengths."
        )

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

    flag_t = torch.tensor(
        flags.to_numpy(),
        device=device,
    )

    masks = {
        "train": flag_t.eq(-1).unsqueeze(1),
        "selection": flag_t.eq(0).unsqueeze(1),
        "testing": flag_t.eq(1).unsqueeze(1),
        "spinup": flag_t.eq(-99999).unsqueeze(1),
    }

    return (
        data,
        flags,
        x,
        y,
        masks,
    )


def evaluate_splits(
    predictions,
    y,
    masks,
):
    rows = []

    for split in (
        "train",
        "selection",
        "testing",
        "spinup",
    ):
        sim = torch.masked_select(
            predictions,
            masks[split],
        ).unsqueeze(1)

        obs = torch.masked_select(
            y,
            masks[split],
        ).unsqueeze(1)

        row = {
            "split": split,
            "n": int(sim.numel()),
        }

        row.update(
            calculate_metrics(sim, obs)
        )

        rows.append(row)

    return pd.DataFrame(rows)


class Model(nn.Module):
    """MA6 wrapper retaining original checkpoint module names."""

    def __init__(
        self,
        spin_len,
        train_time_len,
    ):
        super().__init__()

        self.MCPBRNNNode = (
            MCPBRNN_Generic_PETconstraint_Three_VariantOutputGate(
                input_size=1,
                hidden_size=1,
                gate_dim=1,
                spinLen=spin_len,
                traintimeLen=train_time_len,
                initial_forget_bias=0,
            )
        )

        self.MCPBRNNNode_SW = MCPBRNN_SW_Variant_Routing(
            input_size=1,
            hidden_size=1,
            gate_dim=1,
            spinLen=spin_len,
            traintimeLen=train_time_len,
            initial_forget_bias=0,
        )

        self.MCPBRNNNode_GW = MCPBRNN_GWVariant_Routing(
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
        time_lag,
        c_mean,
        c_std,
        sw_mean,
        sw_std,
        gw_mean,
        gw_std,
        initial_gw_storage,
    ):
        main = self.MCPBRNNNode(
            x,
            0,
            time_lag,
            c_mean,
            c_std,
        )

        surface = self.MCPBRNNNode_SW(
            main[0],
            0,
            time_lag,
            sw_mean,
            sw_std,
        )

        groundwater = self.MCPBRNNNode_GW(
            main[6],
            0,
            time_lag,
            gw_mean,
            gw_std,
            initial_gw_storage,
        )

        total_discharge = (
            main[1]
            + surface[0]
            + groundwater[0]
        )

        return (
            total_discharge,
            main[0],
            surface[0],
            groundwater[0],
            main[2],
            surface[1],
            groundwater[1],
            main[3],
            main[4],
            main[6],
            main[5],
            main[7],
            main[8],
            surface[2],
            groundwater[2],
            main[10],
            main[11],
            main[12],
            surface[3],
            groundwater[3],
            main[13],
            main[1],
            main[9],
        )


def main():
    args = parse_args()

    if args.c_std == 0:
        raise ValueError(
            "c_std must be non-zero."
        )

    if args.sw_std == 0:
        raise ValueError(
            "sw_std must be non-zero."
        )

    if args.gw_std == 0:
        raise ValueError(
            "gw_std must be non-zero."
        )

    device = resolve_device(args.device)

    checkpoint = resolve_path(
        args.checkpoint,
        SCRIPT_DIR / "model_epoch469.pt",
    )

    data_dir = resolve_path(
        args.data_dir,
        PROJECT_DIR / "20220527-MDUPLEX-LeafRiver",
    )

    output_dir = resolve_path(
        args.output_dir,
        SCRIPT_DIR / "evaluation_MA6",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    spin_len = 1095 - args.time_lag
    train_time_len = 8400 - args.time_lag

    data, flags, x, y, masks = load_data(
        data_dir,
        device,
    )

    model = Model(
        spin_len,
        train_time_len,
    ).to(device)

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
        ),
        strict=True,
    )

    model.eval()

    with torch.no_grad():
        result = model(
            x,
            args.time_lag,
            args.c_mean,
            args.c_std,
            args.sw_mean,
            args.sw_std,
            args.gw_mean,
            args.gw_std,
            args.initial_gw_storage,
        )

    predictions = result[0]

    metrics = evaluate_splits(
        predictions,
        y,
        masks,
    )

    time, phase = build_time_axis(
        x.shape[0]
    )

    timeseries = pd.DataFrame(
        {
            "time": time,
            "Qsim": (
                predictions
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            ),
            "phase": phase,
            "P": data["P"].to_numpy(),
            "PET": data["PET"].to_numpy(),
            "Q": data["Q"].to_numpy(),
            "Flag": flags.to_numpy(),
        }
    )

    timeseries["Split"] = (
        timeseries["Flag"].map(SPLIT_NAMES)
    )

    timeseries.to_csv(
        output_dir
        / "evaluation_timeseries.csv",
        index=False,
    )

    metrics.to_csv(
        output_dir
        / "evaluation_metrics.csv",
        index=False,
    )

    diagnostic_series = {
        "Qsim.csv": (
            result[0],
            "Qsim",
        ),
        "main_surface_output.csv": (
            result[1],
            "main_surface_output",
        ),
        "surface_routing_output.csv": (
            result[2],
            "surface_routing_output",
        ),
        "groundwater_output.csv": (
            result[3],
            "groundwater_output",
        ),
        "main_storage.csv": (
            result[4],
            "main_storage",
        ),
        "surface_routing_storage.csv": (
            result[5],
            "surface_routing_storage",
        ),
        "groundwater_storage.csv": (
            result[6],
            "groundwater_storage",
        ),
        "loss_unconstrained.csv": (
            result[7],
            "loss_unconstrained",
        ),
        "loss_constrained.csv": (
            result[8],
            "loss_constrained",
        ),
        "recharge.csv": (
            result[9],
            "recharge",
        ),
        "bypass.csv": (
            result[10],
            "bypass",
        ),
        "gate_input.csv": (
            result[11],
            "gate_input",
        ),
        "main_surface_output_gate.csv": (
            result[12],
            "main_surface_output_gate",
        ),
        "surface_routing_output_gate.csv": (
            result[13],
            "surface_routing_output_gate",
        ),
        "groundwater_output_gate.csv": (
            result[14],
            "groundwater_output_gate",
        ),
        "loss_gate.csv": (
            result[15],
            "loss_gate",
        ),
        "loss_gate_constrained.csv": (
            result[16],
            "loss_gate_constrained",
        ),
        "main_remember_gate.csv": (
            result[17],
            "main_remember_gate",
        ),
        "surface_routing_remember_gate.csv": (
            result[18],
            "surface_routing_remember_gate",
        ),
        "groundwater_remember_gate.csv": (
            result[19],
            "groundwater_remember_gate",
        ),
        "recharge_gate.csv": (
            result[20],
            "recharge_gate",
        ),
        "quickflow.csv": (
            result[21],
            "quickflow",
        ),
        "quickflow_gate.csv": (
            result[22],
            "quickflow_gate",
        ),
    }

    for filename, (
        tensor,
        column_name,
    ) in diagnostic_series.items():
        save_timeseries(
            output_dir / filename,
            time,
            phase,
            tensor,
            column_name,
        )

    print(
        metrics.to_string(index=False)
    )

    print(
        f"\nCheckpoint: {checkpoint}"
    )

    print(
        f"Outputs:    {output_dir}"
    )


if __name__ == "__main__":
    main()
