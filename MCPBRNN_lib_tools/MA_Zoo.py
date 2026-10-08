import torch
import torch.nn as nn
from torch import Tensor

class MCPBRNN_Generic_PETconstraint_Scaling(nn.Module):
    """MA1 single-node MCP with scaled output/loss gates
    and a PET-constrained loss flux.

    MA1 is equivalent to the M5 single-node MCP used in the
    preceding single-node study.

    The output gate varies with normalized storage.
    The unconstrained loss gate varies with normalized PET.

    The actual loss gate is constrained so that the loss flux
    cannot exceed available PET:

        gLc = min(gL, PET / storage)

    for positive storage.

    The remember gate is then defined by mass conservation:

        gR = 1 - gO - gLc

    The input gate/bypass is fixed to zero, so all precipitation
    enters storage.

    Parameter names and shapes are preserved for compatibility
    with the original M5 / MA1 checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MCP Zoo interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MA1 is a single-node MCP and requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names and shapes.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, c_mean, c_std):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("MA1 expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("MA1 requires precipitation and PET.")

        hidden_size = self.hidden_size

        # Initial storage.
        storage = x.new_zeros(1, hidden_size)

        # Pre-update states and fluxes.
        discharge = x.new_zeros(batch_size, hidden_size)
        storage_series = x.new_zeros(batch_size, hidden_size)

        loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        loss_constrained = x.new_zeros(batch_size, hidden_size)

        bypass = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size)
        gate_output = x.new_zeros(batch_size, hidden_size)

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        gate_loss_constrained = x.new_zeros(batch_size, hidden_size)

        gate_remember = x.new_zeros(batch_size, hidden_size)

        # PET scaling from the original formulation.
        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition terms.
        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = exp_output + exp_loss + exp_remember

        output_scale = exp_output / gate_sum
        loss_scale = exp_loss / gate_sum

        output_bias = self.bias_b0_yom.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            # Storage-dependent output gate.
            output_state = torch.addmm(
                output_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )
            output_fraction = output_scale * torch.sigmoid(output_state)

            # PET-dependent unconstrained loss gate.
            loss_state = torch.addmm(
                loss_bias,
                (pet - pet_mean) / pet_std,
                self.weight_b2_ylm,
            )
            loss_fraction = loss_scale * torch.sigmoid(loss_state)

            # PET constraint: actual loss cannot exceed PET.
            if storage.item() > 0:
                loss_fraction_constrained = (
                    loss_fraction
                    - self.relu_l(loss_fraction - pet / storage)
                )
            else:
                loss_fraction_constrained = loss_fraction

            # Remember gate uses the constrained loss gate.
            remember_fraction = 1.0 - output_fraction - loss_fraction_constrained

            # Save pre-update storage and fluxes.
            storage_series[b, :] = storage[0]
            discharge[b, :] = (output_fraction * storage)[0]
            loss_unconstrained[b, :] = (loss_fraction * storage)[0]
            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]

            # Save gate diagnostics.
            gate_output[b, :] = output_fraction[0]
            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = loss_fraction_constrained[0]
            gate_remember[b, :] = remember_fraction[0]

            # Input bypass = 0: all precipitation enters storage.
            storage = remember_fraction * storage + precipitation

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
        )


class MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate(nn.Module):
    """MA2 single-node MCP with two storage-dependent output gates
    and a PET-constrained loss flux.

    MA2 extends MA1 by adding a second output flow path. Both output
    gates vary with normalized storage, while the unconstrained loss
    gate varies with normalized PET.

    The constrained loss gate satisfies:

        gLc = min(gL, PET / storage)

    for positive storage.

    The remember gate is defined by mass conservation:

        gR = 1 - gO1 - gO2 - gLc

    The input gate/bypass is fixed to zero, so all precipitation
    enters storage.

    Parameter names and shapes are preserved for compatibility
    with the original MA2 checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MA Zoo interface.
        del input_size, gate_dim, spinLen, traintimeLen, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MA2 is a single-node MCP and requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names and shapes.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom_gw = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.relu_l = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yom_gw.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_yom_gw.uniform_(0.0, 1.0)
            self.weight_b1_yom_gw.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, c_mean, c_std):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("MA2 expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("MA2 requires precipitation and PET.")

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        # Pre-update states and fluxes.
        output_1 = x.new_zeros(batch_size, hidden_size)
        output_2 = x.new_zeros(batch_size, hidden_size)
        storage_series = x.new_zeros(batch_size, hidden_size)
        loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        loss_constrained = x.new_zeros(batch_size, hidden_size)
        bypass = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size)
        gate_output_1 = x.new_zeros(batch_size, hidden_size)
        gate_output_2 = x.new_zeros(batch_size, hidden_size)
        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        gate_loss_constrained = x.new_zeros(batch_size, hidden_size)
        gate_remember = x.new_zeros(batch_size, hidden_size)

        # PET scaling from the original formulation.
        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition terms.
        exp_output_1 = torch.exp(self.weight_r_yom)
        exp_output_2 = torch.exp(self.weight_r_yom_gw)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = exp_output_1 + exp_output_2 + exp_loss + exp_remember

        output_1_scale = exp_output_1 / gate_sum
        output_2_scale = exp_output_2 / gate_sum
        loss_scale = exp_loss / gate_sum

        output_1_bias = self.bias_b0_yom.unsqueeze(0)
        output_2_bias = self.bias_b0_yom_gw.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            # Two storage-dependent output gates.
            output_1_state = torch.addmm(
                output_1_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )
            output_1_fraction = output_1_scale * torch.sigmoid(output_1_state)

            output_2_state = torch.addmm(
                output_2_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom_gw,
            )
            output_2_fraction = output_2_scale * torch.sigmoid(output_2_state)

            # PET-dependent unconstrained loss gate.
            loss_state = torch.addmm(
                loss_bias,
                (pet - pet_mean) / pet_std,
                self.weight_b2_ylm,
            )
            loss_fraction = loss_scale * torch.sigmoid(loss_state)

            # PET constraint: actual loss cannot exceed PET.
            if storage.item() > 0:
                loss_fraction_constrained = (
                    loss_fraction
                    - self.relu_l(loss_fraction - pet / storage)
                )
            else:
                loss_fraction_constrained = loss_fraction

            remember_fraction = (
                1.0
                - output_1_fraction
                - output_2_fraction
                - loss_fraction_constrained
            )

            # Save pre-update storage and fluxes.
            storage_series[b, :] = storage[0]
            output_1[b, :] = (output_1_fraction * storage)[0]
            output_2[b, :] = (output_2_fraction * storage)[0]
            loss_unconstrained[b, :] = (loss_fraction * storage)[0]
            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]

            # Save gate diagnostics.
            gate_output_1[b, :] = output_1_fraction[0]
            gate_output_2[b, :] = output_2_fraction[0]
            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = loss_fraction_constrained[0]
            gate_remember[b, :] = remember_fraction[0]

            # Input bypass = 0: all precipitation enters storage.
            storage = remember_fraction * storage + precipitation

        return (
            output_1,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            output_2,
            gate_input,
            gate_output_1,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
            gate_output_2,
        )


class MCPBRNN_SW_Variant_Routing(nn.Module):
    """Single routing MCP node used in MA3.

    The node receives the output flux from the upstream soil-moisture
    MCP node and routes it through a second storage state.

    The output gate varies with normalized routing storage:

        gO = scale * sigmoid(...)

    and the remember gate is

        gR = 1 - gO

    so the routing node conserves mass and has no loss gate.

    Parameter names and shapes are preserved for compatibility
    with the original MA3 checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MA Zoo interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MA3 routing node requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names and shapes.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, c_mean, c_std):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if x.ndim != 2:
            raise ValueError("MA3 routing input must have shape [batch, hidden_size].")

        batch_size, n_features = x.shape

        if n_features != self.hidden_size:
            raise ValueError(
                f"Expected {self.hidden_size} routing input feature(s), "
                f"but received {n_features}."
            )

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        # Pre-update routing state and flux.
        discharge = x.new_zeros(batch_size, hidden_size)
        storage_series = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_output = x.new_zeros(batch_size, hidden_size)
        gate_remember = x.new_zeros(batch_size, hidden_size)

        exp_output = torch.exp(self.weight_r_yom)
        exp_remember = torch.exp(self.weight_r_yfm)
        output_scale = exp_output / (exp_output + exp_remember)

        output_bias = self.bias_b0_yom.unsqueeze(0)

        for b in range(time_lag, batch_size):
            inflow = x[b, :].reshape(1, hidden_size)

            output_state = torch.addmm(
                output_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )
            output_fraction = output_scale * torch.sigmoid(output_state)
            remember_fraction = 1.0 - output_fraction

            # Save pre-update state and flux.
            storage_series[b, :] = storage[0]
            discharge[b, :] = (output_fraction * storage)[0]

            # Save gate diagnostics.
            gate_output[b, :] = output_fraction[0]
            gate_remember[b, :] = remember_fraction[0]

            # Route upstream inflow through the storage node.
            storage = remember_fraction * storage + inflow

        return discharge, storage_series, gate_output, gate_remember


class MCPBRNN_GWVariant_Routing(nn.Module):
    """Groundwater routing MCP node used in MA4.

    The node receives recharge from the upstream soil-moisture node
    and routes it through a groundwater storage state.

    The output gate varies with normalized groundwater storage, and
    the remember gate is

        gR = 1 - gO

    so the groundwater node conserves mass and has no loss gate.

    A non-zero initial groundwater storage can be supplied through
    ``initial_storage``.

    Parameter names and shapes are preserved for compatibility
    with the original MA4 checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MA Zoo interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError(
                "MA4 groundwater routing node requires hidden_size == 1."
            )

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names and shapes.
        self.weight_r_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_b0_yom = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

    def forward(
        self,
        x,
        epoch,
        time_lag,
        c_mean,
        c_std,
        initial_storage,
    ):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if x.ndim != 2:
            raise ValueError(
                "MA4 groundwater routing input must have shape "
                "[batch, hidden_size]."
            )

        batch_size, n_features = x.shape

        if n_features != self.hidden_size:
            raise ValueError(
                f"Expected {self.hidden_size} groundwater input feature(s), "
                f"but received {n_features}."
            )

        hidden_size = self.hidden_size

        storage = x.new_full(
            (1, hidden_size),
            float(initial_storage),
        )

        # Pre-update groundwater state and outflow.
        discharge = x.new_zeros(batch_size, hidden_size)
        storage_series = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_output = x.new_zeros(batch_size, hidden_size)
        gate_remember = x.new_zeros(batch_size, hidden_size)

        exp_output = torch.exp(self.weight_r_yom)
        exp_remember = torch.exp(self.weight_r_yfm)

        output_scale = exp_output / (
            exp_output + exp_remember
        )

        output_bias = self.bias_b0_yom.unsqueeze(0)

        for b in range(time_lag, batch_size):
            recharge = x[b, :].reshape(1, hidden_size)

            output_state = torch.addmm(
                output_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )

            output_fraction = (
                output_scale * torch.sigmoid(output_state)
            )

            remember_fraction = 1.0 - output_fraction

            # Save pre-update groundwater state and outflow.
            storage_series[b, :] = storage[0]
            discharge[b, :] = (
                output_fraction * storage
            )[0]

            # Save gate diagnostics.
            gate_output[b, :] = output_fraction[0]
            gate_remember[b, :] = remember_fraction[0]

            # Route recharge through groundwater storage.
            storage = (
                remember_fraction * storage + recharge
            )

        return (
            discharge,
            storage_series,
            gate_output,
            gate_remember,
        )


class MCPBRNN_Generic_PETconstraint_Three_VariantOutputGate(nn.Module):
    """MA6 main MCP node with three storage-dependent output gates
    and a PET-constrained loss flux.

    The three output paths are:

        1. surface-flow path sent to the SW routing node,
        2. direct quick-flow path,
        3. recharge path sent to the GW routing node.

    All three output gates vary with normalized main-tank storage.
    The unconstrained loss gate varies with normalized PET.

    The constrained loss gate satisfies:

        gLc = min(gL, PET / storage)

    for positive storage.

    The remember gate is defined by mass conservation:

        gR = 1 - gO - gOfp - gOgw - gLc

    The input gate/bypass is fixed to zero, so all precipitation
    enters the main storage.

    Parameter names and shapes are preserved for compatibility
    with the original MA6 checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MA Zoo interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MA6 main node requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names and shapes.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yom_fp = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom_gw = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom_fp = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom_fp = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yom_fp.uniform_(0.0, 1.0)
            self.weight_r_yom_gw.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_yom_gw.uniform_(0.0, 1.0)
            self.weight_b1_yom_gw.uniform_(0.0, 1.0)

            self.bias_b0_yom_fp.uniform_(0.0, 1.0)
            self.weight_b1_yom_fp.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, c_mean, c_std):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("MA6 expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("MA6 requires precipitation and PET.")

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        # Pre-update states and fluxes.
        surface_input = x.new_zeros(batch_size, hidden_size)
        quickflow = x.new_zeros(batch_size, hidden_size)
        recharge = x.new_zeros(batch_size, hidden_size)

        storage_series = x.new_zeros(batch_size, hidden_size)

        loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        loss_constrained = x.new_zeros(batch_size, hidden_size)

        bypass = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size)

        gate_surface = x.new_zeros(batch_size, hidden_size)
        gate_quickflow = x.new_zeros(batch_size, hidden_size)
        gate_recharge = x.new_zeros(batch_size, hidden_size)

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        gate_loss_constrained = x.new_zeros(batch_size, hidden_size)

        gate_remember = x.new_zeros(batch_size, hidden_size)

        # PET scaling from the original formulation.
        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition terms.
        exp_surface = torch.exp(self.weight_r_yom)
        exp_quickflow = torch.exp(self.weight_r_yom_fp)
        exp_recharge = torch.exp(self.weight_r_yom_gw)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (
            exp_surface
            + exp_quickflow
            + exp_recharge
            + exp_loss
            + exp_remember
        )

        surface_scale = exp_surface / gate_sum
        quickflow_scale = exp_quickflow / gate_sum
        recharge_scale = exp_recharge / gate_sum
        loss_scale = exp_loss / gate_sum

        surface_bias = self.bias_b0_yom.unsqueeze(0)
        quickflow_bias = self.bias_b0_yom_fp.unsqueeze(0)
        recharge_bias = self.bias_b0_yom_gw.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            # Main output path sent to the SW routing node.
            surface_state = torch.addmm(
                surface_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )
            surface_fraction = (
                surface_scale * torch.sigmoid(surface_state)
            )

            # Direct quick-flow path.
            quickflow_state = torch.addmm(
                quickflow_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom_fp,
            )
            quickflow_fraction = (
                quickflow_scale * torch.sigmoid(quickflow_state)
            )

            # Recharge path sent to the GW routing node.
            recharge_state = torch.addmm(
                recharge_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom_gw,
            )
            recharge_fraction = (
                recharge_scale * torch.sigmoid(recharge_state)
            )

            # PET-dependent unconstrained loss gate.
            loss_state = torch.addmm(
                loss_bias,
                (pet - pet_mean) / pet_std,
                self.weight_b2_ylm,
            )
            loss_fraction = (
                loss_scale * torch.sigmoid(loss_state)
            )

            # PET constraint: actual loss cannot exceed PET.
            if storage.item() > 0:
                loss_fraction_constrained = (
                    loss_fraction
                    - self.relu_l(
                        loss_fraction - pet / storage
                    )
                )
            else:
                loss_fraction_constrained = loss_fraction

            remember_fraction = (
                1.0
                - surface_fraction
                - quickflow_fraction
                - recharge_fraction
                - loss_fraction_constrained
            )

            # Save pre-update storage and fluxes.
            storage_series[b, :] = storage[0]

            surface_input[b, :] = (
                surface_fraction * storage
            )[0]

            quickflow[b, :] = (
                quickflow_fraction * storage
            )[0]

            recharge[b, :] = (
                recharge_fraction * storage
            )[0]

            loss_unconstrained[b, :] = (
                loss_fraction * storage
            )[0]

            loss_constrained[b, :] = (
                loss_fraction_constrained * storage
            )[0]

            # Save gate diagnostics.
            gate_surface[b, :] = surface_fraction[0]
            gate_quickflow[b, :] = quickflow_fraction[0]
            gate_recharge[b, :] = recharge_fraction[0]

            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = (
                loss_fraction_constrained[0]
            )

            gate_remember[b, :] = remember_fraction[0]

            # Input bypass = 0: all precipitation enters storage.
            storage = (
                remember_fraction * storage
                + precipitation
            )

        return (
            surface_input,
            quickflow,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            recharge,
            gate_input,
            gate_surface,
            gate_quickflow,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
            gate_recharge,
        )


class MCPBRNN_Generic_PETconstraint_Scaling_BYPASSM0(nn.Module):
    """MA1-BP1 single-node MCP with threshold-based input bypass.

    BP1 corresponds to the original BYPASSM0 formulation.

    The bypass flux is

        B = max(P + S - exp(theta_C), 0)

    so only the non-bypassed portion of precipitation enters storage:

        S_next = gR * S + P - B

    The streamflow output is the sum of the storage-dependent MCP
    output flux and the bypass flux.

    The loss gate is PET-dependent and PET-constrained, while the
    remember gate is defined by mass conservation:

        gR = 1 - gO - gLc

    The historical parameter name ``theltaC`` is intentionally
    preserved for checkpoint compatibility.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MA Zoo interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MA1-BP1 is a single-node MCP and requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names and shapes.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Historical misspelling retained for checkpoint compatibility.
        self.theltaC = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.relu_l = nn.ReLU()
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

            self.theltaC.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, c_mean, c_std):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("MA1-BP1 expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("MA1-BP1 requires precipitation and PET.")

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        # Pre-update states and fluxes.
        discharge = x.new_zeros(batch_size, hidden_size)
        storage_series = x.new_zeros(batch_size, hidden_size)

        loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        loss_constrained = x.new_zeros(batch_size, hidden_size)

        bypass = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size)
        gate_output = x.new_zeros(batch_size, hidden_size)

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        gate_loss_constrained = x.new_zeros(batch_size, hidden_size)

        gate_remember = x.new_zeros(batch_size, hidden_size)

        # PET scaling from the original formulation.
        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition terms.
        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = exp_output + exp_loss + exp_remember

        output_scale = exp_output / gate_sum
        loss_scale = exp_loss / gate_sum

        output_bias = self.bias_b0_yom.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)

        # Original formulation used scale_factor = 1.
        bypass_threshold = torch.exp(self.theltaC)

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            # Threshold-based bypass / overflow flux.
            bypass_flux = self.relu(
                precipitation + storage - bypass_threshold
            )

            if precipitation.item() > 0:
                input_fraction = bypass_flux / precipitation
            else:
                input_fraction = x.new_zeros(1, hidden_size)

            # Storage-dependent output gate.
            output_state = torch.addmm(
                output_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )
            output_fraction = output_scale * torch.sigmoid(output_state)

            # PET-dependent unconstrained loss gate.
            loss_state = torch.addmm(
                loss_bias,
                (pet - pet_mean) / pet_std,
                self.weight_b2_ylm,
            )
            loss_fraction = loss_scale * torch.sigmoid(loss_state)

            # PET constraint: actual loss cannot exceed PET.
            if storage.item() > 0:
                loss_fraction_constrained = (
                    loss_fraction
                    - self.relu_l(loss_fraction - pet / storage)
                )
            else:
                loss_fraction_constrained = loss_fraction

            remember_fraction = (
                1.0 - output_fraction - loss_fraction_constrained
            )

            # Save pre-update storage and fluxes.
            storage_series[b, :] = storage[0]

            output_flux = output_fraction * storage

            discharge[b, :] = (
                output_flux + bypass_flux
            )[0]

            loss_unconstrained[b, :] = (
                loss_fraction * storage
            )[0]

            loss_constrained[b, :] = (
                loss_fraction_constrained * storage
            )[0]

            bypass[b, :] = bypass_flux[0]

            # Save gate diagnostics.
            gate_input[b, :] = input_fraction[0]
            gate_output[b, :] = output_fraction[0]

            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = loss_fraction_constrained[0]

            gate_remember[b, :] = remember_fraction[0]

            # Only non-bypassed precipitation enters storage.
            storage = (
                remember_fraction * storage
                + precipitation
                - bypass_flux
            )

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
        )


class MCPBRNN_Generic_PETconstraint_Scaling_BYPASSM1(nn.Module):
    """MA1-BP2 single-node MCP with a learned precipitation bypass gate.

    BP2 corresponds to the original BYPASSM1 formulation.

    The bypass fraction depends on normalized storage and precipitation:

        gB = sigmoid(
            bB
            + ((S - mean_S) / std_S) * wB
            + (P / Pmax) * wB
        )

    and the bypass flux is

        B = gB * P

    so only the non-bypassed portion of precipitation enters storage:

        S_next = gR * S + (1 - gB) * P

    Streamflow is the sum of the storage-dependent MCP output flux
    and the bypass flux.

    The loss gate is PET-dependent and PET-constrained, while the
    remember gate is defined by mass conservation:

        gR = 1 - gO - gLc

    Parameter names and shapes are preserved for compatibility
    with the original MA1-BP2 checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MA Zoo interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError(
                "MA1-BP2 is a single-node MCP and requires hidden_size == 1."
            )

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names and shapes.
        self.weight_r_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b2_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_b1_yum = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_b0_yum = nn.Parameter(
            torch.empty(hidden_size)
        )

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

            self.weight_b1_yum.uniform_(0.0, 1.0)
            self.bias_b0_yum.uniform_(0.0, 1.0)

    def forward(
        self,
        x,
        epoch,
        time_lag,
        c_mean,
        c_std,
    ):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("MA1-BP2 expects seq_length == 1.")

        if n_features < 2:
            raise ValueError(
                "MA1-BP2 requires precipitation and PET."
            )

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        # Pre-update states and fluxes.
        discharge = x.new_zeros(
            batch_size,
            hidden_size,
        )
        storage_series = x.new_zeros(
            batch_size,
            hidden_size,
        )

        loss_unconstrained = x.new_zeros(
            batch_size,
            hidden_size,
        )
        loss_constrained = x.new_zeros(
            batch_size,
            hidden_size,
        )

        bypass = x.new_zeros(
            batch_size,
            hidden_size,
        )

        # Gate diagnostics.
        gate_input = x.new_zeros(
            batch_size,
            hidden_size,
        )
        gate_output = x.new_zeros(
            batch_size,
            hidden_size,
        )
        gate_loss_unconstrained = x.new_zeros(
            batch_size,
            hidden_size,
        )
        gate_loss_constrained = x.new_zeros(
            batch_size,
            hidden_size,
        )
        gate_remember = x.new_zeros(
            batch_size,
            hidden_size,
        )

        # Fixed scaling constants from the original formulation.
        pet_mean = 2.9086
        pet_std = 1.8980
        precipitation_max = 221.5190

        # Base mass-partition terms.
        exp_output = torch.exp(
            self.weight_r_yom
        )
        exp_loss = torch.exp(
            self.weight_r_ylm
        )
        exp_remember = torch.exp(
            self.weight_r_yfm
        )

        gate_sum = (
            exp_output
            + exp_loss
            + exp_remember
        )

        output_scale = (
            exp_output / gate_sum
        )
        loss_scale = (
            exp_loss / gate_sum
        )

        output_bias = (
            self.bias_b0_yom.unsqueeze(0)
        )
        loss_bias = (
            self.bias_b0_ylm.unsqueeze(0)
        )
        bypass_bias = (
            self.bias_b0_yum.unsqueeze(0)
        )

        for b in range(
            time_lag,
            batch_size,
        ):
            precipitation = (
                x[0, b, 0].reshape(1, 1)
            )
            pet = (
                x[0, b, 1].reshape(1, 1)
            )

            # Learned precipitation bypass gate.
            bypass_storage_state = torch.addmm(
                bypass_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yum,
            )

            bypass_precip_state = torch.mm(
                precipitation / precipitation_max,
                self.weight_b1_yum,
            )

            input_fraction = torch.sigmoid(
                bypass_storage_state
                + bypass_precip_state
            )

            bypass_flux = (
                input_fraction * precipitation
            )

            # Storage-dependent output gate.
            output_state = torch.addmm(
                output_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )

            output_fraction = (
                output_scale
                * torch.sigmoid(output_state)
            )

            # PET-dependent unconstrained loss gate.
            loss_state = torch.addmm(
                loss_bias,
                (pet - pet_mean) / pet_std,
                self.weight_b2_ylm,
            )

            loss_fraction = (
                loss_scale
                * torch.sigmoid(loss_state)
            )

            # PET constraint: actual loss cannot exceed PET.
            if storage.item() > 0:
                loss_fraction_constrained = (
                    loss_fraction
                    - self.relu_l(
                        loss_fraction
                        - pet / storage
                    )
                )
            else:
                loss_fraction_constrained = (
                    loss_fraction
                )

            remember_fraction = (
                1.0
                - output_fraction
                - loss_fraction_constrained
            )

            # Save pre-update storage and fluxes.
            storage_series[b, :] = storage[0]

            output_flux = (
                output_fraction * storage
            )

            discharge[b, :] = (
                output_flux
                + bypass_flux
            )[0]

            loss_unconstrained[b, :] = (
                loss_fraction * storage
            )[0]

            loss_constrained[b, :] = (
                loss_fraction_constrained
                * storage
            )[0]

            bypass[b, :] = (
                bypass_flux[0]
            )

            # Save gate diagnostics.
            gate_input[b, :] = (
                input_fraction[0]
            )
            gate_output[b, :] = (
                output_fraction[0]
            )
            gate_loss_unconstrained[b, :] = (
                loss_fraction[0]
            )
            gate_loss_constrained[b, :] = (
                loss_fraction_constrained[0]
            )
            gate_remember[b, :] = (
                remember_fraction[0]
            )

            # Only non-bypassed precipitation enters storage.
            storage = (
                remember_fraction * storage
                + (1.0 - input_fraction)
                * precipitation
            )

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
        )


class MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate_BYPASSM0(nn.Module):
    """MA2-BP1 MCP with two storage-dependent output gates
    and threshold-based input bypass.

    BP1 corresponds to the original BYPASSM0 formulation.

    The bypass flux is

        B = max(P + S - exp(theta_C), 0)

    so only the non-bypassed portion of precipitation enters storage:

        S_next = gR * S + P - B

    The main storage has two output gates. The total system discharge is

        Q = Q1 + Q2 + B

    The loss gate is PET-dependent and PET-constrained, and the
    remember gate is defined by mass conservation:

        gR = 1 - gO1 - gO2 - gLc

    The historical parameter name ``theltaC`` is intentionally
    preserved for checkpoint compatibility.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MA Zoo interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError(
                "MA2-BP1 is a single-storage MCP and requires hidden_size == 1."
            )

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names and shapes.
        self.weight_r_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yom_gw = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom_gw = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom_gw = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b2_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        # Historical misspelling retained for checkpoint compatibility.
        self.theltaC = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.relu = nn.ReLU()
        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yom_gw.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_yom_gw.uniform_(0.0, 1.0)
            self.weight_b1_yom_gw.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

            self.theltaC.uniform_(0.0, 1.0)

    def forward(
        self,
        x,
        epoch,
        time_lag,
        c_mean,
        c_std,
    ):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("MA2-BP1 expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("MA2-BP1 requires precipitation and PET.")

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        # Pre-update states and fluxes.
        output_1 = x.new_zeros(batch_size, hidden_size)
        output_2 = x.new_zeros(batch_size, hidden_size)

        storage_series = x.new_zeros(batch_size, hidden_size)

        loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        loss_constrained = x.new_zeros(batch_size, hidden_size)

        bypass = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size)

        gate_output_1 = x.new_zeros(batch_size, hidden_size)
        gate_output_2 = x.new_zeros(batch_size, hidden_size)

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        gate_loss_constrained = x.new_zeros(batch_size, hidden_size)

        gate_remember = x.new_zeros(batch_size, hidden_size)

        # PET scaling from the original formulation.
        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition terms.
        exp_output_1 = torch.exp(self.weight_r_yom)
        exp_output_2 = torch.exp(self.weight_r_yom_gw)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (
            exp_output_1
            + exp_output_2
            + exp_loss
            + exp_remember
        )

        output_1_scale = exp_output_1 / gate_sum
        output_2_scale = exp_output_2 / gate_sum
        loss_scale = exp_loss / gate_sum

        output_1_bias = self.bias_b0_yom.unsqueeze(0)
        output_2_bias = self.bias_b0_yom_gw.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)

        bypass_threshold = torch.exp(self.theltaC)

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            # Threshold-based bypass / overflow flux.
            bypass_flux = self.relu(
                precipitation + storage - bypass_threshold
            )

            if precipitation.item() > 0:
                input_fraction = bypass_flux / precipitation
            else:
                input_fraction = x.new_zeros(1, hidden_size)

            # First storage-dependent output gate.
            output_1_state = torch.addmm(
                output_1_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )
            output_1_fraction = (
                output_1_scale * torch.sigmoid(output_1_state)
            )

            # Second storage-dependent output gate.
            output_2_state = torch.addmm(
                output_2_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom_gw,
            )
            output_2_fraction = (
                output_2_scale * torch.sigmoid(output_2_state)
            )

            # PET-dependent unconstrained loss gate.
            loss_state = torch.addmm(
                loss_bias,
                (pet - pet_mean) / pet_std,
                self.weight_b2_ylm,
            )
            loss_fraction = (
                loss_scale * torch.sigmoid(loss_state)
            )

            # PET constraint: actual loss cannot exceed PET.
            if storage.item() > 0:
                loss_fraction_constrained = (
                    loss_fraction
                    - self.relu_l(
                        loss_fraction - pet / storage
                    )
                )
            else:
                loss_fraction_constrained = loss_fraction

            remember_fraction = (
                1.0
                - output_1_fraction
                - output_2_fraction
                - loss_fraction_constrained
            )

            # Save pre-update storage and fluxes.
            storage_series[b, :] = storage[0]

            output_1[b, :] = (
                output_1_fraction * storage
            )[0]

            output_2[b, :] = (
                output_2_fraction * storage
            )[0]

            loss_unconstrained[b, :] = (
                loss_fraction * storage
            )[0]

            loss_constrained[b, :] = (
                loss_fraction_constrained * storage
            )[0]

            bypass[b, :] = bypass_flux[0]

            # Save gate diagnostics.
            gate_input[b, :] = input_fraction[0]

            gate_output_1[b, :] = output_1_fraction[0]
            gate_output_2[b, :] = output_2_fraction[0]

            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = (
                loss_fraction_constrained[0]
            )

            gate_remember[b, :] = remember_fraction[0]

            # Only non-bypassed precipitation enters storage.
            storage = (
                remember_fraction * storage
                + precipitation
                - bypass_flux
            )

        return (
            output_1,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            output_2,
            gate_input,
            gate_output_1,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
            gate_output_2,
        )


class MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate_BYPASSM1(nn.Module):
    """MA2-BP2 MCP with two storage-dependent output gates
    and a learned precipitation bypass gate.

    BP2 corresponds to the original BYPASSM1 formulation.

    The bypass fraction depends on normalized storage and precipitation:

        gB = sigmoid(
            bB
            + ((S - mean_S) / std_S) * wB
            + (P / Pmax) * wB
        )

    and the bypass flux is

        B = gB * P

    so only the non-bypassed portion of precipitation enters storage:

        S_next = gR * S + (1 - gB) * P

    The main storage has two output gates. The total system discharge is

        Q = Q1 + Q2 + B

    The loss gate is PET-dependent and PET-constrained, and the
    remember gate is defined by mass conservation:

        gR = 1 - gO1 - gO2 - gLc

    Parameter names and shapes are preserved for compatibility
    with the original MA2-BP2 checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MA Zoo interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError(
                "MA2-BP2 is a single-storage MCP and requires hidden_size == 1."
            )

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names and shapes.
        self.weight_r_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yom_gw = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom_gw = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom_gw = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b2_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_b1_yum = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_b0_yum = nn.Parameter(
            torch.empty(hidden_size)
        )

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yom_gw.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_yom_gw.uniform_(0.0, 1.0)
            self.weight_b1_yom_gw.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

            self.weight_b1_yum.uniform_(0.0, 1.0)
            self.bias_b0_yum.uniform_(0.0, 1.0)

    def forward(
        self,
        x,
        epoch,
        time_lag,
        c_mean,
        c_std,
    ):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("MA2-BP2 expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("MA2-BP2 requires precipitation and PET.")

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        # Pre-update states and fluxes.
        output_1 = x.new_zeros(batch_size, hidden_size)
        output_2 = x.new_zeros(batch_size, hidden_size)

        storage_series = x.new_zeros(batch_size, hidden_size)

        loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        loss_constrained = x.new_zeros(batch_size, hidden_size)

        bypass = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size)

        gate_output_1 = x.new_zeros(batch_size, hidden_size)
        gate_output_2 = x.new_zeros(batch_size, hidden_size)

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        gate_loss_constrained = x.new_zeros(batch_size, hidden_size)

        gate_remember = x.new_zeros(batch_size, hidden_size)

        # Fixed scaling constants from the original formulation.
        pet_mean = 2.9086
        pet_std = 1.8980
        precipitation_max = 221.5190

        # Base mass-partition terms.
        exp_output_1 = torch.exp(self.weight_r_yom)
        exp_output_2 = torch.exp(self.weight_r_yom_gw)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (
            exp_output_1
            + exp_output_2
            + exp_loss
            + exp_remember
        )

        output_1_scale = exp_output_1 / gate_sum
        output_2_scale = exp_output_2 / gate_sum
        loss_scale = exp_loss / gate_sum

        output_1_bias = self.bias_b0_yom.unsqueeze(0)
        output_2_bias = self.bias_b0_yom_gw.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)
        bypass_bias = self.bias_b0_yum.unsqueeze(0)

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            # Learned precipitation bypass gate.
            bypass_storage_state = torch.addmm(
                bypass_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yum,
            )

            bypass_precip_state = torch.mm(
                precipitation / precipitation_max,
                self.weight_b1_yum,
            )

            input_fraction = torch.sigmoid(
                bypass_storage_state
                + bypass_precip_state
            )

            bypass_flux = (
                input_fraction * precipitation
            )

            # First storage-dependent output gate.
            output_1_state = torch.addmm(
                output_1_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )

            output_1_fraction = (
                output_1_scale
                * torch.sigmoid(output_1_state)
            )

            # Second storage-dependent output gate.
            output_2_state = torch.addmm(
                output_2_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom_gw,
            )

            output_2_fraction = (
                output_2_scale
                * torch.sigmoid(output_2_state)
            )

            # PET-dependent unconstrained loss gate.
            loss_state = torch.addmm(
                loss_bias,
                (pet - pet_mean) / pet_std,
                self.weight_b2_ylm,
            )

            loss_fraction = (
                loss_scale
                * torch.sigmoid(loss_state)
            )

            # PET constraint: actual loss cannot exceed PET.
            if storage.item() > 0:
                loss_fraction_constrained = (
                    loss_fraction
                    - self.relu_l(
                        loss_fraction
                        - pet / storage
                    )
                )
            else:
                loss_fraction_constrained = (
                    loss_fraction
                )

            remember_fraction = (
                1.0
                - output_1_fraction
                - output_2_fraction
                - loss_fraction_constrained
            )

            # Save pre-update storage and fluxes.
            storage_series[b, :] = storage[0]

            output_1[b, :] = (
                output_1_fraction * storage
            )[0]

            output_2[b, :] = (
                output_2_fraction * storage
            )[0]

            loss_unconstrained[b, :] = (
                loss_fraction * storage
            )[0]

            loss_constrained[b, :] = (
                loss_fraction_constrained
                * storage
            )[0]

            bypass[b, :] = (
                bypass_flux[0]
            )

            # Save gate diagnostics.
            gate_input[b, :] = (
                input_fraction[0]
            )

            gate_output_1[b, :] = (
                output_1_fraction[0]
            )

            gate_output_2[b, :] = (
                output_2_fraction[0]
            )

            gate_loss_unconstrained[b, :] = (
                loss_fraction[0]
            )

            gate_loss_constrained[b, :] = (
                loss_fraction_constrained[0]
            )

            gate_remember[b, :] = (
                remember_fraction[0]
            )

            # Only non-bypassed precipitation enters storage.
            storage = (
                remember_fraction * storage
                + (1.0 - input_fraction)
                * precipitation
            )

        return (
            output_1,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            output_2,
            gate_input,
            gate_output_1,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
            gate_output_2,
        )


class MCPBRNN_Generic_PETconstraint_Three_VariantOutputGate_BYPASSM0(nn.Module):
    """MA6-BP1 main MCP: three output gates + threshold-based bypass.

    BP1 corresponds to the historical BYPASSM0 formulation.

    Main-tank fluxes:
      - surface output
      - quick-flow output
      - recharge / groundwater output
      - PET-constrained loss
      - threshold bypass

    The bypass flux is

        B = max(P + S - exp(theta_C), 0)

    and only the non-bypassed precipitation enters storage:

        S_next = gR * S + P - B

    with

        gR = 1 - gO_surface - gO_quick - gO_gw - gL_constrained

    All diagnostic fluxes/states returned below are PRE-UPDATE values,
    matching the historical implementation.

    The historical misspelling ``theltaC`` is intentionally preserved
    for checkpoint compatibility.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MA Zoo constructor interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError(
                "MA6-BP1 is a single-storage MCP and requires hidden_size == 1."
            )

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names and shapes.
        self.weight_r_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yom_fp = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yom_gw = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom_gw = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom_gw = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom_fp = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom_fp = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b2_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.theltaC = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.relu_l = nn.ReLU()
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yom_fp.uniform_(0.0, 1.0)
            self.weight_r_yom_gw.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_yom_gw.uniform_(0.0, 1.0)
            self.weight_b1_yom_gw.uniform_(0.0, 1.0)

            self.bias_b0_yom_fp.uniform_(0.0, 1.0)
            self.weight_b1_yom_fp.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

            self.theltaC.uniform_(0.0, 1.0)

    def forward(
        self,
        x,
        epoch,
        time_lag,
        c_mean,
        c_std,
    ):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("MA6-BP1 expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("MA6-BP1 requires precipitation and PET.")

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        # PRE-UPDATE states and fluxes.
        surface_output = x.new_zeros(batch_size, hidden_size)
        quickflow = x.new_zeros(batch_size, hidden_size)
        recharge = x.new_zeros(batch_size, hidden_size)

        storage_series = x.new_zeros(batch_size, hidden_size)

        loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        loss_constrained = x.new_zeros(batch_size, hidden_size)

        bypass = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size)
        gate_surface = x.new_zeros(batch_size, hidden_size)
        gate_quickflow = x.new_zeros(batch_size, hidden_size)
        gate_recharge = x.new_zeros(batch_size, hidden_size)

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        gate_loss_constrained = x.new_zeros(batch_size, hidden_size)
        gate_remember = x.new_zeros(batch_size, hidden_size)

        # PET scaling from the original formulation.
        pet_mean = 2.9086
        pet_std = 1.8980

        # Shared base partition denominator.
        exp_surface = torch.exp(self.weight_r_yom)
        exp_quick = torch.exp(self.weight_r_yom_fp)
        exp_recharge = torch.exp(self.weight_r_yom_gw)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (
            exp_surface
            + exp_recharge
            + exp_loss
            + exp_remember
            + exp_quick
        )

        surface_scale = exp_surface / gate_sum
        quick_scale = exp_quick / gate_sum
        recharge_scale = exp_recharge / gate_sum
        loss_scale = exp_loss / gate_sum

        surface_bias = self.bias_b0_yom.unsqueeze(0)
        recharge_bias = self.bias_b0_yom_gw.unsqueeze(0)
        quick_bias = self.bias_b0_yom_fp.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)

        bypass_threshold = torch.exp(self.theltaC)

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            # Threshold / overflow bypass.
            bypass_flux = self.relu(
                precipitation + storage - bypass_threshold
            )

            if precipitation.item() > 0:
                input_fraction = bypass_flux / precipitation
            else:
                input_fraction = x.new_zeros(1, hidden_size)

            # Surface output gate.
            surface_state = torch.addmm(
                surface_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )
            surface_fraction = (
                surface_scale * torch.sigmoid(surface_state)
            )

            # Recharge / groundwater output gate.
            recharge_state = torch.addmm(
                recharge_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom_gw,
            )
            recharge_fraction = (
                recharge_scale * torch.sigmoid(recharge_state)
            )

            # Direct quick-flow output gate.
            quick_state = torch.addmm(
                quick_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom_fp,
            )
            quick_fraction = (
                quick_scale * torch.sigmoid(quick_state)
            )

            # PET-dependent unconstrained loss gate.
            loss_state = torch.addmm(
                loss_bias,
                (pet - pet_mean) / pet_std,
                self.weight_b2_ylm,
            )
            loss_fraction = (
                loss_scale * torch.sigmoid(loss_state)
            )

            # PET constraint: actual loss cannot exceed PET.
            if storage.item() > 0:
                loss_fraction_constrained = (
                    loss_fraction
                    - self.relu_l(
                        loss_fraction - pet / storage
                    )
                )
            else:
                loss_fraction_constrained = loss_fraction

            remember_fraction = (
                1.0
                - surface_fraction
                - quick_fraction
                - recharge_fraction
                - loss_fraction_constrained
            )

            # Save PRE-UPDATE state and fluxes.
            storage_series[b, :] = storage[0]

            surface_output[b, :] = (
                surface_fraction * storage
            )[0]

            quickflow[b, :] = (
                quick_fraction * storage
            )[0]

            recharge[b, :] = (
                recharge_fraction * storage
            )[0]

            loss_unconstrained[b, :] = (
                loss_fraction * storage
            )[0]

            loss_constrained[b, :] = (
                loss_fraction_constrained * storage
            )[0]

            bypass[b, :] = bypass_flux[0]

            # Save gates.
            gate_input[b, :] = input_fraction[0]
            gate_surface[b, :] = surface_fraction[0]
            gate_quickflow[b, :] = quick_fraction[0]
            gate_recharge[b, :] = recharge_fraction[0]

            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = (
                loss_fraction_constrained[0]
            )
            gate_remember[b, :] = remember_fraction[0]

            # Only non-bypassed precipitation enters storage.
            storage = (
                remember_fraction * storage
                + precipitation
                - bypass_flux
            )

        return (
            surface_output,
            quickflow,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            recharge,
            gate_input,
            gate_surface,
            gate_quickflow,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
            gate_recharge,
        )


class MCPBRNN_Generic_PETconstraint_Three_VariantOutputGate_BYPASSM1(nn.Module):
    """MA6-BP2 main MCP: three output gates + learned input bypass.

    BP2 corresponds to the historical BYPASSM1 formulation.

    The learned bypass gate is

        gB = sigmoid(
            bB
            + ((S - mean_S) / std_S) * wB
            + (P / Pmax) * wB
        )

    using the same historical ``weight_b1_yum`` for both the
    storage and precipitation terms.

    The bypass flux is

        B = gB * P

    and storage updates as

        S_next = gR * S + (1 - gB) * P

    with

        gR = 1 - gO_surface - gO_quick - gO_gw - gL_constrained

    All returned state/flux diagnostics are PRE-UPDATE values,
    matching the historical implementation.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MA Zoo constructor interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError(
                "MA6-BP2 is a single-storage MCP and requires hidden_size == 1."
            )

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names and shapes.
        self.weight_r_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yom_fp = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yom_gw = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom_gw = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom_gw = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom_fp = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom_fp = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b2_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_b1_yum = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_b0_yum = nn.Parameter(
            torch.empty(hidden_size)
        )

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yom_fp.uniform_(0.0, 1.0)
            self.weight_r_yom_gw.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_yom_gw.uniform_(0.0, 1.0)
            self.weight_b1_yom_gw.uniform_(0.0, 1.0)

            self.bias_b0_yom_fp.uniform_(0.0, 1.0)
            self.weight_b1_yom_fp.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

            self.weight_b1_yum.uniform_(0.0, 1.0)
            self.bias_b0_yum.uniform_(0.0, 1.0)

    def forward(
        self,
        x,
        epoch,
        time_lag,
        c_mean,
        c_std,
    ):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("MA6-BP2 expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("MA6-BP2 requires precipitation and PET.")

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        # PRE-UPDATE states and fluxes.
        surface_output = x.new_zeros(batch_size, hidden_size)
        quickflow = x.new_zeros(batch_size, hidden_size)
        recharge = x.new_zeros(batch_size, hidden_size)

        storage_series = x.new_zeros(batch_size, hidden_size)

        loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        loss_constrained = x.new_zeros(batch_size, hidden_size)

        bypass = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size)
        gate_surface = x.new_zeros(batch_size, hidden_size)
        gate_quickflow = x.new_zeros(batch_size, hidden_size)
        gate_recharge = x.new_zeros(batch_size, hidden_size)

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        gate_loss_constrained = x.new_zeros(batch_size, hidden_size)
        gate_remember = x.new_zeros(batch_size, hidden_size)

        # Fixed scaling constants from the original formulation.
        pet_mean = 2.9086
        pet_std = 1.8980
        precipitation_max = 221.5190

        # Shared base partition denominator.
        exp_surface = torch.exp(self.weight_r_yom)
        exp_quick = torch.exp(self.weight_r_yom_fp)
        exp_recharge = torch.exp(self.weight_r_yom_gw)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (
            exp_surface
            + exp_recharge
            + exp_loss
            + exp_remember
            + exp_quick
        )

        surface_scale = exp_surface / gate_sum
        quick_scale = exp_quick / gate_sum
        recharge_scale = exp_recharge / gate_sum
        loss_scale = exp_loss / gate_sum

        bypass_bias = self.bias_b0_yum.unsqueeze(0)
        surface_bias = self.bias_b0_yom.unsqueeze(0)
        recharge_bias = self.bias_b0_yom_gw.unsqueeze(0)
        quick_bias = self.bias_b0_yom_fp.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            # Learned bypass gate.
            bypass_storage_state = torch.addmm(
                bypass_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yum,
            )

            bypass_precip_state = torch.mm(
                precipitation / precipitation_max,
                self.weight_b1_yum,
            )

            input_fraction = torch.sigmoid(
                bypass_storage_state
                + bypass_precip_state
            )

            bypass_flux = (
                input_fraction * precipitation
            )

            # Surface output gate.
            surface_state = torch.addmm(
                surface_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )

            surface_fraction = (
                surface_scale
                * torch.sigmoid(surface_state)
            )

            # Recharge / groundwater output gate.
            recharge_state = torch.addmm(
                recharge_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom_gw,
            )

            recharge_fraction = (
                recharge_scale
                * torch.sigmoid(recharge_state)
            )

            # Direct quick-flow output gate.
            quick_state = torch.addmm(
                quick_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom_fp,
            )

            quick_fraction = (
                quick_scale
                * torch.sigmoid(quick_state)
            )

            # PET-dependent unconstrained loss gate.
            loss_state = torch.addmm(
                loss_bias,
                (pet - pet_mean) / pet_std,
                self.weight_b2_ylm,
            )

            loss_fraction = (
                loss_scale
                * torch.sigmoid(loss_state)
            )

            # PET constraint.
            if storage.item() > 0:
                loss_fraction_constrained = (
                    loss_fraction
                    - self.relu_l(
                        loss_fraction
                        - pet / storage
                    )
                )
            else:
                loss_fraction_constrained = (
                    loss_fraction
                )

            remember_fraction = (
                1.0
                - surface_fraction
                - quick_fraction
                - recharge_fraction
                - loss_fraction_constrained
            )

            # Save PRE-UPDATE state and fluxes.
            storage_series[b, :] = storage[0]

            surface_output[b, :] = (
                surface_fraction * storage
            )[0]

            quickflow[b, :] = (
                quick_fraction * storage
            )[0]

            recharge[b, :] = (
                recharge_fraction * storage
            )[0]

            loss_unconstrained[b, :] = (
                loss_fraction * storage
            )[0]

            loss_constrained[b, :] = (
                loss_fraction_constrained
                * storage
            )[0]

            bypass[b, :] = bypass_flux[0]

            # Save gates.
            gate_input[b, :] = input_fraction[0]
            gate_surface[b, :] = surface_fraction[0]
            gate_quickflow[b, :] = quick_fraction[0]
            gate_recharge[b, :] = recharge_fraction[0]

            gate_loss_unconstrained[b, :] = (
                loss_fraction[0]
            )

            gate_loss_constrained[b, :] = (
                loss_fraction_constrained[0]
            )

            gate_remember[b, :] = (
                remember_fraction[0]
            )

            # Only non-bypassed precipitation enters storage.
            storage = (
                remember_fraction * storage
                + (1.0 - input_fraction)
                * precipitation
            )

        return (
            surface_output,
            quickflow,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            recharge,
            gate_input,
            gate_surface,
            gate_quickflow,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
            gate_recharge,
        )

class MCPBRNN_GWVariant_Routing_MRRegular(nn.Module):
    """Groundwater routing MCP with constrained mass-relaxation (MR).

    This is the constrained counterpart of
    ``MCPBRNN_GWVariant_Routing_MRRegular_Relaxed``.

    The MR reference storage is constrained positive through

        S_ref = exp(bias_b0_yrm) * 500

    and the signed mass-relaxation flux is

        MR = -gMR * |S - S_ref|

    so the state update is

        S_next = gR * S + recharge + MR

    State, routed discharge, and diagnostics are PRE-UPDATE values,
    matching the historical implementation.

    Historical parameter names and effective checkpoint shapes are
    preserved. Although ``bias_b0_yrm`` was initially declared as
    a matrix in the old source, ``reset_parameters`` replaced it by
    a 1-D Parameter, so its effective checkpoint shape is (hidden_size,).
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MA-Zoo constructor interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError(
                "Constrained GWMR routing requires hidden_size == 1."
            )

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.weight_r_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yvm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_s_yvm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_b0_yrm = nn.Parameter(
            torch.empty(hidden_size)
        )

        self.relu_v = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.weight_r_yvm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.weight_s_yvm.uniform_(0.0, 1.0)
            self.bias_b0_yrm.uniform_(0.0, 1.0)

    def forward(
        self,
        x,
        epoch,
        time_lag,
        c_mean,
        c_std,
        initial_storage,
    ):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if x.ndim != 2:
            raise ValueError(
                "Constrained GWMR input must have shape "
                "[batch, hidden_size]."
            )

        batch_size, n_features = x.shape

        if n_features != self.hidden_size:
            raise ValueError(
                f"Expected {self.hidden_size} groundwater input feature(s), "
                f"but received {n_features}."
            )

        hidden_size = self.hidden_size

        storage = x.new_full(
            (1, hidden_size),
            float(initial_storage),
        )

        # PRE-UPDATE diagnostics.
        discharge = x.new_zeros(batch_size, hidden_size)
        storage_series = x.new_zeros(batch_size, hidden_size)
        mass_relaxation_flux = x.new_zeros(batch_size, hidden_size)

        gate_output = x.new_zeros(batch_size, hidden_size)
        gate_remember = x.new_zeros(batch_size, hidden_size)
        gate_mass_relaxation = x.new_zeros(batch_size, hidden_size)

        exp_output = torch.exp(self.weight_r_yom)
        exp_remember = torch.exp(self.weight_r_yfm)

        output_scale = exp_output / (
            exp_output + exp_remember
        )

        output_bias = self.bias_b0_yom.unsqueeze(0)

        # Historical fixed scaling.
        mr_scale = 500.0

        # Constrained positive MR reference level.
        mr_reference_normalized = torch.exp(
            self.bias_b0_yrm
        ).unsqueeze(0)

        mr_reference_storage = (
            mr_reference_normalized * mr_scale
        )

        for b in range(time_lag, batch_size):
            recharge = x[b, :].reshape(1, hidden_size)

            # Standard groundwater output gate.
            output_state = torch.addmm(
                output_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )

            output_fraction = (
                output_scale
                * torch.sigmoid(output_state)
            )

            remember_fraction = (
                1.0 - output_fraction
            )

            # Mass-relaxation gate.
            mr_state = torch.mm(
                storage / mr_scale
                - mr_reference_normalized,
                torch.exp(self.weight_s_yvm),
            )

            mr_raw = (
                torch.sigmoid(self.weight_r_yvm)
                * torch.tanh(mr_state)
            )

            # Historical upper constraint: gMR <= gR.
            # Negative gMR remains allowed.
            mr_fraction = (
                mr_raw
                - self.relu_v(
                    mr_raw - remember_fraction
                )
            )

            mr_flux = (
                -mr_fraction
                * torch.abs(
                    storage - mr_reference_storage
                )
            )

            # Save PRE-UPDATE state / flux.
            storage_series[b, :] = storage[0]

            discharge[b, :] = (
                output_fraction * storage
            )[0]

            mass_relaxation_flux[b, :] = (
                mr_flux[0]
            )

            # Save gates.
            gate_output[b, :] = (
                output_fraction[0]
            )

            gate_remember[b, :] = (
                remember_fraction[0]
            )

            gate_mass_relaxation[b, :] = (
                mr_fraction[0]
            )

            # Route recharge and apply signed MR flux.
            storage = (
                remember_fraction * storage
                + recharge
                + mr_flux
            )

        return (
            discharge,
            storage_series,
            gate_output,
            gate_remember,
            gate_mass_relaxation,
            mass_relaxation_flux,
        )

class MCPBRNN_GWVariant_Routing_MRRegular_Relaxed(nn.Module):
    """Groundwater routing MCP with relaxed mass-relaxation (MR) flux.

    This node extends the standard groundwater-routing MCP used in MA4
    by adding a cell-state-dependent mass-relaxation term.

    The ordinary groundwater output gate is

        gO = softmax-share * sigmoid(storage-dependent term)

    with remember gate

        gR = 1 - gO

    The relaxed MR gate is

        z = (S / 500 - b_MR) @ exp(w_MR_scale)
        gMR_raw = sigmoid(w_MR_rate) * tanh(z)
        gMR = gMR_raw - ReLU(gMR_raw - gR)

    and the signed mass-relaxation flux is

        MR = -gMR * |S - 500 * b_MR|

    so the state update is

        S_next = gR * S + recharge + MR

    Positive MR adds water to storage; negative MR removes water.
    State, routed discharge, and diagnostics are saved PRE-UPDATE,
    matching the historical implementation.

    Parameter names and effective shapes are preserved for checkpoint
    compatibility with the original relaxed GWMR model.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MA-Zoo constructor interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError(
                "Relaxed GWMR routing requires hidden_size == 1."
            )

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve historical parameter names / effective checkpoint shapes.
        self.weight_r_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_r_yvm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_yom = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_s_yvm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_b0_yrm = nn.Parameter(
            torch.empty(hidden_size)
        )

        self.relu_v = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) class initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.weight_r_yvm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.weight_s_yvm.uniform_(0.0, 1.0)
            self.bias_b0_yrm.uniform_(0.0, 1.0)

    def forward(
        self,
        x,
        epoch,
        time_lag,
        c_mean,
        c_std,
        initial_storage,
    ):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if x.ndim != 2:
            raise ValueError(
                "Relaxed GWMR input must have shape "
                "[batch, hidden_size]."
            )

        batch_size, n_features = x.shape

        if n_features != self.hidden_size:
            raise ValueError(
                f"Expected {self.hidden_size} groundwater input feature(s), "
                f"but received {n_features}."
            )

        hidden_size = self.hidden_size

        storage = x.new_full(
            (1, hidden_size),
            float(initial_storage),
        )

        # PRE-UPDATE state / flow diagnostics.
        discharge = x.new_zeros(batch_size, hidden_size)
        storage_series = x.new_zeros(batch_size, hidden_size)
        mass_relaxation_flux = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_output = x.new_zeros(batch_size, hidden_size)
        gate_remember = x.new_zeros(batch_size, hidden_size)
        gate_mass_relaxation = x.new_zeros(batch_size, hidden_size)

        exp_output = torch.exp(self.weight_r_yom)
        exp_remember = torch.exp(self.weight_r_yfm)

        output_scale = exp_output / (
            exp_output + exp_remember
        )

        output_bias = self.bias_b0_yom.unsqueeze(0)
        mr_bias = self.bias_b0_yrm.unsqueeze(0)

        # Historical fixed scaling used by this GWMR formulation.
        mr_scale = 500.0

        for b in range(time_lag, batch_size):
            recharge = x[b, :].reshape(1, hidden_size)

            # Standard groundwater output gate.
            output_state = torch.addmm(
                output_bias,
                (storage - c_mean) / c_std,
                self.weight_b1_yom,
            )

            output_fraction = (
                output_scale * torch.sigmoid(output_state)
            )

            remember_fraction = 1.0 - output_fraction

            # Relaxed mass-relaxation gate.
            mr_state = torch.mm(
                storage / mr_scale - mr_bias,
                torch.exp(self.weight_s_yvm),
            )

            mr_raw = (
                torch.sigmoid(self.weight_r_yvm)
                * torch.tanh(mr_state)
            )

            # Historical upper constraint: gMR <= gR.
            # Negative gMR remains allowed in the relaxed formulation.
            mr_fraction = (
                mr_raw
                - self.relu_v(
                    mr_raw - remember_fraction
                )
            )

            mr_flux = (
                -mr_fraction
                * torch.abs(
                    storage - mr_bias * mr_scale
                )
            )

            # Save PRE-UPDATE state / flux.
            storage_series[b, :] = storage[0]

            discharge[b, :] = (
                output_fraction * storage
            )[0]

            mass_relaxation_flux[b, :] = mr_flux[0]

            # Save gate diagnostics.
            gate_output[b, :] = output_fraction[0]
            gate_remember[b, :] = remember_fraction[0]
            gate_mass_relaxation[b, :] = mr_fraction[0]

            # Route recharge and apply the signed MR flux.
            storage = (
                remember_fraction * storage
                + recharge
                + mr_flux
            )

        return (
            discharge,
            storage_series,
            gate_output,
            gate_remember,
            gate_mass_relaxation,
            mass_relaxation_flux,
        )