# Mass-Conserving Perceptron (MCP) HyMod Architecture Model Zoo

This repository organizes the research code accompanying Wang & Gupta (2024) into a documented collection of mass-conserving-perceptron (MCP) models. The implementations were originally released on [Zenodo](https://zenodo.org/records/13840681) and have since been cleaned, refactored, and reorganized with assistance from OpenAI's ChatGPT (GPT-6.1 Sol, Max intelligence mode).

The collection includes six progressively more complex MCP-based architectures, in which nodes represent physically interpretable conceptual state variables and links represent flow pathways. It also includes variants incorporating input bypass and groundwater mass relaxation.

The conceptual reference is the three-tank, two-flow-path HyMod-like structure inspired by Boyle (2000), as illustrated in Figure 1. The three tanks represent soil-moisture storage, surface-routing storage, and groundwater storage, while the two flow pathways represent surface and subsurface flow.

![Three-tank, two-flow-path HyMod-like MCP architecture](Figure1.png)

*Figure 1. MCP representation of the three-tank, two-flow-path HyMod-like conceptual structure adopted in this study.*

**Note:** HyMod has several versions that differ in the number and arrangement of storage and routing elements. This study adopts the three-tank, two-flow-path representation shown in Figure 1 as its conceptual reference, with MCP gating functions used to describe the storage and flow processes.

The repository also provides cleaned training and evaluation scripts, historical model checkpoints, standardized model notation, and documented execution conventions to support experiment reproduction, continued training, fine-tuning, and further model development.

Readers should consult the original paper for the formal model names, notation, and mathematical definitions associated with each MCP variant.

- Full article: [Wang & Gupta (2024)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024WR037224)
- DOI: [10.1029/2024WR037224](https://doi.org/10.1029/2024WR037224)

> Wang, Y.-H., & Gupta, H. V. (2024). Towards interpretable physical-conceptual catchment-scale hydrological modeling using the mass-conserving-perceptron. *Water Resources Research, 60*(10), e2024WR037224. https://doi.org/10.1029/2024WR037224

> Boyle, D. P. (2000). *Multicriteria Calibration of Hydrologic Models*. University of Arizona, Department of Hydrology and Water Resources, Tucson.

## MCP-Based Model Architectures

Figure 2 presents the six main MCP-based model architectures, MA₁–MA₆. Each node represents a physically interpretable storage state, while the links represent water transfers and flow pathways. Compared with conventional conceptual models, these architectures replace time-constant coefficients for flow partitioning and storage release with time-variable gating functions that respond to the evolving states and relevant inputs. The parameters defining these functions remain fixed during evaluation, while the gate values vary over time.

![Six main MCP-based model architectures](Figure2.png)

*Figure 2. Detailed structures of the six main MCP-based model architectures, MA₁–MA₆.*

### Progressive Model Development

The models were trained using a progressive model development strategy. As new storage elements or flow pathways were introduced, parameters associated with components shared with earlier architectures were initialized from their previously trained values. Newly introduced components received new parameter initialization. The inherited parameters remained trainable and were further adjusted together with the new parameters.

For example, MA₅ inherited initial parameter values for its soil-moisture, surface-routing, and groundwater components from MA₂, MA₃, and MA₄, respectively.

Figure 3 provides a complementary conceptual overview, highlighting the number of storage states and streamflow pathways represented by each architecture.

![Conceptual comparison of storage states and flow pathways](Figure3.png)

*Figure 3. Simplified conceptual representations of MA₁–MA₆, showing their storage elements and streamflow pathways.*

| Model | Storage states | Flow pathways | Conceptual representation |
| :--- | :---: | :---: | :--- |
| MA₁ | 1 | 1 | Soil-moisture storage only |
| MA₂ | 1 | 2 | Soil-moisture storage with two outlet flow pathways |
| MA₃ | 2 | 1 | Soil-moisture storage followed by surface routing |
| MA₄ | 2 | 2 | Soil-moisture and groundwater storage |
| MA₅ | 3 | 2 | Soil-moisture, surface-routing, and groundwater storage |
| MA₆ | 3 | 3 | MA₅ with an additional direct overland-flow pathway |

## Running the Models

The scripts for MA₁–MA₆ are organized in the corresponding `MA1`–`MA6` folders. Pretrained Leaf River checkpoints are included, allowing users to evaluate the trained models directly or use their weights as starting points for continued training and fine-tuning.

For example, run the following commands from the repository root to work with MA₁.

### Continue Training or Fine-Tune MA₁

```bash
python MA1/mcpbrnn_Main_MA1_clean.py --epoch_no 100 --case_no 1 --checkpoint model_epoch27.pt
```

This command loads the supplied MA₁ checkpoint and performs 100 additional training epochs. The case identifier `1` is used to name the output folder and files. The script saves the checkpoint with the highest selection-set KGE obtained during the current run.

### Evaluate a Pretrained MA₁ Checkpoint

```bash
python MA1/evaluate_Main_MA1_clean.py --checkpoint model_epoch27.pt
```

This command evaluates the checkpoint without updating its parameters and exports simulated discharge, diagnostic time series, and performance metrics.

| Argument | Purpose |
| :--- | :--- |
| `--epoch_no` | Number of training epochs to run; used by the training script |
| `--case_no` | Case identifier used to name training outputs |
| `--checkpoint` | Model checkpoint to load for training or evaluation |

For MA₁, both scripts default to `model_epoch27.pt` if `--checkpoint` is omitted. Relative checkpoint paths are resolved from the script's folder. For other architectures, use the corresponding scripts and compatible checkpoints in their model folders.
