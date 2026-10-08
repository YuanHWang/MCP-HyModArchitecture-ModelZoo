# Mass-Conserving Perceptron (MCP) HyMod Architecture Model Zoo

This repository organizes the research code accompanying Wang & Gupta (2024) into a documented collection of mass-conserving-perceptron (MCP) models. The implementations were originally released on [Zenodo](https://zenodo.org/records/13840681) and have since been cleaned, refactored, and reorganized with assistance from OpenAI's ChatGPT (GPT-6.1 Sol, High reasoning mode).

The collection includes six progressively more complex MCP-based architectures that explore alternative representations of catchment storage and flow. These include a HyMod-like configuration inspired by Boyle (2000), with two flow pathways (surface and subsurface flow) and three storage elements (soil moisture, groundwater, and surface routing), along with variants incorporating input bypass and groundwater mass relaxation.

The repository also provides cleaned training and evaluation scripts, historical model checkpoints, standardized model notation, and documented execution conventions to support experiment reproduction, continued training, fine-tuning, and further model development.

Readers should consult the original paper for the formal model names, notation, and mathematical definitions associated with each MCP variant.

- Full article: [Wang & Gupta (2024)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024WR037224)
- DOI: [10.1029/2024WR037224](https://doi.org/10.1029/2024WR037224)

> Wang, Y.-H., & Gupta, H. V. (2024). Towards interpretable physical-conceptual catchment-scale hydrological modeling using the mass-conserving-perceptron. *Water Resources Research, 60*(10), e2024WR037224. https://doi.org/10.1029/2024WR037224

> Boyle, D. P. (2000). *Multicriteria Calibration of Hydrologic Models*. University of Arizona, Department of Hydrology and Water Resources, Tucson.
