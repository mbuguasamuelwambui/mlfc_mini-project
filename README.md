# ⚡ MLFC Mini-Project: Mapping Power Infrastructure in Kenya

[![Tests](https://github.com/lawrennd/fynesse_template/workflows/Test/badge.svg)](https://github.com/lawrennd/fynesse_template/actions/workflows/test.yml)
[![Code Quality](https://github.com/lawrennd/fynesse_template/workflows/Code%20Quality/badge.svg)](https://github.com/lawrennd/fynesse_template/actions/workflows/code-quality.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.0+-blue.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project explores **where and how power stations can be strategically set up across Kenya** by combining **open geospatial data** (OSM, GADM, DEM, climate/weather datasets) with *machine learning*.

It is built on the *Fynesse framework*, which structures data work into three modular steps:

- **Access** → Collect & preprocess open datasets (counties, power stations, OSM rivers/roads, etc.)

- **Assess** → Validate & visualize environmental and infrastructure features

- **Address** → Model and generate insights, e.g., predicting optimal power station sites with Gaussian Processes

The ultimate goal is to provide a reproducible, data-driven foundation for **sustainable energy planning, infrastructure expansion, and environmental stewardship** in Kenya.

## 🧭 Objectives

- Map Kenya’s power stations and surrounding environmental features

- Query and clip OSM features (rivers, lakes, forests, roads, grids, etc.) at the county level

- Compute distance-based features for each power station (e.g., distance to river, distance to grid)

- Train a Gaussian Process Classifier to model probability of power station presence given environmental conditions

- Build a scalable pipeline with outputs saved to CSV/GeoJSON for reuse

## 🧱 Fynesse Framework

| Module               | Purpose                                            |
| -------------------- | -------------------------------------------------- |
| `fynesse/access.py`  | Download & preprocess open datasets                |
| `fynesse/assess.py`  | Data validation, visualization, feature extraction |
| `fynesse/address.py` | Modeling & answering key research questions        |


## Quick Start

### Prerequisites
- Python 3.9 or higher
- Poetry (install via `curl -sSL https://install.python-poetry.org | python3 -`)

### Installation
```bash
git clone https://github.com/mbuguasamuelwambui/mlfc_mini-project.git
cd mlfc_mini-project
poetry install
poetry shell

```

### Development Workflow
```bash
# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black fynesse/

# Type checking
poetry run mypy fynesse/

# Linting
poetry run flake8 fynesse/
```


The Fynesse paradigm considers three aspects to data analysis, Access, Assess, Address.

## Framework Structure

The template provides a structured approach to implementing the Fynesse framework:

```
fynesse/
├── access.py      # Data access functionality
├── assess.py      # Data assessment and quality checks
├── address.py     # Question addressing and analysis
├── config.py      # Configuration management
├── defaults.yml   # Default configuration values
└── tests/         # Comprehensive test suite
    ├── test_access.py
    ├── test_assess.py
    └── test_address.py
```
## 🌍 Data Sources

- **OpenStreetMap (OSM)** – Rivers, roads, land use, power grids, protected areas

- **GADM** – Kenya counties shapefiles

- **Kenya Power Stations Dataset** –
  

## License

MIT License - see LICENSE file for details.
