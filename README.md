# Zonal Convergence


## Overview
Zonal Convergence is a Python-based toolkit designed for optimizing the I/O wiring architecture in automobile platforms. The project leverages clustering and pathfinding algorithms to achieve efficient wiring topology by:
- Clustering I/O signals based on spatial proximity and functional zones using **K-Means**.
- Computing the shortest paths between I/O nodes, extenders, and HPCs using **Dijkstra's Algorithm**.

This repository facilitates the reduction of wiring complexity and supports the design of a modular, scalable zonal architecture.

---

## Features
- **K-Means Clustering**:
  - Groups I/O points based on spatial location and signal grouping.
  - Determines optimal number and placement of I/O extenders (Zonal ECUs).

- **Dijkstra Pathfinding**:
  - Calculates the shortest path for wiring between I/O devices, extenders, and central units (e.g., HPC).
  - Generates visual and quantitative analysis of wiring length and efficiency.

---
## 🛠 Installation

### Prerequisites
- Python 3.8 or above
- Recommended IDE: VSCode                    

### Steps
```bash
# Clone the repository
git clone https://github.com/sca7r/Zonal convergence gui.git
cd Zonal convergence gui

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install required dependencies
# Navigate to requirements
cd requirements
# Run the following PIP command
pip install -r requirements.txt
```


## Directory Structure
```text
ZonalConvergence/
├── k-means/
│   ├── data
│   ├── notebooks
│   ├── Results
│   ├── src                # Source codes
│   └── README.md                  
├── dijkstra
│   ├── node_graphs
│   ├── results
│   ├── src                # Source codes
│   └── README.md
├── requirements
└──README.md              # Project documentation
```
### Maintainers - Harshawardhan Patil
### Note- The project is under development and gets updated frequently, Please pull the latest version before running.
