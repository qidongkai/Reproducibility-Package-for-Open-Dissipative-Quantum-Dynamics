# Reproducibility-Package-for-Open-Dissipative-Quantum-Dynamics


This package implements the Lindbladian-based dissipative quantum dynamical framework. It evaluates the rule to judge whether a boundary minimum state stays stable under perturbations, quantifies system robustness over different delay configurations, and estimates statistical uncertainty for numerical outputs. A built-in consistency checker prevents common logical mistakes in the derivation of the stability rule.
﻿
## Core Capabilities
﻿
1. Simulate time evolution of open quantum systems with adaptive-step ordinary differential equation solvers
2. Perform systematic parameter sweeps to compute threshold values and test the stability rule
3. Run regularized multi-start fitting and nested model comparison
4. Calculate confidence intervals using bootstrap and delete-d jackknife resampling
5. Estimate parameter sensitivity and confidence ellipses via finite differences
6. Generate journal-standard figures and export LaTeX tables
7. Execute automatic consistency checks to match simulation outputs with the manuscript
8. Include unit tests for core modules to guard against code changes breaking the results
﻿

## Execution Workflow
﻿
Run the scripts in the listed order:
﻿
```
python scripts/01_run_evolution.py
python scripts/02_parameter_scan.py
python scripts/03_fit_and_compare.py
python scripts/04_resampling_analysis.py
python scripts/05_plot_main_figures.py
python scripts/06_plot_supp_figures.py
python scripts/07_export_latex_tables.py
python scripts/08_pre_submit_validation.py
```
﻿
To pack everything into a zip archive:
﻿
```
python build_archive.py
```
﻿
## Output Folder Layout
﻿
- `output/cache`: Checkpoint files for long simulations
- `output/csv`: Generated datasets (evolution records, sweep data, fitting statistics)
- `output/figs`: Main and supplementary figures in PDF
- `output/latex_tables`: Ready-to-use LaTeX table code
- `output/logs`: Runtime logs for debugging
- `output/reports`: Automated consistency check report
﻿
## Validation
﻿
The consistency checker cross-validates numerical outputs against the analytical statements in Chapters 2–5 of the manuscript. It targets the known counterexample where the stability logic is easy to reverse. A failed check means the simulation result conflicts with the manuscript text.

﻿
## Author
﻿
Qi Dongkai
For questions about reproducibility, please contact the author.
