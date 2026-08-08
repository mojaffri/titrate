# Real-data validation dataset

**`reizman_suzuki_case1.csv`** — Suzuki-Miyaura cross-coupling reaction data from:

> Reizman, B. J.; Wang, Y.-M.; Buchwald, S. L.; Jensen, K. F. "Suzuki–Miyaura cross-coupling optimization enabled by automated feedback." *React. Chem. Eng.* **2016**, *1*, 658–666. https://doi.org/10.1039/C6RE00153J

Retrieved from the public data mirror bundled with the [Summit](https://github.com/sustainable-processes/summit) benchmarking package (`summit/benchmarks/data/reizman_suzuki_case_1.csv`), which itself packages this dataset (among others) specifically for benchmarking optimization algorithms in chemistry — the same purpose it's used for here.

96 real automated-flow-chemistry experiments across 8 catalyst/ligand combinations, each varying residence time (`t_res`, seconds), temperature (`temperature`, °C), and catalyst loading (`catalyst_loading`, mol%), with measured yield (`yld`, %) and turnover number (`ton`).

**How it's used in Titrate:** see [`titrate/environments/reizman_suzuki_env.py`](../src/titrate/environments/reizman_suzuki_env.py) and the README's "Real-data validation" section. In short: the 37 experiments run with the most-sampled catalyst (P1-L4) are used to fit a Gaussian Process **emulator** — the same technique Summit itself uses (`ExperimentalEmulator`) to turn a fixed set of real measurements into a continuously queryable benchmark function. The emulator's own held-out accuracy is reported alongside the optimization results, so its limitations are visible rather than hidden.
