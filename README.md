# 🟢🍋 LimeQO: Low Rank Learning for Offline Query Optimization

### 🌟 Overview

LimeQO is a novel offline query optimizer that leverages low-rank structure in the workload matrix to optimize query performance efficiently.

<details>
  <summary>What is the workload matrix?</summary>

- Each **row** represents a **query**.
- Each **column** represents a **hint** (e.g., _disable Nested Loop Join_).
- Each **cell** represents the **runtime** of the query under the given hint (in seconds).
</details>

<details>
  <summary>Offline query optimization?</summary>

  - **What**: Effectively explore alternative query execution plans offline, in a way that minimizing total workload runtime while minimizing offline computation time.
  - **Why**: Focus on repetitive workloads and prevent regression.
  - **How**: Explores optimal plans within the available query hint space by **steering** the existing query optimizer.
  ![offline](fig/limeqo_sys_model.png)
</details>

<details>
  <summary>How LimeQO works in one sentence?</summary>

  - LimeQO uses ALS (Alternating Least Squares) algorithm to predict the runtime, then leverages the completed workload matrix to efficiently select query hints for offline exploration.
</details>

#### 🚩Features:
  - ✅ **Simple implementation**: Utilizes a straightforward matrix factorization algorithm (ALS). DO NOT need any featurization!
  - ✅ **Low overhead**: Negligible additional computation time.
  - ✅ **High performance**: Achieves **2x speedup** within just one hour.

![linear](fig/als.png)


### 🔗 Dataset

Download dataset from [dropbox](https://www.dropbox.com/scl/fo/y4e88tmcx7ywo4ou1unnh/ABN6iqV1t_ecktO51gsPKRc?rlkey=hedjnmkpak3r3gxjzx48s9etu&st=uxnr4s17&dl=0). This dataset contains the four workload we used in the experiments: ceb, job, stack, and dsb. Each zip  includes query runtimes and corresponding `EXPLAIN` plans for all queries and hints.
```bash
wget -O dataset/qo_dataset.zip "https://www.dropbox.com/scl/fo/y4e88tmcx7ywo4ou1unnh/ABN6iqV1t_ecktO51gsPKRc?rlkey=hedjnmkpak3r3gxjzx48s9etu&st=uxnr4s17&dl=1"
unzip dataset/qo_dataset.zip -d dataset/
rm dataset/qo_dataset.zip
```

### 📂 Project Structure 

```
.
├── dataset/              # Workload datasets (CEB, JOB, Stack, DSB)
├── src/                  
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Core LimeQO implementation
│   ├── utils/            # Helper functions and utilities
│   ├── strategies/       # Optimization strategies: Random, Greedy, LimeQO, LimeQO+
│   └── run_experiment.py # Run main experiments
├── draw/                 # Figures and visualizations
├── experiment/           # Experiment results
└── limeqo.ipynb          # Interactive demo notebook
```

### 🚀 Try LimeQO! 
Check out the interactive demo in [`limeqo.ipynb`](limeqo.ipynb) to see how LimeQO works! Achieve **2x speedup** in just less than 1 hour! ⬇️

![LimeQO](fig/limeqo.png)

### 🧐 Detailed instructions on reproducing the results in the [paper](https://zixy17.github.io/pdf/limeqo_sigmod25.pdf):

<details> 
  <summary> Download code and set up environment </summary>

  ```
  git clone https://github.com/zixy17/LimeQO.git
  pip install -e .
  ```
</details>

<details> 
<summary>Data and Experiment Results </summary>

- **Experiment results**  
    The directory `experiment/` already contains all results used in the paper.  
    You can directly reproduce every figure from the paper using these results, **without downloading any dataset**.
- **Datasets**  
    The directory `dataset/` contains partially packaged datasets.  
    To fully re-run experiments, you need to download the complete datasets (including all EXPLAIN plans) from Dropbox:
    ```
    wget -O dataset/qo_dataset.zip "https://www.dropbox.com/scl/fo/y4e88tmcx7ywo4ou1unnh/ABN6iqV1t_ecktO51gsPKRc?rlkey=hedjnmkpak3r3gxjzx48s9etu&st=uxnr4s17&dl=1" 
    unzip dataset/qo_dataset.zip -d dataset/ 
    rm dataset/qo_dataset.zip
    ```
    
    - The EXPLAIN plans are required only for **LimeQO+** (training requires the plan trees).
    - They are **not needed** for LimeQO or for reproducing figures.
</details>

<details> <summary>Reproduce Figures from Experiment Results </summary>

1. All experiments results are stored in `experiment/`:
	- `ceb/`, `job/`, `stack/`, `dsb/` correspond to each dataset.
	- These results were produced by `src/run_experiment.py` and are used for figure generation.
2. To reproduce the plots from the paper, run the notebooks in `draw/`.
	- Example: Figure 5 (the first figure in the experiments section) can be reproduced by running: `draw_ceb_fig1.ipynb`, `draw_job_fig1.ipynb`, `draw_stack_fig1.ipynb`, `draw_dsb_fig1.ipynb`
</details>

<details> 
<summary> Reproduce ALL experiment results </summary>

1. Download datasets as described above.
2. Run experiments:
    `python src/run_experiment.py --dataset ceb`
    - `--dataset` can be one of `ceb`, `job`, `stack`, `dsb`.
    - This will run all methods, including baselines (QOAdvisor, Random, Greedy) and ours (LimeQO, LimeQO+).
    - Each baseline is repeated 20 times; LimeQO+ is repeated 5 times (since it involves training a neural network).
    - Running each dataset may take more than 1 hour. For example, the LimeQO+ method on CEB dataset took ~2 hours per run (on CPU), so the total time is ~10 hours. 
3. Figures can then be regenerated by running the notebooks in `draw/`.
</details>

<details> <summary> Demo: limeqo.ipynb </summary>

- This demo reproduces **LimeQO results on the CEB benchmark** (one figure from the paper).
- It is designed as a quick exploratory example and does **not** reproduce baselines or LimeQO+.
- No dataset download is required for this demo.
- Running the demo may produce a `RuntimeWarning: overflow encountered in expm1`. This does not affect correctness of the results or the reproduced figure.
</details>

### ❤️ If you find our data, code, or the paper useful, please cite [our paper 📑](https://zixy17.github.io/pdf/limeqo_sigmod25.pdf):

```
@article{yi2025low,
  title={Low Rank Learning for Offline Query Optimization},
  author={Yi, Zixuan and Tian, Yao and Ives, Zachary G and Marcus, Ryan},
  journal={Proceedings of the ACM on Management of Data},
  volume={3},
  number={3},
  pages={1--26},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```