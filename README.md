# Project Title

**sPINNacle: Scale-Invariant Collocation Point Selection with NTK Learning Rate Adaptation**  
Author: Moritz Hauschulz (moritz.hauschulz@gmail.com)

---

## 📖 Abstract

Physics-informed neural networks (PINNs) have demonstrated remarkable performance in modeling partial differential equations. PINNacle is a recently proposed algorithm that efficiently selects the most informative points in the domain during training. However, as we show, it is not robust to domain-rescaling. In this paper, we propose sPINNacle, which mitigates this short-coming by introducing NTK-based learning rate adaptation. We conduct experiments on the two-dimensional Poisson equation and find that sPINNacle improves on PINNacle across scales in this setting. It also outperforms an alternative based on the MultiAdam optimizer.
---

## 🚀 Project Overview

This repository contains the implementation of the paper **"sPINNacle: Scale-Invariant Collocation Point Selection with NTK Learning Rate Adaptation"**. The code is based on that from the paper **"PINNACLE: PINN Adaptive ColLocation and Experimental points selection"** (https://arxiv.org/abs/2404.07662).


# ⚡ Quick Start

Follow these steps to quickly set up and run the project.

---

## 1️ Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/moritzhauschulz/sPINNacle

cd sPINNacle/
```

## 2 Install Packages (ideally in a .venv) 
```bash
pip install -r requirements.txt
```

## 2 Download Data

Some data is available in the repo, other data needs to be obtained from https://github.com/pdebench/PDEBench due to large file sizes.

## 4 Run Code

e.g.

```bash
cd pinnacle_code/

bash al_pinn_meh.sh "0" "0" "0" "--pdebench_dir /path/to/pdebench"
```
