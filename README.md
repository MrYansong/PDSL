# PDSL
PDSL: Propagation Dynamics Aware Framework for Source Localization
This repository contains the official implementation of PDSL, a novel framework designed to identify the source nodes of information propagation within complex networks. Unlike traditional methods that rely on static mappings, PDSL integrates propagation dynamics and continuous-time modeling to explicitly address the uncertainty caused by the stochastic nature of diffusion.

## 🚀 Overview
* **Source localization is a critical inverse inference task. PDSL addresses two primary challenges in this field:  
* **Multi-dimensional Stochasticity:** Effectively integrating network topology, propagation dynamics, and node attributes.  
* **Continuous Evolution:** Modeling the continuous state transitions of nodes rather than assuming discrete state jumps.  

### Key Contributions
* **Probabilistic Inference**: Avoids oversimplified assumptions by integrating propagation dynamics directly into generative models.
* **Continuous-Time Modeling**: Captures the latent continuous nature of social influence using Graph Neural ODEs.
* **Matching Strategy**: Employs a result-similarity matching mechanism using archived training data to provide robust initialization for the inference phase.

## 📜 Citation

If you use this code or the PDSL framework in your research, please cite our work:

```bibtex
@article{wang2026pdsl,
  title={PDSL: Propagation Dynamics Aware Framework for Source Localization},
  author={Wang, Yansong and Chai, Qisen and Lin, Longlong and Jia, Tao},
  journal={IEEE Transactions on Network Science and Engineering},
  year={2026},
  publisher={IEEE}
}
