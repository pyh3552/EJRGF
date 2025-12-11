# EJRGF: Efficient Joint Registration of Multiple Point Clouds Using Fast Gaussian Filter

This is an official implementation of [EJRGF: Efficient Joint Registration of Multiple Point Clouds Using Fast Gaussian Filter](https://ieeexplore.ieee.org/document/11155196) that is accepted to IEEE Robotics and Automation Letters.

# Abstract
Joint registration plays a critical role when it comes to aligning multiple point clouds. Despite its capacity to obtain unbiased solutions, current joint registration approaches face substantial computational challenges, particularly regarding processing speed and resource consumption, which impede their practical implementation with large-scale and long-sequence data. Accordingly, we present a novel probabilistic framework called EJRGF for joint registration that achieves substantially higher efficiency, reduced resource consumption, and state-of-the-art accuracy. We assume that each data point is generated from a Gaussian Mixture Model (GMM) with isotropic components and the joint registration is then reformulated as a maximum likelihood estimation problem. To solve this optimization problem efficiently, we formally derive an innovative Expectation-Maximization (EM) algorithm accelerated by a modified fast Gaussian filter based on permutohedral lattice that estimates the transformations and the GMM parameters without compromising accuracy. Furthermore, we extend our method to address global registration tasks through feature augmentation. Comprehensive experiments demonstrate that our approach outperforms state-of-the-art methods by a large margin in terms of efficiency and accuracy, achieving at least an order of magnitude acceleration compared to JRMPC.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Sj3kG4Jxfao/0.jpg)](https://www.youtube.com/watch?v=Sj3kG4Jxfao)

# Installation
1. Clone the repository
```bash
git clone https://github.com/pyh3552/EJRGF.git
```
2. Install EJRGF
```bash
pip install .
```

# Demo
Run the demo script
```bash
cd example
python demo.py
```

# Cite
If you find this work useful for your research, please consider citing:
```
@ARTICLE{11155196,
  author={Pan, Yihan and Yi, Jianjun and Dai, Zhiyong and Zhao, Yibin and Wang, Liansheng},
  journal={IEEE Robotics and Automation Letters}, 
  title={EJRGF: Efficient Joint Registration of Multiple Point Clouds Using Fast Gaussian Filter}, 
  year={2025},
  volume={10},
  number={11},
  pages={11212-11219},
  keywords={Point cloud compression;Filtering algorithms;Accuracy;Clustering algorithms;Transforms;Lattices;Three-dimensional printing;Training;Registers;Computational modeling;Mapping;probability;statistical methods},
  doi={10.1109/LRA.2025.3608652}}

```
Thank you for your interest in our work!
# Acknowledgment
We would like to thank the authors of [JRMPC](https://team.inria.fr/perception/research/jrmpc/), [permutohedral_lattice](https://github.com/MiguelMonteiro/permutohedral_lattice) for open sourcing their codes.