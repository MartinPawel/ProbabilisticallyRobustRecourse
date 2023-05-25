
# PROBE
While several approaches (e.g., see ``Implemented Counterfactual Methods``) have been proposed to construct recourses for affected individuals, the recourses output by these methods either achieve low costs (i.e., ease-of-implementation) or robustness to small perturbations (i.e., noisy implementations of recourses), but not both due to the inherent trade-offs between the recourse costs and robustness. Our framework Probabilistically ROBust rEcourse (\texttt{PROBE}) lets users choose the probability with which a recourse could get invalidated (recourse invalidation rate) if small changes are made to the recourse i.e., the recourse is implemented somewhat noisily. To this end, we propose a novel objective function which simultaneously minimizes the gap between the achieved (resulting) and desired recourse invalidation rates, minimizes recourse costs, and also ensures that the resulting recourse achieves a positive model prediction.

### Available Datasets

- Adult Data Set: [Source](https://archive.ics.uci.edu/ml/datasets/adult)
- COMPAS: [Source](https://www.kaggle.com/danofer/compass)
- Give Me Some Credit (GMC): [Source](https://www.kaggle.com/c/GiveMeSomeCredit/data)

### Implemented Counterfactual Methods
- Actionable Recourse (AR): [Paper](https://arxiv.org/pdf/1809.06514.pdf)
- Diverse Counterfactual Explanations (DiCE): [Paper](https://arxiv.org/pdf/1905.07697.pdf)
- Growing Sphere (GS): [Paper](https://arxiv.org/pdf/1910.09398.pdf)
- Wachter: [Paper](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf)
- ROAR: [Paper](https://proceedings.neurips.cc/paper/2021/hash/8ccfb1140664a5fa63177fb6e07352f0-Abstract.html)
- ARAR: [Paper](https://proceedings.mlr.press/v162/dominguez-olmedo22a.html)

### Provided Machine Learning Models
- **ANN**: Artificial Neural Network with 2 hidden layers and ReLU activation function
- **LR**: Linear Model with no hidden layer and no activation function

### Dependence on CARLA

CARLA is a python library to benchmark counterfactual explanation and recourse models. It comes out-of-the box with commonly used datasets and various machine learning models. Documentation [here](https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/) and the corresponding NeurIPS paper [here](https://arxiv.org/pdf/2108.00783.pdf) can be found using the corresponding links.

## Installation
Using python directly or within activated virtual environment:

```sh
pip install -U pip setuptools wheel
pip install -e .
```
### Requirements

- `python3.7`
- `pip`

## Experiments
- To run our experiments, navigate to carla/recourse_invalidation_results/experiment/ and run recourseInvalidationRate.py
- To recreate some of the plots in the paper, run the following notebooks:
  - Bounds_Linear.ipynb, and
  - Bounds_ANN_Approx.ipynb.

## Credit
This project was recently accepted to ICLR 2023.
If you find our content helpful for your research, please cite:

```sh
@inproceedings{pawelczyk2023probabilistic,
      title={Probabilistically Robust Recourse: Navigating the Trade-offs between Costs and Robustness in Algorithmic Recourse},
      author={Martin Pawelczyk and Teresa Datta and Johannes van-den-Heuvel and Gjergji Kasneci and Himabindu Lakkaraju},
      booktitle={11th International Conference on Learning Representations (ICLR)},
      year={2023}
}
```
