# Assessing Model Generalization in Vicinity

 [Yuchi Liu](https://liuyvchi.github.io/), [Yifan Sun](https://yifansun-reid.github.io/), [Jingdong Wang](https://jingdongwang2017.github.io/), [Liang Zheng](https://zheng-lab.cecs.anu.edu.au)

This repository contains the implementation for the paper "Assessing Model Generalization in Vicinity".



![Figure 1](./figs/fig1.png)

---

## Preparation



#### :wrench: Installation

Before running the scripts, ensure you have Python installed along with the necessary packages. To install the required packages, execute the following command:

```bash
pip install -r requirements.txt
```

#### :wrench:  Downlaod Models

Models for 
Models for Cifar10
Model for 

#### :wrench:  Feature extractiion

```bash
bash src/test_getOutput.sh
```

##### Note

You can cahnge the python files in `test_getOutput.sh` to change the datasets"
- `test_savePredictions.py` : for Iamgenet Setup
- `test_savePredictions_cifar.py` : for cifar Setup
- `test_savePredictions_iwilds.py` : for iWilds Setup
  


#### Directory Structure Preparation

To utilize the provided scripts effectively, please organize your data according to the following directory structure:

```
├── data
│   ├── ImageNet-Val
│   ├── ImageNet-A
│   ├── ImageNet-R
│   ├── ImageNet-S
│   ├── ObjectNet
│   └── ...
└── modelOutput
    ├── imagenet_a_out_colorjitter
    │   └── tv_reesnet152.npy
    ├── imagenet_a_out_grey
    ├── imagenet_a_out_colorjitter
    └── ...
|── iwildcam_weights
└── src
```


---


## Compute Model Risks Proxies


To execute model risk estimation under different setups, we can run the following commands:

- :wrench: ImageNet setup:
```bash
python src/test_mentric.py
```

- :wrench: Cifar10 setup:
```bash
python src/test_mentric_cifar.py
```

- :wrench: Cifar10 setup:
```bash
python src/test_mentric_iWilds.py
```



---

## Citation
If you find our code helpful, please consider citing our paper:

```bibtex
@misc{liu2024hierarchical,
      title={Towards Hierarchical Multi-Agent Workflows for Zero-Shot Prompt Optimization}, 
      author={Yuchi Liu and Jaskirat Singh and Gaowen Liu and Ali Payani and Liang Zheng},
      year={2024},
      eprint={2405.20252},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

This project is open source and available under the [MIT License](LICENSE.md).

