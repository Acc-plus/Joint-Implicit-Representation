## Joint-Implicit-Representation

<!-- <p align="center"> 
<img src="/imgs/JIF.png" width="800">
</p> -->

## Installation

### Requirements
- Python 3.8 or above
- PyTorch 1.9.0 or above
- Torchvision 0.10.0 or above
  

1. Clone the repository
```
git clone https://github.com/Acc-plus/Joint-Implicit-Representation.git
git --init --recursive
cd Joint-Implicit-Representation
pip install -r requirements.txt
```

1. Install [torchmeta](https://github.com/tristandeleu/pytorch-meta).
```
cd pytorch-meta
python setup.py install
```

1. (Recommend) Install specified version of pytorch. **If you install a higher version of pytorch, the torchmeta module will not be import correctly. You might resolve the problem by commenting out [line 39 in torchmeta/dataset/utils.py](https://github.com/tristandeleu/pytorch-meta/blob/d55d89ebd47f340180267106bde3e4b723f23762/torchmeta/datasets/utils.py#L39).**
```
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
```

## Training a model from scratch

### Data preparation

Download VecFont dataset

### Training

## Inference

## Citation

Please cite the following paper if this work helps your research:

    @inproceedings{chen2023joint,
		title={Joint Implicit Neural Representation for High-fidelity and Compact Vector Fonts},
    	author={},
	    booktitle={},
	    year={2023}
	}

## Contact
If you have any questions, please contact accplusjh@gmail.com

## License

Licensed under the MIT license.

## Acknowledgement
This implementation takes [DIF-Net](https://github.com/vsitzmann/siren) as a reference. We thank the authors for their excellent work. 

