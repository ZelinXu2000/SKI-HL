# [SIGSPATIAL '23] Spatial Knowledge-Infused Hierarchical Learning: An Application in Flood Mapping on Earth Imagery 
This repository is the official implementation of the paper: 
> **Spatial Knowledge-Infused Hierarchical Learning: An Application in Flood Mapping on Earth Imagery** <br>
> Zelin Xu<sup>1</sup>, Tingsong Xiao<sup>1</sup>, Wenchong He<sup>1</sup>, Yu Wang<sup>2</sup>, Zhe Jiang<sup>1</sup> <br>
> <sup>1</sup> *Department of Computer & Information Science & Engineering, University of Florida* <br>
> <sup>2</sup> *Department of Mechanical & Aerospace Engineering, University of Florida* <br>
> in SIGSPATIAL 2023 (**Best Paper Award**)

## Preparation
### Dependencies
- Python 3.10
- PyTorch 1.13.1
- CUDA 11.6
### Datasets
The following datasets need to be downloaded for training from [Google Drive](https://drive.google.com/drive/folders/1jytlsS9yEdcPpOgSIGqOeM1ZlxvhqnBl?usp=sharing). Please put them in the `/data` folder.

## Get Started
### Train
To train the model, you can run this command:
```
python main.py --device YOUR_DEVICE --dataset 1 
```

## Citation
Please cite our paper if you find this code useful for your work:
```
@inproceedings{xu2023spatial,
  title={Spatial knowledge-infused hierarchical learning: An application in flood mapping on earth imagery},
  author={Xu, Zelin and Xiao, Tingsong and He, Wenchong and Wang, Yu and Jiang, Zhe},
  booktitle={Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems},
  pages={1--10},
  year={2023}
}
```

## Acknowledgement
The codes are based on [U-Net](https://github.com/milesial/Pytorch-UNet) and [Geographical Hidden Markov Tree](https://github.com/spatialdatasciencegroup/HMTFIST). Thanks for the awesome open-source code!!
