# [FeatLoc: Absolute Pose Regressor for Indoor 2D Sparse Features with Simplistic View Synthesizing](https://doi.org/10.1016/j.isprsjprs.2022.04.021)
This is the PyTorch implementation of our [ISPRS 2022 paper](https://doi.org/10.1016/j.isprsjprs.2022.04.021). We introduce FeatLoc, a neural network for end-to-end learning of camera localization from indoor 2D sparse features. 
Authors: [Thuan Bui Bach](https://scholar.google.co.kr/citations?user=_uvHRywAAAAJ&hl=en), [Tuan Tran Dinh](https://sites.google.com/view/tuantd), [Joo-Ho Lee](https://research-db.ritsumei.ac.jp/rithp/k03/resid/S000220)

<p align="center">
<img src="https://github.com/ais-lab/FeatLoc/blob/main/doc/fig1.svg" width="500" height="320">
<p>

## BibTex Citation 
If you find this project useful, please cite:
```
@article{BACH202250,
title = {FeatLoc: Absolute pose regressor for indoor 2D sparse features with simplistic view synthesizing},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {189},
pages = {50-62},
year = {2022},
doi = {https://doi.org/10.1016/j.isprsjprs.2022.04.021},
}
```

## About
### Network Architecture

<p align="center">
<img src="https://github.com/ais-lab/FeatLoc/blob/main/doc/fig4.svg" width="800" height="290">
<p>

### Results on [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and [12Scenes](http://graphics.stanford.edu/projects/reloc/)


 <div align="center">
 <table>
  <caption>Comparison of average median localization errors</caption>
  <thead>
    <tr>
      <th> Name </th>
      <th><a href="https://arxiv.org/abs/1704.00390">PoseNet17</a></th>
      <th><a href="https://arxiv.org/abs/1712.03342">MapNet</a></th>
      <th> FeatLoc(ours) </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7Scenes</td>
      <td> 0.24m, 8.12° </td>
      <td> 0.21m, 7.77° </td>
      <td> <strong>0.14m, 5.89°</strong></td>
    </tr>
  </tbody>
   <tbody>
    <tr>
      <td>12Scenes</td>
      <td> 0.74m, 6.48° </td>
      <td> 0.63m, 5.85° </td>
      <td><strong> 0.38m, 5.04° </strong></td>
    </tr>
  </tbody>
</table>
</div>

## Documentation
### Setup 
 * The codes are tested along with 
   * Python 3.7,
   * Pytorch 1.5.0,
   * [PointNet++ lib](https://github.com/erikwijmans/Pointnet2_PyTorch),
   * Others python packages including matplotlib, pandas, h5py, tqdm, and numpy.
 * To directly install these packages, run 
```
sudo pip install -r requirements.txt
```
 * If you are familiar with conda environments, please run 
```
conda create -f environment.yml
conda activate FeatLoc
```
 * Note that [PointNet++ lib](https://github.com/erikwijmans/Pointnet2_PyTorch) needs to install seperately.
### Data
 1. Install the [hierarchical localization toolbox](https://github.com/cvg/Hierarchical-Localization)(hloc) into the ```dataset``` folder, then change its name to ```Hierarchical_Localization``` as bellow. 
 ```
FeatLoc
├── dataset
│   ├── Generated_Data
│   ├── Hierarchical_Localization
|   ├── gendata.py
|   └── gendata_lib.py
├── model
│   ├── ...
├── ...
└── README.md
 ```
 2. Generate 3D model 
  * For [7scenes ](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)dataset, please process following this [guideline](https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/pipelines/7Scenes). 

  * For [12Scenes](http://graphics.stanford.edu/projects/reloc/) dataset, please run [dsac setup](https://github.com/vislearn/dsacstar/blob/master/datasets/setup_12scenes.py) to download the dataset, then use [hloc](https://github.com/cvg/Hierarchical-Localization) for generating 3D model for each scene. Note that you need to create a 3D model of entire train and test data per scene using [Colmap](https://colmap.github.io/) first, then use [hloc](https://github.com/cvg/Hierarchical-Localization) for only training data.
 3. Generate training and testing data. 
  * Please use the same environment with [hierarchical localization toolbox](https://github.com/cvg/Hierarchical-Localization) for this part.
```
cd dataset
python gendata.py --dataset 7scenes --scene chess --augment 1
```
<p align="center">
<img src="https://github.com/ais-lab/FeatLoc/blob/main/doc/out_gendata_chess.PNG" width="500">
<p>

### Running the code
#### Demo/Inference 
  * Please run the executable script ```eval.py``` for evaluating each scene independently . For example, we can evaluate FeatLoc++ on ```apt1_living``` scene as follows:
```
python eval.py --scene apt1_living --checkpoint results/apt1_living_featloc++au.pth.tar --version 2

Median error in translation = 0.2601 m
Median error in rotation    = 3.8867 degrees
```
<p align="center">
<img src="https://github.com/ais-lab/FeatLoc/blob/main/doc/apt1_living_featloc.svg" width="500" height="320">
<p>

  * You can download the prepared testing data and trained models of 12scenes from the [Google drive](https://drive.google.com/drive/folders/1K5CdXdSPOQv3EJbwL9FbbuaocNAc-4dF?usp=sharing) (please move the data folders and model files to ```dataset/Generated_Data``` and ```results``` folder respectively)


#### Train
  * Please run the executable script ```train.py``` to train each scene independently. For example, we can train FeatLoc++ on ```apt1_living``` scene as follows:
```
python train.py --scene apt1_living --n_epochs 200 --version 2 --augment 1
```

## Further enquiry
  If you have any problem in running the code, please feel free to contact me: thuan.aislab@gmail.com 

