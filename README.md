# FeatLoc: Absolute Pose Regressor for Indoor 2D Sparse Features with Simplistic View Synthesizing
We introduce FeatLoc, a neural network for end-to-end learning of camera localization from 2D sparse features. 
  * Authors: Thuan Bui Bach, [Tuan Tran Dinh](https://sites.google.com/view/tuantd), [Joo-Ho-Lee](https://research-db.ritsumei.ac.jp/rithp/k03/resid/S000220)
  * to appear at ISPRS 2022
  * This repository will host the training and inference code. We will release the code in very next some weeks!

<p align="center">
<img src="https://github.com/ais-lab/FeatLoc/blob/main/doc/fig1.svg" width="500" height="320">
<p>
 
## Abstract
Precise localization using visual sensors is a fundamental requirement in many applications, including robotics, augmented reality, and autonomous systems. Traditionally, the localization problem has been tackled by leveraging 3D-geometry registering approaches. Recently, end-to-end regressor strategies using deep convolutional neural networks have achieved impressive performance, but they do not achieve the same performance as 3D structure-based methods. To some extent, this problem has been tackled by leveraging the beneficial properties of sequential images or geometric constraints. However, these approaches can only achieve a slight improvement. In this work, we address this problem for indoor scenarios, and we argue that regressing the camera pose using sparse feature descriptors could significantly improve the pose regressor performance compared with deep single-feature-vector representation. We propose a novel approach that can directly consume sparse feature descriptors to regress the camera pose effectively. More importantly, we propose a simplistic data augmentation procedure to exploit the sparse descriptors of unseen poses, leading to a remarkable enhancement in the generalization performance. Lastly, we present an extensive evaluation of our method on publicly available indoor datasets. Our FeatLoc achieves 22% and 40% improvements in translation errors on 7-Scenes and 12-Scenes relatively, compared with recent state-of-the-art absolute pose regression-based approaches. 
 
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



## BibTex Citation 
Please consider citing our work if you use our code from this repo.
 

