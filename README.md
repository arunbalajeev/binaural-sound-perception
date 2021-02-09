## Semantic Object Prediction and Spatial Sound Super-Resolution with Binaural Sounds
#### Arun Balajee Vasudevan, Dengxin Dai, Luc Van Gool

This repo contains the code of our [ECCV 2020 paper](https://arxiv.org/pdf/2003.04210.pdf)

## License

This software is released under a creative commons license which allows for personal and research use only. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).

## Installation 

* The code is tested with pytorch 1.5 and python 3.7.1

## Data

```
wget http://data.vision.ee.ethz.ch/arunv/binaural_perception/data.zip
unzip data.zip 
```
To extract video segments and corresponding sound time-frequency representations:
```
python extract_videosegments.py
python extract_spectrograms.py
```


## Training

a) Semantic prediction and Spatial sound super-resolution
```
python train_noBG_Paralleltask.py
```

b) Depth prediction and Spatial sound super-resolution
```
python train_noBG_Paralleltask_depth_noSeman.py
```

c) Semantic and Depth prediction and Spatial sound super-resolution
```
python train_noBG_Paralleltask_depth.py
```

## Citation

```
@inproceedings{vasudevan2020semantic,
  title={Semantic object prediction and spatial sound super-resolution with binaural sounds},
  author={Vasudevan, Arun Balajee and Dai, Dengxin and Van Gool, Luc},
  booktitle={European Conference on Computer Vision},
  pages={638--655},
  year={2020},
  organization={Springer}
}
```
