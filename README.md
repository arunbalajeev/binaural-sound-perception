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
Note: Audio track 3 (LAB-T***.WAV/LAB-T***_Tr3.WAV) and Audio track 4 (LAB-T***.WAV/LAB-T***_Tr4.WAV) of scenes from 1 to 27 are missing in the dataset. This is due to the manual error while recording the initial 27 videos of the dataset.

To extract video segments and corresponding sound time-frequency representations:
```
python extract_videosegments.py
python extract_spectrograms.py
```


## Training

a) Semantic prediction, Depth prediction and Spatial sound super-resolution
```
python train_noBG_Paralleltask_depth.py
```

b) Depth prediction and Spatial sound super-resolution
```
python scripts/train_noBG_Paralleltask_depth_noSeman.py
```

c) Semantic prediction and Spatial sound super-resolution
```
python scripts/train_noBG_Paralleltask.py
```


## Acknowledgement
This work is funded by Toyota Motor Europe via the research project TRACE (Toyota Research on Automated Cars in Europe) Zurich and was carried out at the CV Lab at ETH Zurich.
Our codes include adapted version from the external repository of NVIDIA:

- Semantic Segmentation: <https://github.com/NVIDIA/semantic-segmentation> The associated license is [here](https://github.com/NVIDIA/semantic-segmentation/blob/main/LICENSE).


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
