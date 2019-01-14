# Automatic license plate recognition
Artifical Intelligence Fundamentals course project

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

## Authors
- Anton Borkivskyy ([@AntonBorkivskyi][1])
- Ivan Kosarevych ([@IvKosar][2])
- Marian Petruk ([@marianpetruk][3])

## Stages of project:
- [x] License plate detection and transformation ([@marianpetruk][3])
  - [x] Web scraping (data/images collection) from [РАГУlive facebook group](https://www.facebook.com/groups/rahu.live/)
  - [x] Images labeling with [Labelbox](https://www.labelbox.com/)
  - [x] CNN developing (custom CNN, similar approach as in [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937) [![DOI:10.1007/978-3-319-46484-8_292](https://zenodo.org/badge/DOI/10.1007/978-3-319-46484-8_29.svg)](https://doi.org/10.1007/978-3-319-46484-8_29))
  - [x] CNN training
  - [x] CNN testing
- [ ] Character segmentation ([@AntonBorkivskyi][1])
- [ ] Character recognition ([@IvKosar][2])


## Dependencies
  - [![Anaconda-Server Badge](https://anaconda.org/anaconda/numpy/badges/installer/conda.svg)](https://conda.anaconda.org/anaconda): `conda install numpy`
  - [![Anaconda-Server Badge](https://anaconda.org/anaconda/scipy/badges/installer/conda.svg)](https://conda.anaconda.org/anaconda): `conda install scipy`
  - [![Anaconda-Server Badge](https://anaconda.org/anaconda/matplotlib/badges/installer/conda.svg)](https://conda.anaconda.org/anaconda): `conda install matplotlib`



[1]: https://github.com/AntonBorkivskyi
[2]: https://github.com/IvKosar
[3]: https://github.com/marianpetruk