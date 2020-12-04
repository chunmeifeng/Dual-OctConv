# Dual-Octave Convolution for Fast Parallel MR Image Reconstruction (AAAI 2021)

## Dependencies
* Python 3.7
* Tensorflow 1.14
* numpy
* h5py
* skimage
* matplotlib
* tqdm

Install dependencies as follows:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh.sh
source ~/.bashrc
conda install python=3.7
conda install tensorflow-gpu==1.14
conda install numpy
conda install scikit-learn
conda install scikit-image
conda install tqdm
conda install opencv
```

## Dataset and Prepartion
All data that we used for our experiments are released at GLOBUS(https://app.globus.org/file-manager?origin_id=15c7de28-a76b-11e9-821c-02b7a92d8e58&origin_path=%2F).
Before training, we recommend you to process data into ```.tfrecords``` to accelerate the progress.  File ```./data_preparation/data2tfrecords.py``` specifies the route of data processing.

## How to train and test on Dual-OctMRI
Unpack the dataset file to the folder you defined. Then, change the ```data_dst``` argument in ```./option.py``` to the place where datasets are located.

Enter in the folder ```/Dual-OctConv/code```

**Train**
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --n_GPU 1 --name OctComplex_B10_a0.125_cpd320_1Un3 --n_blocks 10 --n_feats 64 --lr 1e-3 --alpha 0.125 --data_dst coronal_pd_320 --epoch 50 --mask_name 1Un3_320
```

**Test**
```bash
CUDA_VISIBLE_DEVICES=0 python tester.py --n_GPU 1 --rsname OctComplex_B10_a0.125_cpd320_1Un3 --n_blocks 10 --n_feats 64 --alpha 0.125 --data_dst coronal_pd_320 --mask_name 1Un3_320 --test_only --save_gt --save_results
```
## Citation
If you find Dual-OctConv useful for your research, please consider citing the following papers:
```
@inproceedings{feng2021DualOctConv,
  title={Dual-Octave Convolution for Fast Parallel MR Image Reconstruction},
  author={Feng, Chun-Mei and Yang, Zhanyuan and Chen, Geng and Xu, Yong and Shao, Ling},
  booktitle={Proceedings of the 35th AAAI Conference on Artificial Intelligence (AAAI)},
  year={2021}
}
