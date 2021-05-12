# pytorch-DISN
pytorch implementation of DISN

make sure to create info.json, start by making a copy of info_example.json then modify the paths.


To get started with the dataloader

run: python3 test_dataloader.py --img_feat_onestream --category="chair"

this will run my tester which gives a nice demo

# System Requirements
* GPU: 2080Ti (Other models can consider decrease the batch size if overflow)
* Python 3.6 - 3.8
* Pytorch 1.8.1
* h5py 3.2.1
* pymcubes 0.1.2
* pymesh 1.0.2

# Data preperation
Please setup the following file structure to put the datasets in.  

Modifying ```..\preprocessing\info.json``` to your desire location for storing data.

```
"raw_dirs_v1": {
	"mesh_dir": "~/../datasets/ShapeNet/ShapeNetCore.v1/",
        "norm_mesh_dir": "~/../datasets/ShapeNet/march_cube_objs_v1/",
        "norm_mesh_dir_v2": "~/../datasets/ShapeNet/march_cube_objs/",
        "sdf_dir": "~/../datasets/ShapeNet/SDF_v1/",
        "sdf_dir_v2": "~/../datasets/ShapeNet/SDF_v2/",
        "rendered_dir": "~/../datasets/ShapeNet/ShapeNetRendering/",
        "renderedh5_dir": "~/../datasets/ShapeNet/ShapeNetRenderingh5_v1/",
        "renderedh5_dir_v2": "~/../datasets/ShapeNet/ShapeNetRenderingh5_v2/",
        "renderedh5_dir_est": "~/../datasets/ShapeNet/ShapeNetRenderingh5_v1_pred_3d/",
        "3dnnsdf_dir": "~/../datasets/ShapeNet/SDF_full/"
}
```

The following datasets are need to run the DISN to train a model.

* ShapeNet Core V1 
download the dataset following the instruction of https://www.shapenet.org/account/ (about 30GB)

* SDF ground truth
download from [here](https://drive.google.com/file/d/1cHDickPLKLz3smQNpOGXD2W5mkXcy1nq/view) then place it at your "sdf_dir" in json.

* Marching cube reconstructed ground truth models from the sdf file 
Download from [here](https://drive.google.com/drive/folders/1QGhDW335L7ra31uw5U-0V7hB-viA0JXr) then place it at your "norm_mesh_dir" in your json.

* Download and generate 2d image h5 files
Download from [here](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz) from [
run h5 file generation (about 26 GB) :
```
cd {pytorch-DISN}
nohup python -u preprocessing/create_img_h5.py &> log/create_imgh5.log &
```
