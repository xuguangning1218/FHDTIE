#  FHDTIE: Fine-grained Heterogeneous Data Fusion for Tropical Cyclone Intensity Estimation

Official source code for paper 《FHDTIE: Fine-grained Heterogeneous Data Fusion for Tropical Cyclone Intensity Estimation》
### Overall Architecture of FHDTIE
![image](https://github.com/xuguangning1218/FHDTIE/blob/master/figure/model.png)

### Environment Installation
```
pip install -r requirements.txt
```  
### Reproducibility 
* Download the required GridSat dataset from NOAA official site through [here](<https://www.ncei.noaa.gov/products/gridded-geostationary-brightness-temperature> "here"), ERA5 Reanalysis data from ERA5 official sit through [here](<https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview> "here") and the tropical cyclone best track dataset from NOAA official site through [here](<https://www.ncdc.noaa.gov/ibtracs/>  "here"). 
* Or you can download the preprocessing GridSat data through [here](<https://pan.baidu.com/s/1ADa_P7atzMJ7xvmFDfclCw?pwd=j5g8#list/path=%2FTFG-Net%2Fdata> "here") (The size of the GridSat is 301 x 301 in here. You need to crop it to 256 x 256 from the center at first.) and the preprocessing ERA5 Reanalysis data through [here](<https://pan.baidu.com/s/1rizZvfEieYrnh5KiHXUzuw?pwd=yfcj#list/path=%2FFHDTIE%2Fdata> "here"). Note that the ibtracs tropical cyclone best track dataset is provided in folder ***data***.
* The saved model is provided in [here](<https://pan.baidu.com/s/1rizZvfEieYrnh5KiHXUzuw?pwd=yfcj#list/path=%2FFHDTIE%2Fmodel_saver&parentPath=%2F> "here").


###  Source Files Description

```
-- data # data folder
	-- 01_process_kmeans.ipynb # code for kmean label generation
	-- 02_process_kmeans_adj.ipynb # code for adjacency matrix generation
	-- gridsat.img.min.max.npy # min and max value of the gridsat
    	-- gridsat.path.ibtr.windspeed.csv # ibtracs tropical cyclone best track dataset
	-- reanalysis.min_max.npy # min and max value of the era5
-- data_loader # data loader folder
	-- dataloader.py # dataloader in train, validate, test
-- figure # figure provider
	-- model.png # architecture of TLS-MWP model 
-- model # proposed model
	-- GCN.py # the GCN module
	-- UNet.py # the UNet module
    -- FHDTIE.py # the FHDTIE model
	-- Model.py # model handler of train, validate, test, etc.
requirements.txt # requirements package of the project
setting.config # model configure
Run.ipynb # jupyter visualized code for the model
```

###  Citation
If you think our work is helpful. Please kindly cite
```
@article{XU2024FHDTIE,
author={Xu, Guangning and Ng, Michael K. and Ye, Yunming and Zhang, Bowen},
journal={IEEE Transactions on Geoscience and Remote Sensing}, 
title={FHDTIE: Fine-grained Heterogeneous Data Fusion for Tropical Cyclone Intensity Estimation}, 
year={2024},
volume={62},
number={},
pages={1-15},
keywords={Tropical cyclones;Estimation;Satellite images;Statistical analysis;Deep learning;Data models;Feature extraction;Shape;Data integration;Clustering methods;intensity estimation;tropical cyclone intensity;typhoon;fine-grained},
doi={10.1109/TGRS.2024.3489674}}
```
