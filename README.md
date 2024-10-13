#  FHDTIE: Fine-grained Heterogeneous Data Fusion for Tropical Cyclone Intensity Estimation

Official source code for paper 《FHDTIE: Fine-grained Heterogeneous Data Fusion for Tropical Cyclone Intensity Estimation》
### Overall Architecture of FHDTIE
![image](https://github.com/xuguangning1218/FHDTIE/blob/master/figure/model.png)

### Environment Installation
```
pip install -r requirements.txt
```  
### Data Preparation 
* Download the required GridSat dataset from NOAA official site through [here](<https://www.ncei.noaa.gov/products/gridded-geostationary-brightness-temperature> "here"), ERA5 Reanalysis data from ERA5 official sit through [here](<https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview> "here") and the tropical cyclone best track dataset from NOAA official site through [here](<https://www.ncdc.noaa.gov/ibtracs/>  "here"). 
* Or you can download the preprocessing GridSat data through [here](<https://pan.baidu.com/s/1ADa_P7atzMJ7xvmFDfclCw?pwd=j5g8#list/path=%2FTFG-Net%2Fdata> "here") and the preprocessing ERA5 Reanalysis data through [here](<https://pan.baidu.com/s/1rizZvfEieYrnh5KiHXUzuw?pwd=yfcj> "here"). Note that the ibtracs tropical cyclone best track dataset is provided in folder ***data***.


###  Source Files Description

```
-- data # data folder
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
