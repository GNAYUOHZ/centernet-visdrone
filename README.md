# centernet-visdrone
Implement of CenterNet on [visdrone2019](http://aiskyeye.com) dataset. The neck is modified to fpn with deconv.  
The entire project has less than 2000 lines of code.   

![image](https://z3.ax1x.com/2021/05/15/g65sQs.png)


## Dependencies

- Python >= 3.6
- PyTorch >= 1.6
- opencv-python
- pycocotools
- numba



## Result on validation set

| backbone |  mAP/flip | AP50/flip | AP75/flip| inference time/flip | download | 
| :------: |  :------: | :------: | :------: | :------: | :------: | 
| resnet18 |  24.70/26.26 | 49.22/51.56 | 21.33/23.10 | 0.017s/0.027s | [google drive](https://drive.google.com/file/d/191ImxjqmeKEvvNJIv-I7vQzAaT5de8kl/view?usp=sharing) |
| resnet50 |  28.13/29.46 | 53.91/55.67 | 25.36/26.75 | 0.026s/0.043s | [google drive](https://drive.google.com/file/d/1A_ohoLV6NOHpwACm7twEGxKAyZ1YMQc1/view?usp=sharing) |
| res2net50 | 29.93/31.05 | 56.46/58.01 | 27.47/28.58 | 0.035s/0.055s | [google drive](https://drive.google.com/file/d/1m-RgMCMvEYk0FftTeg20nR5LsSE714oT/view?usp=sharing) |

The inference time(pure net time) is measured on a single NVIDIA Titan V GPU.  
The resolution of image is 1280*960.  
Flip means using flip test.  


## Data

The data structure would look like:
```
data/
    visdrone/
        annotations/
        train/
        test/
        val/
```

Coco format visdrone2019 download from [google drive](https://drive.google.com/drive/folders/1FaXxOn349-YUsKa95G22etVlOf_Gj6rg?usp=sharing). 
You can also download the original dataset from http://aiskyeye.com and use the tools in src/tools to convert the format by yourself. 



## Train

```
python main.py \
--arch resnet18 \
--min_overlap 0.3 \
--gpus 0,1 \
--num_epochs 100 \
--lr_step 60,80 \
--batch_size 4 \
--lr 0.15625e-4 \
--exp_id <save_dir>
```

You can specify more parameters in src/opt.py.  
Results(weights and logs) will default save to exp/default if you dont specify --exp_id.  
Arch supports resnet18,resnet34,resnet50,resnet101,resnet152,res2net50,res2net101.  
If you scale batch_size, lr should scale too. 

## Test

```
python test.py \
--arch resnet18 \
--gpus 0 \
--load_model <path/to/weight_name.pt> \
--flip_test 
```


## Demo

``` 
python demo.py \
--arch resnet18 \
--gpus 0 \
--load_model <path/to/weight_name.pt> \
--image <path/to/your_picture.jpg>
```


## Reference

```text
https://github.com/xingyizhou/CenterNet
https://github.com/yjh0410/CenterNet-Lite
https://github.com/tjiiv-cprg/visdrone-det-toolkit-python
https://blog.csdn.net/mary_0830/article/details/103361664
```


