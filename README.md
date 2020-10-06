# Anime-Colorization-Machine

User-interactive anime colorization

Prerequisites
------

  `pytorch`
  
  `torchvision`
  
  `numpy`
  
  `openCV2`
  
  `PyQT5`
  
  `opencv_transforms` (For training) (You can simply install this by `pip install opencv_transforms`)
  
  `matplotlib` (For training)
  
Results
-----
[Demo video](https://youtu.be/r9HG7dkug4k)

1. Grayscale to color

  The grayscale to color model still in training. Sorry!

2. Sketch to color

  ![ex_screenshot](./example/7.PNG)
  ![ex_screenshot](./example/8.png)
  ![ex_screenshot](./example/9.png)
  ![ex_screenshot](./example/10.png)
  ![ex_screenshot](./example/11.png)
    
Dataset
------

  1. Taebum Kim, "Anime Sketch Colorization Pair", https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair
  
  2. 68K illustrations crawled from web
  
    
Usage
------

  1. Download model weight from [here](https://drive.google.com/file/d/1ihPLm4mhQYYgSzUdP5-2bfLr0_mhL_ie/view?usp=sharing) and unzip on `src/model/checkpoint`
     (The grayscale to color model still in training. Sorry!)
  2. `python main.py`
  3. Select `Grayscale` or `Sketch` mode
  4. Open image and draw
  
Training details
------

| <center>Parameter</center> | <center>Value</center> |
|:--------|:--------:|
| Learning rate | 2e-4 | 
| Batch size | 2 | 
| Iteration | 150K | 
| Optimizer | Adam |
| (beta1, beta2) | (0.5, 0.999) |
| Data Augmentation | RandomResizedCrop(512)<br>RandomHorizontalFlip() |
| HW | CPU : Intel i5-8400<br>RAM : 16G<br>GPU : NVIDIA GTX1060 6G |
| Training Time | About 1.37s per iteration<br>(About 50 hours for 150K iterations) |
