# Anime-Colorization-Machine

User-interactive colorization of anime sketch image

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


  ![ex_screenshot](./example/7_1.PNG)
  ![ex_screenshot](./example/7_2.PNG)
  ![ex_screenshot](./example/7_3.PNG)
  ![ex_screenshot](./example/8_1.png)
  ![ex_screenshot](./example/8_2.png)
  ![ex_screenshot](./example/8_3.png)
  ![ex_screenshot](./example/10_1.png)
  ![ex_screenshot](./example/10_2.png)
  ![ex_screenshot](./example/10_3.png)
  ![ex_screenshot](./example/11_1.png)
  ![ex_screenshot](./example/11_2.png)
  ![ex_screenshot](./example/11_3.png)  
  
Dataset
------

  1. Taebum Kim, "Anime Sketch Colorization Pair", https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair
  
  2. 68K illustrations crawled from web
  
    
Usage
------

  1. Download model weight from [here](https://drive.google.com/file/d/1dQgcjruRg-aoX_7eem-b1KCIYNeAp3u9/view?usp=sharing) and unzip on `src/model/checkpoint`

  2. `python main.py`
  
  3. Load an image. The image will be automatically converted to sketch-style image.
  
  4. Draw scribbles and press 'colorize' button.
  
  5. Save the colorized image.
  
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
