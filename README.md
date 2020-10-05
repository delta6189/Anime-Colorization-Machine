# Anime-Colorization-Machine

User-interactive anime colorization

Prerequisites
------

  `pytorch`
  
  `torchvision`
  
  `numpy`
  
  `openCV2`
  
  `matplotlib` (For training)
    
Dataset
------

  1. Taebum Kim, "Anime Sketch Colorization Pair", https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair
  
  2. 68K illustrations crawled from web
  
    
Usage
------

  1. Download model weight from here() and unzip on `src/model/checkpoint`
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