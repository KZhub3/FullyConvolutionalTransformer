## Fully Convolutional Transformer
MIT License
Copyright (c) 2022 Thanos-DB
	
## Fully Convolutional Transformer (custom fork)

**Forked from**: [Thanos-DB/FullyConvolutionalTransformer](https://github.com/Thanos-DB/FullyConvolutionalTransformer) (MIT)  
**This fork contains**:
- Customised data pre-processing
- 3D stacking of eight 384×384 PNG slices  
- Hybrid BCE (0.75) + Dice (0.25) loss


## Weights
We host our trained weights file on Google Drive. To fetch, you could download it from the link below and saved it at weights/train/:
TBA

And insert a line just before your model.predict():

```
model.load_weights("weights/train/myunet.weights_combined_dice_ce.h5")
predicted = model.predict(val_data, batch_size=batch_size)
```

## Loss history
We record our loss histories in NumPy files under `results/`. To reproduce the training curves:
```
import numpy as np
import matplotlib.pyplot as plt

history=np.load(r'history_warmup_bestmodel.npy',allow_pickle='TRUE').item()
plt.plot(history["loss"])
plt.show()

history=np.load(r'history_rlrop_bestmodel.npy',allow_pickle='TRUE').item()
plt.plot(history["loss"])
plt.plot(history["val_loss"])
```

## Citation
Please cite this paper if you want to use it in your work,

	@article{tragakis2022fully,
	title={The Fully Convolutional Transformer for Medical Image Segmentation},
	author={Tragakis, Athanasios and Kaul, Chaitanya and Murray-Smith, Roderick and Husmeier, Dirk},
	journal={arXiv preprint arXiv:2206.00566},
	year={2022}
	}

