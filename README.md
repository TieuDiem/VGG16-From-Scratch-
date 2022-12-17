## VGG16-From-Scratch-
Code-VGG16-From-Scratch with Keras

## Inside VGG16 
* This's Architecture of VGG16 (CNN)
<div align="center">
<p>
<img src="Images/Architecture-VGG16.png" width="800" height="auto"></img>
</p>
</div>   

- Block 1
``` bash
Conv2D(filters =64,kernel_size =(3,3),padding="same",activation='relu',input_shape =(224,224,3)),
Conv2D(filters =64,kernel_size= (3,3),padding="same",activation='relu'),
MaxPooling2D(pool_size=(2,2),strides=(2,2)),
```
- Block 2    
``` bash
Conv2D(filters =128,kernel_size=(3,3),padding="same",activation="relu"),
Conv2D(filters =128,kernel_size=(3,3),padding="same",activation='relu'),
MaxPooling2D(pool_size =(2,2),strides =(2,2)),
``` 
- Block 3  
``` bash
Conv2D(filters =256,kernel_size=(3,3),padding="same",activation='relu'),
Conv2D(filters =256,kernel_size=(3,3),padding="same",activation='relu'),
Conv2D(filters =256,kernel_size=(3,3),padding="same",activation='relu'),
MaxPooling2D(pool_size =(2,2),strides =(2,2)),
```  
- Block 4    
``` bash
Conv2D(filters =512,kernel_size=(3,3),padding="same",activation='relu'),
Conv2D(filters =512,kernel_size=(3,3),padding="same",activation='relu'),
Conv2D(filters =512,kernel_size=(3,3),padding="same",activation='relu'),
MaxPooling2D(pool_size =(2,2),strides =(2,2)),
``` 
- Block 5    
``` bash
Conv2D(filters =512,kernel_size=(3,3),padding="same",activation='relu'),
Conv2D(filters =512,kernel_size=(3,3),padding="same",activation='relu'),
Conv2D(filters =512,kernel_size=(3,3),padding="same",activation='relu'),
MaxPooling2D(pool_size =(2,2),strides =(2,2)),
```  
## Result
<div align="center">
<p>
<img src="Images/134206.jpg" width="600" height="auto"></img>
<p>Input Image</p>

<img src="Images/block_1.jpg" width="600" height="auto"></img>
<p>Feature Map After Block 1</p>
<p>TensorShape= [1, 112, 112, 64]<\p>

<img src="Images/block_2.jpg" width="600" height="auto"></img>
<p>Feature Map After Block 2</p>
<p>TensorShape([1, 56, 56, 128])<\p>

<img src="Images/block_3.jpg" width="600" height="auto"></img>
<p>Feature Map After Block 3</p>
<p>TensorShape([1, 28, 28, 256])<\p>

<img src="Images/block_4.jpg" width="600" height="auto"></img>
<p>Feature Map After Block 4</p>
<p>TensorShape([1, 14, 14, 512])<\p>

<img src="Images/block_5.jpg" width="600" height="auto"></img>
<p>Feature Map After Block 5</p>
<p>TensorShape([1, 7, 7, 512])<\p>
</p>
</div> 
