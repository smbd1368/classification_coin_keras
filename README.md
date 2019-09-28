# classification_coin_keras
Problem:
Detect coin class from photo
Dataset:
The number of 3060 photographs of the coins 5 , 10 , 25 , 50 and 100 of the Brazilian real 
due to the value of coins into 5 categories divided is at our disposal.
The photos are categorized by their names. 
In this case filename separator " _ " has been split in two parts. 
The first part specifies the coin class and the second part is a unique number for each coin. 
The photos are taken from different angles of the coin with 
different light on both sides of the coin.
Solution :
In this project the use of neural network convolution (CNN) And the library keras In
 Python we will classify coins. 
 The convolution network consists of two deep layers that operate 
 on each layercov, maxpooling Is done.
 
 To reduce the load, we just added the functions of each library to the project .
Due to the large number of photos and the increased speed, 
the free Google cloud service Colab Google We have used.
 The advantage of this service is that it can algorithms usingGPU Run at a much higher speed.
To use this service first upload photos to Google Cloud Storage Google Drive We then 
load the code written in Google Colab We put and run 


Tools used :
Book Houses Important used in Python :
•	Numpy
•	Random
•	Skimage
•	Keras
•	Sklearn


 
