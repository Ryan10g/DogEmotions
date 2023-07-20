# DogEmotions
My project is an attempt to have the ResNet-18 AI model to be able to recognize a dog's feelings. It is able to take a picture of a dog and return an emotion from angry, happy, relaxed and sad. This AI model is useful for situations when you would like a computer to make judgements based on what a dog is feeling. For example, if you want to have a robot feed the dog everytime it gets anxious when you're gone, you can use the model to detect it.
 
https://imgur.com/a/YpBmX81

## The Algorithm

This project uses the ResNet-18 AI model, which is a convolutional neural network that is trained on more than a million images in the ImageNet database. This model is an image detecting software, so it can also be re-trained with a separate database for a different output. I've re-trained this software to recognize the emotion of a dog after looking at an image. However, because emotion is complicated, and my data isn't that accurate, the results are also not as accurate. 

The code that causes machine learning uses neural networks to make the AI recognize images better. There are three categories of data I fed the AI. Training images, validation images and test images. The training images are used to show the AI and let it guess and train which image belongs in which category. This would help the AI find patterns in the images that it can then use to identify categories. Then there is the validation phase, where the AI gets tested to see what it's missing. It then goes through the same thing over and over again. The amount of times it repeats this process is called an epoch. For this project, I ran 50 epochs, which took around 10 hours. The more epochs are ran, the more accurate the model will be, since it has more training. Lastly, we can run the testing images through the model to see how well it performs, as the AI never saw those images. Also, if we are disappointed in the model's accuracy, we can keep training it. I didn't keep going because I don't have enough time to train it more times. 

In short, this project is a model that can recognize dog's emotions through a picture. It uses the ResNet-18 image recognition software and a Jetson-Nano. A lot of the instructions I found were from GitHub. 

## Running this project

1. First of all, you need a jetson-nano that has enough disk space to run some machine learning. Also, you will need reliable internet. If you are on windows, you can use the Nano using PuTTY. If you are on Mac, I have no idea how you will access the nano lmao.
2. On the nano, you need to download jetson-inference, including all it's image recognition software. You also want to have all the directories like python and training ready to go.
3. You also need to link the Nano to VSCode, where you can view photos. Also, the directories are easily readable from the contents bar on the left in VSCode.
4. Then, you want to go to Kaggle and download the dataset for dog emotions. The link is here: https://www.kaggle.com/datasets/devzohaib/dog-emotions-prediction. Download all the pictures and import it on to the nano, whether through a usb or through wget in the linux interface.
5. All the dog photos will come in 4 folders: angry, happy, relaxed, and sad. You should delete or add enough photos so that each emotion has the same amount of data to make the model a bit more accurate.
6. Sort all the images into 3 main categories with sub-categories of angry, happy, relaxed, and sad. These three categories are train, val, and test, meaning training data, validation data, and testing data. The training data should have the most photos, then the validation data, and the testing data should be the least, as it is simply to test the complete model. Also, each sub-category within a main category should have the same amount of photos.
7. Put the directory with all the data into jetson-inference/python/training/classification/data and name it whatever you want, like dog_emotions.
8. Also, you need to make a label.txt in the same directory, and the text file needs to write anger, happy, relaxed, sad all on separate lines. (The file is also here in the repository)
9. Then, go to the docker container
10. (If you don't want to train the model, but just use resnet18.onnx, skip steps to 14) Run the command: python3 train.py --model-dir=models/cat_dog data/dog_emotions     The models/cat_dog is the ResNet18 model, and the data/dog_emotions is your dataset that you will run through AI with. Remember, you can add --epochs # with any number to specify how many epochs you want to run. I specifically ran 50 because I don't have a lot of time to run much more. The more epochs, the more accurate, but 100 is around the point where it doesn't get better with more epochs.
11. After you finish training your model, which takes a long time, make a directory in dog_emotions called test_output.
12. Make a directory in jetson-inference/python/training/classification/models called dog_emotions
13. Run this command to export the model, make sure you're in jetson-inference/python/training/classification: python3 onnx_export.py --model-dir=models/dog_emotions
14. Exit the docker container with ctrl+D
15. Make sure you're in jetson-inference/python/training/classification
16. Check that the model is in dog_emotions with ls model/dog_emotions
17. Then, set variables $NET and $DATASET with:
NET=models/dog_emotions
DATASET=data/dog_emotions
18. Use the following command to test the model on all angry pictures.
  imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/../labels.txt \
           $DATASET/test/angry $DATASET/test_output/angry_output
19. Do the same command with happy, relaxed, and sad test pictures by changing the last line to $DATASET/test/Emotion here $DATASET/test_output/Emotion here_output
20. Once you are done with all 4 emotions, you can log on to VSCode and open up the images in test_output to see the model's prediction on the dog's emotions. The percetage is its confidence level on its prediction. When predicting, the model uses Class #0, 1, 2, 3. 0 is angry, 1 is happy, 2 is relaxed, 3 is sad.
21. Lastly, you can take any image link and use wget to download it on to your nano. Then you can run the same code on the downloaded image by simply changing the place where you get the image and output in the last line.

ENJOY!

My video link: https://youtu.be/9uNzd6aGC-I
