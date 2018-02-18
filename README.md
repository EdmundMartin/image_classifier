# image_classifier
An image classifier built using TFLearn.

## Training A Model
```python3
import glob
images = glob.glob('*.png')
c = ImageClassify(['yes', 'not'], image_size=100, learning_rate=0.001)
c.prepare_data(images)
c.train_model('my_example_model')
```
Training a model is very easy. Simply feed in the list of classes you want to train, and set the image size and learning rate parameters. The prepare data function takes in a list of images. Images are expected to be in the following format: "class.number.jpg", by passing in the images like this we can label them without the need for another labels file. Calling the train_model method will train a model and save it with the name you passed in. 
