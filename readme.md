# DeepASL Project

## Objective
This work focuses on developing a Deep Learning model capable of using the optical flow produced by the gestures of a signer performing a specific ASL sign and predicting the correct gloss. The work explores models which use spatial and temporal information. There are a total of 20 glosses in the dataset.

![DeepASL](https://github.com/CognitionTree/Deep-ASL-Translator/blob/master/Python-Implementation/documents/images/ASL.png)

# For More Information
[DeepASL Paper Report](https://github.com/CognitionTree/Deep-ASL-Translator/blob/master/Python-Implementation/documents/deepasl-classifying-asl-9.pdf)

## Require Libraries
1. cv2: `sudo apt-get install python-opencv`

2. numpy: `sudo pip install numpy`

3. tensorflow: `pip install --upgrade tensorflow`

4. scipy: `sudo pip install numpy scipy`

5. scikit-learn: `sudo pip install scikit-learn`

6. pillow: `sudo pip install pillow`

7. h5py: `sudo pip install h5py`

8. keras: `sudo pip install keras`

9. matplotlib: `sudo pip install matplotlib`

## How To Run The Code
Execute the following command: `python experiments.py agent` where agent can be any of the these 3 values {random, bias, conv}

### Agents
1. Random: It chooses randomly any gloss.

2. Bias: It chooses the gloss with more repetitions.

3. Convolutional Model: It is aconvolutional network trained using supervised learning. The input to the network is the cumulative optical flow of the video of the person performing a sign, the output is the gloss. In order to train the model categorical crossentropy loss was used and the weights were optimized using Adam optimizer.

![Convolutional Model](https://github.com/CognitionTree/Deep-ASL-Translator/blob/master/Python-Implementation/model_images/Convolutional%20Model.png)

4. LSTM Model: This model instead of a cumulative optical flow of the entire video, uses a the optical flow of each pair of frames and combine them using an LSTM in order to learn temporal information. This agent can not be currently tested due to an issue uploading the data. It will be available soon.

### Output
1. The final total loss.

2. The top 4 accuracy.

3. A confussion matrix.

5. It saves in the directory results some visualisation of the training including accuracy per epoch, loss per epoch, a confusion matrix image. It also saves the trained model and its weights.

## Some Results Of The Convolutional Model
Rank1 Accuraccy: 55.4%
Rank2 Accuraccy: 66.2%
Rank3 Accuraccy: 72.2%
Rank4 Accuraccy: 77.4%

Confusion Matrix

![Confusion Matrix](https://github.com/CognitionTree/Deep-ASL-Translator/blob/master/Python-Implementation/results/Convolutional%20Model%20Top%204%20Confusion%20Matrix.png)

Top 4 Accuracy Per Epoch

![Top 4 Accuracy Per Epoch](https://github.com/CognitionTree/Deep-ASL-Translator/blob/master/Python-Implementation/results/Convolutional%20Model%20Top%204_accuracy_historytraining_all.png)

Validation and Training Loss

![Validation and Training Loss](https://github.com/CognitionTree/Deep-ASL-Translator/blob/master/Python-Implementation/results/Convolutional%20Model%20Top%204_loss_history.png)

TSNE applied to the feature vectors created by the trained convolutional model for each video of the signer. This shows in 2d how the model learns create similar representations (small eucledean distance) for the same and similar classes and different representations for different classes.

![TSNE](https://github.com/CognitionTree/Deep-ASL-Translator/blob/master/Python-Implementation/tsne/tsne_conv_model_dots.png)
