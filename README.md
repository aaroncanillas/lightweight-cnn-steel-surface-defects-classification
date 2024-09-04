
# Image-based Classification of Steel Surface Defects using Lightweight Convolutional Neural Networks (CNNs)

This project showcases an image-based classification system for steel surface defects developed by adapting three popular CNN architectures: MobileNetV1, EfficientNetB0, and VGG-16 through transfer learning using the TensorFlow and Keras API. Grid search technique was also performed to obtain the most suitable learning rate parameter for the models, with all networks achieving the best results using a learning rate of **0.0001**. Post-training quantization (PTQ) techniques, specifically Float16 (16-bit) and full integer (8-bit) quantization, were applied to the best-performing models for each architecture using TensorFlow Lite.

The 16-bit MobileNetV1 model experienced minimal accuracy degradation, achieving **94.44% ± 0.94%**. In contrast, the 8-bit MobileNetV1 model showed a significant performance reduction, with an accuracy of **58.07% ± 6.03%**. The 16-bit EfficientNetB0 model was able to preserve accuracy similar to the original model, achieving **99.48% ± 0.38%**. Minimal performance degradation was observed in the 8-bit EfficientNetB0 model, which attained an accuracy of **97.56% ± 1.01%**. Similarly, the 16-bit VGG-16 model maintained accuracy close to the original, achieving **99.63% ± 0.33%**. Minimal performance degradation was also observed in the 8-bit VGG-16 model, which achieved an accuracy of **99.56% ± 0.28%**. Additionally, the model sizes on disk were observed to decrease in accordance with their bit-precision.

## Methodology

![Flowchart](https://github.com/aaroncanillas/lightweight-cnn-steel-surface-defects-classification/blob/1b960105e2498205084c6e5c046f5fa9d6fc7742/flowchart.png)

**Data Collection**

The dataset used in this project is the Northeastern University (NEU) surface defect database. It contains 1,800 grayscale images of six types of defects occurring on steel surfaces, namely: crazing, inclusion, rolled-in scale, scratches, patches, and pitted surface. Additionally, all images have an original uniform resolution of 200 × 200 pixels. The image dataset is obtained from http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm.

**Data Preprocessing**

The size of the images in the dataset were reshaped into 224 × 224 pixels to match the expected input shape of the networks using the OpenCV library. Afterwards, the image dataset is divided into a subset, for training and validation, and a testing set, each containing 85% and 15% of the whole dataset respectively.

**Model Development**

Pre-trained MobileNetV1, EfficientNetB0, and VGG-16 models on the ImageNet database are adapted through the use of the transfer learning technique using Keras API. On top of the pre-trained models, two fully connected layer containing 128 neurons 28 and 64 neurons with ReLU activation functions were added respectively. A dropout layer of 20% dropout probability is included between the two fully connected layers to prevent overfitting and increase the performance on unseen data. Lastly, the output layer is composed of 6 nodes and a softmax activation function to classify the 6 surface defect classes.

**Training the Model**

The models are trained using the subset containing 85% of the dataset’s images Stratified 5-fold cross-validation is also implemented to prevent data imbalance issues and provides consistent estimates of the performance scores. Additionally, grid search technique is employed to determine the most suitable learning rate parameter for the models. Lastly, the models were compiled using Adam optimizer and categorical cross entropy was employed as the loss function.

**Implementation of Post-training Quantization**

The trained models were converted into the TensorFlow Lite version using the TensorFlow Lite library. To further optimize the trained models, post-training quantization techniques were implemented. In this project, float16 quantization and full integer quantization were implemented to quantize the original MobileNetV1, EfficientNetB0, and VGG-16 models and determine their effect on the specified performance metrics.





## Tech Stack

**Programming Language:** Python

**Libraries:** TensorFlow, Keras, Pandas, NumPy, Matplotlib, Scikit-learn

**IDE:** Jupyter Notebook

## Results

![mb1](https://github.com/aaroncanillas/lightweight-cnn-steel-surface-defects-classification/blob/1b960105e2498205084c6e5c046f5fa9d6fc7742/results/mobilenetv1%20results.png)

![eff0](https://github.com/aaroncanillas/lightweight-cnn-steel-surface-defects-classification/blob/1b960105e2498205084c6e5c046f5fa9d6fc7742/results/efficientnetb0%20results.png)

![vgg16](https://github.com/aaroncanillas/lightweight-cnn-steel-surface-defects-classification/blob/1b960105e2498205084c6e5c046f5fa9d6fc7742/results/vgg%2016%20results.png)

![modelsize](https://github.com/aaroncanillas/lightweight-cnn-steel-surface-defects-classification/blob/1b960105e2498205084c6e5c046f5fa9d6fc7742/results/model%20size%20results.png)

![inftime](https://github.com/aaroncanillas/lightweight-cnn-steel-surface-defects-classification/blob/1b960105e2498205084c6e5c046f5fa9d6fc7742/results/inference%20time%20results.png)


The study successfully demonstrated that the MobileNetV1, EfficientNetB0, and VGG-16 models were able to perform well in classifying images of steel surface defects. EfficientNetB0 and VGG-16 models consistently demonstrated better accuracy, precision, and recall compared to MobileNetV1 models. Meanwhile, MobileNetV1 models have the advantage of having lower model sizes and inference time than the EfficientNetB0 and VGG-16 models. Accordingly, the EfficientNetB0 models acquired comparable accuracy, precision, and recall performances while acquiring lower inference time and model size than the VGG-16 models.
The implementation of post-training quantization techniques has been effective in decreasing the model sizes across the three architectures, while only causing minimal performance degradation. Therefore, PTQ can be effectively implemented on applications that are resource-limited. The only exception to this was the case of the 8-bit quantized MobileNetV1 models, wherein quantization led to a significant decline in the model performance.


## References

 - Song, K., & Yan, Y. (2013). A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects. Applied Surface Science, 285, 858–864. https://doi.org/10.1016/j.apsusc.2013.09.002

 - Song, K. & Yan, Y. (2013). NEU surface defect database [Data set]. http://faculty.neu.edu.cn/songkc/en/zhym/263264/list/index.htm


