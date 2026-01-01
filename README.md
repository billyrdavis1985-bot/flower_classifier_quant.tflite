# Flower Classifier with TensorFlow Lite 🌻

A lightweight, mobile-ready image classifier that identifies 5 types of flowers (daisy, dandelion, rose, sunflower, tulip) with **100% validation accuracy**!

Built using transfer learning with **MobileNetV2** and quantized for on-device inference with TensorFlow Lite.

Perfect practice project for TensorFlow Lite, edge ML, and computer vision.

## Demo / Results

![Flower Classification Demo](https://i.ytimg.com/vi/qaRF1V6Zj9I/maxresdefault.jpg)

Training converged extremely fast — here's the accuracy/loss curves:

![Training Accuracy & Loss](https://www.researchgate.net/publication/364441531/figure/fig4/AS:11431281093798626@1667285871089/Training-Vs-Validation-Accuracy-for-MobileNetV2-Model.jpg)

- **Final Validation Accuracy**: 100%  
- **Final Validation Loss**: ~0.0016  
- Model size (quantized): ~2.2 MB (75% smaller than float version!)

## Model Files

- `flower_classifier_quant.tflite` ← **Recommended**: Int8 quantized for fastest mobile/edge performance
- (Add the float version if you upload it too)

Ready to drop into Android, iOS, Raspberry Pi, or Coral apps!

## How It Was Built

1. Used the classic TensorFlow flowers dataset (~3,670 images)
2. Transfer learning with pre-trained MobileNetV2 (froze base layers)
3. Data augmentation + sparse categorical crossentropy
4. Trained for 10 epochs on Google Colab GPU → perfect accuracy
5. Exported to TFLite with full int8 quantization

Full code in the Colab notebook (link or upload .ipynb here if you want).

## Usage Example (Inference)

```python
# Simple TFLite inference snippet
interpreter = tf.lite.Interpreter(model_path="flower_classifier_quant.tflite")
# ... (preprocess image, run inference, get prediction)
