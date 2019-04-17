# TensorFlow-To-CoreML
In this sample, I am demonstrating how to convert TensorFlow model to CoreML model. 
I am assuming that you already trained your model with TensorFlow for image classification, and you already have .pb and .txt file which TF generates after completion of trainnig.

Generaly if you want to use TF model directly with iOS, you have to integrate TF lite sdk, which is litle bit complex and will also increate your ipa file size. I prefer to use Vision[https://developer.apple.com/documentation/vision] framework which is provided by Apple, and to use Vision framework, you required mlmodel.
