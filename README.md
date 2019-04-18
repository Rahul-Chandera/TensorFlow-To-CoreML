# TensorFlow-To-CoreML
In this sample, I am demonstrating how to convert TensorFlow model to CoreML model.  I am assuming that you already trained your model with TensorFlow for image classification, and you already have .pb and .txt file which TensorFlow generates after completion of training.

Generally if you want to use TensorFlow model directly with iOS, you have to integrate TensorFlow lite sdk, which is little bit complex and will also increate your ipa file size. So I prefer to use native [Vision](https://developer.apple.com/documentation/vision) framework to perform machine learning tasks. Vision framework only supports mlmodel, so you have to convert your .pb file into .mlmodel file.

TFCoreML
-
TensorFlow provides tfcoreml package, which you can use to convert TensorFlow model to CoreML model. 

I suggest to use virtual environment, instead of install all this packages into base environment. [Conda](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) is a good option for virtual environment.
Open terminal and use this command to install tfcoreml Pypi package,
```
pip install -U tfcoreml
```
As of now, tfcoreml is only supported with python 2.7, so make sure to verify your current python version.
Use this command to check python version,
```
python --version
```

This is how you can convert model,
```
import tfcoreml
tfcoreml.convert(
    tf_model_path="my_model.pb",
    mlmodel_path="my_model.mlmodel",
    input_name_shape_dict={"input:0": [1, 224, 224, 3]},
    output_feature_names=["final_result:0"],
    image_input_names=["input:0"],
    class_labels="labels.txt",
)
```
Here, input and output parameters value will be different based on your model. You can find graph info of your model using tensorflow graph definition api. In "get_model_info.py" file, I have demonstrate how to do that.

You can also improve your mlmodel by providing red_bias, green_bias, blue_bias & image_scale parameters into convert function. And to get values for this parameters, you can use coremltools, once you have the mlmodel. In "get_image_param_info.py" file, I have demonstrate how to do that.


Reference:
-
- https://github.com/tf-coreml/tf-coreml


