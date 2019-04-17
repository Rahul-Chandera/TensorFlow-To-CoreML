""" CONVERT TF TO CORE ML """
import tfcoreml

# Model Shape
input_tensor_shapes = {"input:0": [1, 224, 224, 3]}

# Input Name
image_input_name = ["input:0"]

# Output CoreML model path
coreml_model_file = "./MyModel.mlmodel"

# Output name
output_tensor_names = ["final_result:0"]

# PB file path
tf_model_path = "./graph.pb"

# Label file for classification
class_labels = "./labels.txt"

# Convert Process
coreml_model = tfcoreml.convert(
    tf_model_path=tf_model_path,
    mlmodel_path=coreml_model_file,
    input_name_shape_dict=input_tensor_shapes,
    output_feature_names=output_tensor_names,
    image_input_names=image_input_name,
    class_labels=class_labels,
)

# coreml_model = tfcoreml.convert(
#         tf_model_path=tf_model_path,
#         mlmodel_path=coreml_model_file,
#         input_name_shape_dict=input_tensor_shapes,
#         output_feature_names=output_tensor_names,
#         image_input_names = image_input_name,
#         class_labels = class_labels,
#         red_bias = -1,
#         green_bias = -1,
#         blue_bias = -1,
#         image_scale = 2.0/255.0)
