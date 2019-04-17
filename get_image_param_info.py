# Get image pre-processing parameters of a saved CoreML model
import coremltools
import tfcoreml
from coremltools.proto import FeatureTypes_pb2 as _FeatureTypes_pb2

coreml_model_file = "./MyModel.mlmodel"

spec = coremltools.models.utils.load_spec(coreml_model_file)

if spec.WhichOneof("Type") == "neuralNetworkClassifier":
    nn = spec.neuralNetworkClassifier
if spec.WhichOneof("Type") == "neuralNetwork":
    nn = spec.neuralNetwork
if spec.WhichOneof("Type") == "neuralNetworkRegressor":
    nn = spec.neuralNetworkRegressor

preprocessing = nn.preprocessing[0].scaler
print("channel scale: ", preprocessing.channelScale)
print("blue bias: ", preprocessing.blueBias)
print("green bias: ", preprocessing.greenBias)
print("red bias: ", preprocessing.redBias)

inp = spec.description.input[0]
if inp.type.WhichOneof("Type") == "imageType":
    colorspace = _FeatureTypes_pb2.ImageFeatureType.ColorSpace.Name(
        inp.type.imageType.colorSpace
    )
    print("colorspace: ", colorspace)

