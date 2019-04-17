import tensorflow as tf
import tfcoreml
from coremltools.proto import FeatureTypes_pb2 as _FeatureTypes_pb2
import coremltools

"""FIND GRAPH INFO"""
tf_model_path = "./graph.pb"
with open(tf_model_path, "rb") as f:
    serialized = f.read()
tf.reset_default_graph()
original_gdef = tf.GraphDef()
original_gdef.ParseFromString(serialized)

with tf.Graph().as_default() as g:
    tf.import_graph_def(original_gdef, name="")
    ops = g.get_operations()
    N = len(ops)
    for i in [0, 1, 2, N - 3, N - 2, N - 1]:
        print('\n\nop id {} : op type: "{}"'.format(str(i), ops[i].type))
        print("input(s):")
        for x in ops[i].inputs:
            print("name = {}, shape: {}, ".format(x.name, x.get_shape()))
        print("\noutput(s):"),
        for x in ops[i].outputs:
            print("name = {}, shape: {},".format(x.name, x.get_shape()))
