from google.protobuf.json_format import MessageToDict

from .dataset_pb2 import *
from .geom_pb2 import *
from .map_pb2 import *
from .tags_pb2 import *


def proto_to_dict(proto, preserving_proto_field_name=True):
    return MessageToDict(proto, preserving_proto_field_name=preserving_proto_field_name)
