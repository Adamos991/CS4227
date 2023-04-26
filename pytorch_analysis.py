import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Define models for each component or group of related components
class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1d = nn.Conv1d(3, 6, 5)
        self.conv2d = nn.Conv2d(3, 6, 5)
        self.conv3d = nn.Conv3d(3, 6, 5)

    def forward(self, x1, x2, x3):
        x1 = self.conv1d(x1)
        x2 = self.conv2d(x2)
        x3 = self.conv3d(x3)
        return x1, x2, x3

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=8)

    def forward(self, src, tgt):
        src = self.encoder_layer(src)
        tgt = self.decoder_layer(tgt, src)
        return src, tgt

class ActivationModel(nn.Module):
    def __init__(self):
        super(ActivationModel, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.relu = nn.ReLU()
        self.threshold = nn.Threshold(threshold=0.0, value=0.0)

    def forward(self, x):
        attn_output, _ = self.multihead_attention(x, x, x)
        x_relu = self.relu(x)
        x_threshold = self.threshold(x)
        return attn_output, x_relu, x_threshold

# Create instances of the models
conv_model = ConvModel()
transformer_model = TransformerModel()
activation_model = ActivationModel()

# Create input tensors for the models
input_tensor_1d = torch.randn(1, 3, 32)
input_tensor_2d = torch.randn(1, 3, 32, 32)
input_tensor_3d = torch.randn(1, 3, 32, 32, 32)
input_tensor_transformer = torch.randn(10, 1, 64)

# Create a SummaryWriter for each model
conv_writer = SummaryWriter('runs/conv_model')
transformer_writer = SummaryWriter('runs/transformer_model')
activation_writer = SummaryWriter('runs/activation_model')

# Add the model graphs
conv_writer.add_graph(conv_model, (input_tensor_1d, input_tensor_2d, input_tensor_3d))
transformer_writer.add_graph(transformer_model, (input_tensor_transformer, input_tensor_transformer))
activation_writer.add_graph(activation_model, (input_tensor_transformer,))

# Close the SummaryWriters
conv_writer.close()
transformer_writer.close()
activation_writer.close()







# import torch
# from torch.nn.modules.conv import _ConvNd, Conv1d, Conv2d, Conv3d
# from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerDecoderLayer
# from torch.nn.modules.activation import MultiheadAttention, ReLU, Threshold
# from torch.nn.functional import relu, threshold
# from torch.autograd.function import _SingleLevelFunction, Function
# from torch.utils.checkpoint import CheckpointFunction
# from pycallgraph2 import PyCallGraph, Config
# from pycallgraph2.output import GraphvizOutput
# from pycallgraph2.globbing_filter import GlobbingFilter
# from typing import Optional

# def custom_filter_func(name: str, module_name: Optional[str] = None) -> bool:
#     if module_name is None:
#         return False
#     if module_name.startswith("torch") or name.startswith("run_pytorch_components"):
#         return True
#     return False

# class CustomGlobbingFilter(GlobbingFilter):
#     def __call__(self, full_name: str) -> bool:
#         name, _, module_name = full_name.partition('.')
#         if module_name is None:
#             return False
#         if module_name.startswith("torch") or name.startswith("run_pytorch_components"):
#             return True
#         return False

# #from torch.autograd.function import Function as TorchFunction
# config = Config(max_depth=10)
# #config.trace_filter = GlobbingFilter(include=['torch.*', 'SimpleModule*', 'forward*'])
# config.trace_filter = CustomGlobbingFilter()

# class CustomConv2d(torch.nn.Conv2d):
#     def forward(self, input):
#         return super().forward(input)

# class CustomTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
#     def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
#                 src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         return super().forward(src, src_mask, src_key_padding_mask)
# # Create instances of the classes
# # conv1d = Conv1d(3, 6, 5)
# # #conv2d = Conv2d(3, 6, 5)
# # conv2d = CustomConv2d(3, 6, 5)
# # conv3d = Conv3d(3, 6, 5)

# #transformer_encoder_layer = TransformerEncoderLayer(d_model=64, nhead=8)
# transformer_encoder_layer = CustomTransformerEncoderLayer(d_model=64, nhead=8)
# transformer_decoder_layer = TransformerDecoderLayer(d_model=64, nhead=8)

# multihead_attention = MultiheadAttention(embed_dim=64, num_heads=8)
# relu_activation = ReLU()
# threshold_activation = Threshold(threshold=0.0, value=0.0)

# class CustomSingleLevelFunction(_SingleLevelFunction):
#     @staticmethod
#     def forward(ctx, input):
#         return input.clone()

# class CustomFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return input.clone()

# def simple_function(x):
#     return x
# # # Dummy input data
# # input_tensor_1d = torch.randn(1, 3, 32)
# # input_tensor_2d = torch.randn(1, 3, 32, 32)
# # input_tensor_3d = torch.randn(1, 3, 32, 32, 32)

# # # Wrapper function to call the components
# # def run_pytorch_components():
# #     # Convolution layers
# #     conv1d_result = conv1d(input_tensor_1d)  # Update input_tensor to input_tensor_1d
# #     conv2d_result = conv2d(input_tensor_2d)  # Update input_tensor to input_tensor_2d
# #     conv3d_result = conv3d(input_tensor_3d)

# #     # Transformer layers
# #     src = torch.randn(10, 32, 512)
# #     tgt = torch.randn(20, 32, 512)
# #     encoder_result = transformer_encoder_layer(src)
# #     decoder_result = transformer_decoder_layer(tgt, src)

# #     # Multihead attention
# #     attn_output, attn_output_weights = multihead_attention(src, src, src)

# #     # Activation functions
# #     relu_result = relu(src)
# #     threshold_result = threshold(src, 0.0, 0.0)

# #     # This is just to avoid unused variable warnings
# #     _ = (conv1d_result, conv2d_result, conv3d_result, encoder_result,
# #          decoder_result, attn_output, attn_output_weights, relu_result, threshold_result)
# def returnX(x):
#     return x

# def run_pytorch_components():
#     conv1d = Conv1d(3, 6, 5)
#     #conv2d = Conv2d(3, 6, 5)
#     conv2d = CustomConv2d(3, 6, 5)
#     conv3d = Conv3d(3, 6, 5)
#     input_tensor_1d = torch.randn(1, 3, 32)
#     input_tensor_2d = torch.randn(1, 3, 32, 32)
#     input_tensor_3d = torch.randn(1, 3, 32, 32, 32)
#     torch.randn(1, 3, 32, 32, 32)
#     input_tensor_transformer = torch.randn(32, 1, 64)
#     returnX(5)
#     class SimpleModule(torch.nn.Module):
#         def forward(self, x):
#             return x
#     simple_module = SimpleModule()
#     # Convolution layers
#     conv1d_result = conv1d(input_tensor_1d)
#     conv2d_result = conv2d(input_tensor_2d)
#     conv3d_result = conv3d(input_tensor_3d)

#     # Transformer layers
#     transformer_encoder_result = transformer_encoder_layer(input_tensor_transformer)
#     transformer_decoder_result = transformer_decoder_layer(input_tensor_transformer, transformer_encoder_result)

#     # MultiheadAttention
#     multihead_attention_result, _ = multihead_attention(input_tensor_transformer, input_tensor_transformer, input_tensor_transformer)

#     # Activation functions
#     relu_result = relu(input_tensor_2d)
#     threshold_result = threshold(input_tensor_2d, 0.5, 0)

#     # Torch Tensor
#     tensor_result = torch._tensor.Tensor.sum(input_tensor_2d)

#     # Autograd Functions
#     single_level_function_result = CustomSingleLevelFunction.apply(input_tensor_2d)
#     function_result = CustomFunction.apply(input_tensor_2d)
#     CheckpointFunction(simple_module, (input_tensor_2d,))

#     # Checkpoint function
#     checkpoint_function_result = CheckpointFunction(simple_module, (input_tensor_2d,))


# # Set up PyCallGraph
# graphviz = GraphvizOutput()
# graphviz.output_file = 'pytorch_callgraph.png'

# # Run the PyTorch components with PyCallGraph
# with PyCallGraph(output=graphviz, config=config):
#     run_pytorch_components()

# with torch.profiler.profile(
#     schedule=torch.profiler.schedule(wait=0, warmup=2, active=6),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True
# ) as prof:
#     for _ in range(8):
#         run_pytorch_components()
#         prof.step()
