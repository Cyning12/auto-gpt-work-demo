import torch.nn as nn
import torch

projection = nn.Linear(12288, 128)
# print(projection)

input_tensor = torch.randn(3, 12288)
# print(input_tensor)

output_tensor = projection(input_tensor)
# print(output_tensor)

# 形状从 [3, 12288] -> [3, 96, 128]
view_output = input_tensor.view(3, 96, 128)
print("view_output=\n", view_output)

from einops import rearrange

einops_view = rearrange(input_tensor, "t (h d) -> t h d", h=96)
print("einops_view=\n", einops_view)
