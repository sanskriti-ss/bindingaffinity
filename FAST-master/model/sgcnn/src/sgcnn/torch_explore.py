import torch
from torch_geometric.data import Data, Batch

# # Create some example Data objects
# data1 = Data(x=torch.randn(5, 16), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
# data2 = Data(x=torch.randn(3, 16), edge_index=torch.tensor([[0, 1], [1, 2]]))
# data3 = Data(x=torch.randn(7, 16), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))

# # Create a list of Data objects
# data_list = [data1, data2, data3]

# # Create a Batch object from the list
# batch = Batch.from_data_list(data_list)

# print(batch)
# print(batch.x.shape) # Batched node features
# print(batch.edge_index.shape) # Batched edge indices
# print(batch.batch.shape) # Assignment vector

x = torch.cat( [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]) ])
print(type(x))
print(x.shape)
print(x)
