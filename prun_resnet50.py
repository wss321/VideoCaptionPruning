import torch
from torchvision.models import resnet18, resnet50
import torch_pruning as tp

model = resnet18(pretrained=True).eval()
ori_size = tp.utils.count_params(model)
example_inputs = torch.randn(1, 3, 224, 224)
imp = tp.importance.MagnitudeImportance(p=2)  # L2 norm pruning
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m)

total_steps = 5
pruner = tp.pruner.LocalMagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    total_steps=total_steps,  # number of iterations
    ch_sparsity=0.5,  # channel sparsity
    ignored_layers=ignored_layers,  # ignored_layers will not be pruned
)

for i in range(total_steps):  # iterative pruning
    pruner.step()
    print(
        "  Params: %.2f M => %.2f M"
        % (ori_size / 1e6, tp.utils.count_params(model) / 1e6)
    )
