import numpy as np
import torch
import torch.nn.functional as F

def denoise(author, labels, edges, real, pod, classes, steps=3):
    # Call the pinned author's sum propagation, including its self-loop and
    # zero-dropout Bernoulli behavior. Only L labels and pseudo OOD are supplied.
    onehot = torch.zeros(len(labels), classes + 1, device=labels.device)
    onehot[real] = F.one_hot(labels[real], classes + 1).float()
    onehot[pod, classes] = 1.
    with torch.no_grad():
        propagated = author.labelConv().label_propagate(onehot, edges, real, pod, order=steps, drop_rate=0)
    keep = pod[(propagated[pod].argmax(1) == classes).cpu().numpy()]
    if len(keep) == 0:
        raise ValueError("CFC denoising retained no pseudo OOD; mixup is undefined")
    return keep, propagated

def mixup_loss(net, x, edges, labels, real, pod, beta, count, classes):
    hidden = net.Conv1(x, edges)
    center = hidden[pod].mean(0)
    # Source ranks max raw logits, not softmax confidence. Selection has no
    # gradient, while both parents of the mixed representations retain theirs.
    with torch.no_grad():
        confidence = net(x, edges)[real].max(1).values
        boundary = real[torch.sort(confidence).indices.cpu().numpy()[:count]]
    mixed = beta * hidden[boundary] + (1 - beta) * center
    expanded = torch.cat([hidden, mixed])
    logits = net.clf1(net.Conv2(expanded, edges))
    selected = np.concatenate([real, pod])
    supervised = torch.cat([logits[selected], logits[-len(mixed):]])
    targets = torch.cat([labels[real], torch.full((len(pod) + len(mixed),), classes,
                                                 dtype=torch.long, device=labels.device)])
    return F.cross_entropy(supervised, targets), boundary
