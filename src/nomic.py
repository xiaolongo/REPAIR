import torch
import torch.nn.functional as F

def encode(batch, tokenizer, model):
    inputs=tokenizer(batch,padding=True,truncation=True,max_length=8192,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        hidden=model(**inputs).last_hidden_state
        mask=inputs['attention_mask'].unsqueeze(-1).float()
        pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
        normal=F.layer_norm(pooled,(pooled.shape[1],))
        features=F.normalize(normal[:,:384],p=2,dim=1)
    assert features.shape==(len(batch),384) and torch.isfinite(features).all()
    return features.cpu().numpy(),pooled.cpu().numpy()
