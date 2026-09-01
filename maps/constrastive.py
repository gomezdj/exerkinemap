def contrastive_loss(zr, zp, temperature=0.1):
    zr = F.normalize(zr, dim=-1)
    zp = F.normalize(zp, dim=-1)
    logits = zr @ zp.T / temperature
    labels = torch.arange(len(zr), device=zr.device)
    return F.cross_entropy(logits, labels)
