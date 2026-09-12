def test_rna_forward():
    from exerkinemap.models.ExerkineRNA import ExerkineRNA, rna_seq_to_ids
    model = ExerkineRNA()
    ids = rna_seq_to_ids("AUGCUUAGC")
    import torch
    x = torch.tensor([ids])
    out = model(x)
    assert "seq_embedding" in out

def test_protein_forward():
    from exerkinemap.models.ExerkineProtein import ExerkineProtein, protein_seq_to_ids, protein_features_to_ids
    model = ExerkineProtein()
    ids = protein_seq_to_ids("MKTLLILAVV")
    feats = protein_features_to_ids("MKTLLILAVV")
    import torch
    x = torch.tensor([ids])
    out = model(x, [feats])
    assert "seq_embedding" in out

def test_map_forward():
    import torch
    from exerkinemap.maps.ExerkineMap import ExerkineMap
    model = ExerkineMap()
    zr, zp = model(torch.randn(2,512), torch.randn(2,512))
    assert zr.shape == zp.shape

