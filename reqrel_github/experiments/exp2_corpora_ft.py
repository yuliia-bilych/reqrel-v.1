import numpy as np

def finetune_ce(tr_txt, test_pairs, base="cross-encoder/nli-deberta-v3-xsmall", n_pos=1000, n_neg=2000, epochs=1):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(42); rng = np.random.default_rng(42)
    pos = [t for t in tr_txt if t[2] == 1]; neg = [t for t in tr_txt if t[2] == 0]
    pos = [pos[i] for i in rng.permutation(len(pos))[:n_pos]]; neg = [neg[i] for i in rng.permutation(len(neg))[:n_neg]]
    data_ = pos + neg; order = rng.permutation(len(data_))
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2, ignore_mismatched_sizes=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    model.train()
    for _ in range(epochs):
        for s0 in range(0, len(order), 16):
            b = [data_[k] for k in order[s0:s0 + 16]]
            enc_ = tok([x[0] for x in b], [x[1] for x in b], truncation=True, max_length=128, padding=True, return_tensors="pt")
            out = model(**enc_, labels=torch.tensor([x[2] for x in b]))
            out.loss.backward(); opt.step(); opt.zero_grad()
    model.eval(); preds = []
    with torch.no_grad():
        for s0 in range(0, len(test_pairs), 64):
            b = test_pairs[s0:s0 + 64]
            enc_ = tok([x[0] for x in b], [x[1] for x in b], truncation=True, max_length=128, padding=True, return_tensors="pt")
            preds += model(**enc_).logits.argmax(-1).tolist()
    return np.array(preds)

