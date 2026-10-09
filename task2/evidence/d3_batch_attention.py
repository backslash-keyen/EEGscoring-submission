"""Defect 3 evidence: with batch_first=False the (B, L, d) input is read as (L=B, B=L, d), so attention mixes WINDOWS.
Same weights, two layouts. Test 1: replace one neighbouring epoch of window 0 -> does window 0's output move?
Test 2: replace a whole other window in the batch -> does window 0's output move (it should not)?"""
import sys, torch, torch.nn as nn
sys.path.insert(0, "..")
from sleep_pipeline import SleepTransformer, SEQ_LEN, D_MODEL, set_seed

def model(batch_first):
    set_seed(0)
    m = SleepTransformer(n_ch=3)
    state = m.state_dict()
    layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=4, dim_feedforward=128, dropout=0.1,
                                       batch_first=batch_first)
    m.transformer = nn.TransformerEncoder(layer, num_layers=2)
    m.load_state_dict(state)                                       # identical weights in both layouts
    return m.eval()

set_seed(1)
x = torch.randn(16, SEQ_LEN, 3, 3000)
c = SEQ_LEN // 2
for label, bf in (("given (batch_first=False)", False), ("fixed (batch_first=True) ", True)):
    m = model(bf)
    with torch.no_grad():
        base = m(x)
        x1 = x.clone(); x1[0, c + 1] = torch.randn(3, 3000)        # neighbour of window 0's centre epoch
        own = float((m(x1)[0] - base[0]).abs().max())
        x2 = x.clone(); x2[5] = torch.randn(SEQ_LEN, 3, 3000)      # an unrelated window in the same batch
        other = float((m(x2)[0] - base[0]).abs().max())
    print(f"{label}: change in window 0's output when its own neighbour changes {own:.2e}; "
          f"when another window in the batch changes {other:.2e}")
