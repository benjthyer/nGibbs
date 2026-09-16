"""Exercise the sidecar mechanics on synthetic tables: attach, delete, split, grow, save."""
import sys, os, shutil, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/imp')
import numpy as np, pandas as pd
from builder.processing import sidecar as S

WORK='/home/claude/imp/sc'; shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK); os.chdir(WORK)

class Fake:                      # minimal stand-in exposing what sidecar.py uses
    chunk_size = 7
    def __init__(self, n, m):
        self.filename='tbl'
        self.table=np.arange(n*m, dtype=np.float32).reshape(n, m)
    def _chunked_mask_copy(self, src, dst, keep, chunk_size=None):
        out=np.lib.format.open_memmap(dst, mode='w+', dtype=src.dtype,
                                      shape=(int(keep.sum()), src.shape[1]))
        out[:]=np.asarray(src)[keep]; out.flush(); del out
    def _chunked_copy_into(self, src, dst, chunk_size=None):
        dst[:src.shape[0]]=np.asarray(src)

N, M = 20, 4
t = Fake(N, M)
for suffix, base in (('_dndP', 100.0), ('_dndT', 1000.0)):
    df = pd.DataFrame(base + np.arange(N*M, dtype=np.float32).reshape(N, M),
                      columns=[f'c{i}' for i in range(M)])
    df.to_csv(f'tbl{suffix}.csv', index=False)
found = S.attach(t, 'tbl', memmap_mode='r+', chunk_size=8, verbose=False)
print(f'attach -> {found}   dndp{t.dndp.shape} dndt{t.dndt.shape}')
assert t.dndp.shape == (N, M) and t.dndt.shape == (N, M)

keep = np.ones(N, bool); keep[[3, 7, 11]] = False
expect_dp = np.asarray(t.dndp)[keep].copy()
S.apply_mask(t, keep, 8)
t.table = t.table[keep]
print(f'delete  -> rows {t.table.shape[0]}  dndp {t.dndp.shape[0]}  '
      f'values preserved: {np.array_equal(np.asarray(t.dndp), expect_dp)}')
assert t.table.shape[0] == t.dndp.shape[0] == t.dndt.shape[0] == N-3

n2 = t.table.shape[0]
move = np.zeros(n2, bool); move[:5] = True; keep2 = ~move
moved_expect = np.asarray(t.dndp)[move].copy(); kept_expect = np.asarray(t.dndp)[keep2].copy()
new = Fake(int(move.sum()), M); new.filename='tbl_split'
S.split_into(t, new, move, keep2, 'tbl_split', 8)
t.table = t.table[keep2]
print(f'split   -> moved {new.dndp.shape[0]} kept {t.dndp.shape[0]}  '
      f'moved ok: {np.array_equal(np.asarray(new.dndp), moved_expect)}  '
      f'kept ok: {np.array_equal(np.asarray(t.dndp), kept_expect)}')

old_rows = t.table.shape[0]; nres = 3
cm = np.zeros(old_rows, bool); cm[[0, 2]] = True
plan = [(0, old_rows, cm, nres)]
before = np.asarray(t.dndp).copy()
new_total = old_rows + int(cm.sum())*nres
S.grow_with_repeats(t, new_total, old_rows, plan, 64)
t.table = np.zeros((new_total, M), np.float32)   # caller grows the main table after
dup_ok = np.array_equal(np.asarray(t.dndp)[old_rows:],
                        np.repeat(before[cm], nres, axis=0))
print(f'grow    -> {old_rows} -> {t.dndp.shape[0]} rows   duplicated block exact: {dup_ok}')
assert dup_ok

S.save(t, 'final')
print(f'save    -> {sorted(f for f in os.listdir(".") if f.startswith("final"))}')

t.table = np.zeros((t.dndp.shape[0]+1, M), np.float32)
try:
    S.assert_aligned(t, 'test'); print('MISALIGNMENT NOT CAUGHT')
except AssertionError as e:
    print(f'guard   -> misalignment raises: {str(e)[:62]}...')
