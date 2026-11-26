"""Stage 1: pull the 04:00-10:00 window out of final.parquet into small per-row-group part files.
Parallel over row groups. Output keeps the original sort order (ticker, date, dt)."""
import os, sys, time, glob
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from multiprocessing import Pool

F = r"C:/datum-api-examples-main/OriON/CRACEN/final.parquet"
OUT = r"C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/win"
COLS = ["ticker", "dt", "o", "h", "l", "c", "v", "prev_close", "Bench%", "sigma", "beta", "sigma_day", "beta_day"]
F32 = ["o", "h", "l", "c", "v", "prev_close", "Bench%", "sigma", "beta", "sigma_day", "beta_day"]


def work(g):
    outp = os.path.join(OUT, f"part_{g:04d}.parquet")
    if os.path.exists(outp):
        return g, -1
    f = pq.ParquetFile(F)
    t = f.read_row_group(g, columns=COLS)
    dt = t["dt"]
    hm = pc.utf8_slice_codeunits(dt, 11, 16)
    m = pc.and_(pc.greater_equal(hm, "04:00"), pc.less_equal(hm, "10:00"))
    t = t.filter(m)
    if t.num_rows == 0:
        return g, 0
    dt = t["dt"]
    hh = pc.cast(pc.utf8_slice_codeunits(dt, 11, 13), pa.int16())
    mm = pc.cast(pc.utf8_slice_codeunits(dt, 14, 16), pa.int16())
    minute = pc.add(pc.multiply(hh, pa.scalar(60, pa.int16())), mm)
    cols = {"ticker": t["ticker"], "date": pc.utf8_slice_codeunits(dt, 0, 10), "minute": minute}
    for c in F32:
        cols[c] = pc.cast(t[c], pa.float32())
    out = pa.table(cols)
    tmp = outp + ".tmp"
    pq.write_table(out, tmp, compression="zstd")
    os.replace(tmp, outp)
    return g, out.num_rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    n = pq.ParquetFile(F).metadata.num_row_groups
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else n
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    t0 = time.time(); rows = 0
    with Pool(workers) as p:
        for k, (g, r) in enumerate(p.imap_unordered(work, range(lim)), 1):
            rows += max(r, 0)
            if k % 25 == 0 or k == lim:
                print(f"[{k}/{lim}] rows so far {rows:,}  {time.time()-t0:,.0f}s", flush=True)
    print("done", len(glob.glob(os.path.join(OUT, "part_*.parquet"))), "parts", f"{time.time()-t0:,.0f}s", flush=True)
