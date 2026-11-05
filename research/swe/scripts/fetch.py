# Stream row groups of the remote parquet, keep only uniformly sampled rows.
import fsspec, pyarrow.parquet as pq, pyarrow as pa, random, sys, time
URL=sys.argv[1]; OUT=sys.argv[2]; N=int(sys.argv[3]); SEED=int(sys.argv[4])
f=fsspec.open(URL,"rb",block_size=8<<20,cache_type="none").open()
pf=pq.ParquetFile(f); m=pf.metadata
total=m.num_rows
idx=sorted(random.Random(SEED).sample(range(total),N))
cols=["trajectory_id","instance_id","repo","trajectory","exit_status","resolved"]
tables=[]; start=0; t0=time.time()
for g in range(m.num_row_groups):
    n=m.row_group(g).num_rows
    want=[i-start for i in idx if start<=i<start+n]
    if want:
        t=pf.read_row_group(g,columns=cols)
        tables.append(t.take(pa.array(want)))
    start+=n
    print(g,len(want),round(time.time()-t0),flush=True)
pq.write_table(pa.concat_tables(tables),OUT,compression="zstd")
print("done",total,N)
