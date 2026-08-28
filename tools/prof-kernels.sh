#!/bin/bash
# Per-kernel breakdown of a decode run under rocprofv3 (run inside the ROCm container via ./build.sh).
# usage: tools/prof-kernels.sh <ntok-forwards> <cmd...>   e.g. tools/prof-kernels.sh 41 ./build/runfn $M docs/ref/fn-code.json 4
NTOK=$1; shift
rm -rf build/prof; rocprofv3 --kernel-trace -d build/prof -o k --output-format csv -- "$@" > build/prof-run.log 2>&1
f=$(find build/prof -name '*kernel_trace.csv' | head -1)
python3 - "$f" "$NTOK" <<'PY'
import csv, sys, collections
f, ntok = sys.argv[1], int(sys.argv[2])
agg = collections.defaultdict(lambda: [0, 0.0]); tot = 0.0; t0 = None; t1 = None
for r in csv.DictReader(open(f)):
    s, e = int(r['Start_Timestamp']), int(r['End_Timestamp']); d = (e - s) / 1e6
    n = r['Kernel_Name'].replace('(anonymous namespace)::', '').replace('void ', '').replace('(WFmt)', 'F'); n = n.split('(')[0][:70]
    agg[n][0] += 1; agg[n][1] += d; tot += d
    t0 = s if t0 is None else min(t0, s); t1 = e if t1 is None else max(t1, e)
print(f"kernel time per forward: {tot/ntok:.2f} ms ({ntok} forwards; trace span {(t1-t0)/1e6/ntok:.1f} ms/forward incl. load)")
for n, (c, d) in sorted(agg.items(), key=lambda x: -x[1][1])[:24]:
    print(f"  {n:70s} {c/ntok:6.0f}/fwd {d/ntok:8.2f} ms/fwd  avg {d/c*1000:7.1f} us {100*d/tot:5.1f}%")
PY
