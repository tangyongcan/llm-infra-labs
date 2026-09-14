#!/usr/bin/env bash
# Partial measurement matrix. Each invocation is its own process so CUDA
# memory numbers stay clean. Keep going on failure so a broken stage does
# not hide the rest.
#
# This script writes:
#   results_{fp8-weight,int8}.json
#   results_{bf16,int8,fp8-weight,fp8-dynamic}_ppl.json
#   results_{bf16,fp8-dynamic,fp8-weight}_compiled_decode1.json
#
# It does not write eager bf16 / fp8-dynamic speed files or any compiled
# prefill files. Those committed JSONs were produced by the extra
# compress_bench.py commands listed in README.md.
set +e
cd "$(dirname "$0")"

run() {
  echo "######## $* ########"
  python -u compress_bench.py "$@" 2>&1 \
    | grep -viE "^Loading weights|Warning: You are sending|_pytree|Not enough SMs|_stable_hash_for_caching|warn_once"
  echo "-------- exit=$? --------"
}

# Eager speed + memory matrix.
run --mode fp8-weight --skip-ppl
run --mode int8 --skip-ppl

# Perplexity regression, identical eval for every mode.
for m in bf16 int8 fp8-weight fp8-dynamic; do
  run --mode "$m" --only ppl
done

# Compiled decode: inductor fusion is what makes torchao fp8 competitive.
# bitsandbytes int8 is excluded: Linear8bitLt does not compile cleanly.
for m in bf16 fp8-dynamic fp8-weight; do
  run --mode "$m" --compile --only decode1
done

echo "ALL RUNS DONE"
