#!/bin/bash
# Certified run of the full model (sat_full.py, both stages) for one block type:
#   export DIMACS, run the CaDiCaL binary with DRAT proof output, verify UNSAT proofs with drat-trim.
# Usage: bash certify_type.sh TYPES.json ID SECONDS [--alldis]
# Output line (appended to certs/ledger.txt): ID status [drat-trim verdict, sha256 of cnf and proof]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
TYPES=$1; ID=$2; SECS=$3; FLAG=${4:-}
mkdir -p "$HERE/certs"
cd "$HERE"
python3 sat_full.py "$TYPES" "$ID" 1 $FLAG --dimacs > "certs/type${ID}${FLAG}.encode.out" 2>&1 || { echo "$ID ENCODE-ERROR" >> certs/ledger.txt; exit 1; }
CNF=$(grep -o 'whole_[a-z]*_[0-9]*\.cnf' "certs/type${ID}${FLAG}.encode.out" | head -1)
mv "$CNF" "certs/type${ID}${FLAG}.cnf"
CNF="certs/type${ID}${FLAG}.cnf"; PR="certs/type${ID}${FLAG}.drat"
timeout "$SECS" "$ROOT/tools/cadical/build/cadical" -q "$CNF" "$PR" > "certs/type${ID}${FLAG}.cadical.out" 2>&1; rc=$?
if [ $rc = 20 ]; then
  "$ROOT/tools/drat-trim/drat-trim" "$CNF" "$PR" > "certs/type${ID}${FLAG}.drattrim.out" 2>&1
  V=$(grep -o 's VERIFIED\|s NOT VERIFIED' "certs/type${ID}${FLAG}.drattrim.out" | head -1)
  echo "$ID${FLAG} UNSAT drat-trim:${V:-none} cnf:$(sha256sum "$CNF" | cut -c1-16) proof:$(sha256sum "$PR" | cut -c1-16) $(du -h "$PR" | cut -f1)" >> certs/ledger.txt
elif [ $rc = 10 ]; then
  echo "$ID${FLAG} SAT (witness in certs/type${ID}${FLAG}.cadical.out)" >> certs/ledger.txt
else
  echo "$ID${FLAG} UNKNOWN (rc $rc after ${SECS}s)" >> certs/ledger.txt; rm -f "$PR"
fi
tail -1 certs/ledger.txt
