#!/bin/bash
# Cloud worker (note 44): runs a task file on all cores with minimal console output (the agent only polls 'status').
#   setup                       pip install the numeric stack (quiet)
#   task types also: 'adr START END' / 'adt reps logs...' (structures' AD sweep; adr runs in phase 1), 'U json index prefix' (unified model), 'fb script cfg seconds workers' (structures' full-block CP-SAT; run the fb file with P=1), 'g18s START END' (C1's g-=18 t=0 split sweep, recorded) and 'g18r T START END' (C1's torsion regimes), 'gp G' (C1's even test g+ = G, G = 14 or 16), 'satp cfg secs' (structures' sat_local via pysat: saves a local solution if SAT), 'tier / tier2 TYPES.json i secs [proof]' (C1's block types; tier2 = structures' encoding 2 + the same lex-leader: CNF with lex-leader, CaDiCaL; SAT -> decoded, checked local solution; proof -> DRAT + drat-trim), 'rt ci rep' (root_job.py)
#   run <taskfile> [P]          phase 1: 'gen ci rep' lines; phase 2: 'run ci rep lo hi' lines (stage 1, cloud_stage1.py);
#                               also 'or ci idx' (orientation root LP) and 'fx json index prefix' (unified block fixing)
#   status <taskfile>           one line: done/total and the last result
#   push <branch> <message>     commit the result files and push them to <branch>
cd "$(dirname "$0")/.."
cmd=$1; shift
case "$cmd" in
setup)
  python3 -m pip install -q --disable-pip-version-check numpy scipy highspy networkx ortools python-sat > /tmp/pip.log 2>&1 && echo SETUP OK || { echo SETUP FAILED; tail -3 /tmp/pip.log; } ;;
run)
  T=$1; P=${2:-$(nproc)}; mkdir -p logs/cloud_stage1 logs/cloud_done
  one() {
    set -- $1; key=$(echo "$*" | tr ' /' '__'); [ -f "logs/cloud_done/$key" ] && return
    case "$1" in
      gen) python3 scripts/cloud_stage1.py gen $2 $3 >> logs/cloud_worker.log 2>&1 ;;
      run) MAXBB=0 python3 scripts/cloud_stage1.py run $2 $3 $4 $5 >> logs/cloud_worker.log 2>&1 ;;
      or)  ONLY=$3 SMOOTH=0.6 LOWMEM=1 TL=20000 python3 scripts/orient_campaign.py $2 >> logs/cloud_worker.log 2>&1 ;;
      fx)  SMOOTH=0.6 LOWMEM=1 TL=20000 python3 scripts/run_U_fix.py $2 $3 auto $4 >> logs/cloud_worker.log 2>&1 ;;
      g18s) TL1=60 TL2=60 python3 scripts/c1_g18_sweep_wrap.py $2 $3 >> logs/cloud_worker.log 2>&1 ;;
      fb)  mkdir -p logs/fullblock && o="logs/fullblock/${2%.py}_${3%.json}.out" && (cd scripts/structures && python3 $2 $3 $4 $5 > "../../$o" 2>&1; cp cfg_*sol*.json ../../logs/fullblock/ 2>/dev/null; true) && echo "FB $2 $3: $(grep -h -i -E 'status|infeasible|feasible|unknown' "$o" | tail -n 1 | cut -c1-100)" >> logs/cloud_worker.log ;;
      adr) mkdir -p logs/ad && (cd scripts/structures_ad && python3 run_relaxed.py ad_reps_10to21.json $2 $3 > ../../logs/ad/relaxed1017_$2_$3.log 2>&1) && echo "ADR $2 $3 done" >> logs/cloud_worker.log ;;
      adt) mkdir -p logs/ad && n=$(echo "$*" | tr ' /' '__' | cut -c1-80) && (cd scripts/structures_ad && ls ../../logs/ad/relaxed1017_*.log >/dev/null 2>&1 && cp ../../logs/ad/relaxed1017_*.log . 2>/dev/null; python3 retest_relaxed.py ${@:2} > "../../logs/ad/retest_$n.log" 2>&1) && echo "ADT $2 done" >> logs/cloud_worker.log ;;
      U)   LOWMEM=1 SMOOTH=0.6 TL=20000 python3 scripts/run_U.py $2 $4 $3 >> logs/cloud_worker.log 2>&1 ;;
      g18c) mkdir -p logs/c1_g18c && (cd scripts/c1 && Q7_POINTWISE=1 python3 q7_rigid_cpsat2.py $2 $3 $4 > ../../logs/c1_g18c/$2.out 2>&1) && echo "G18C $2: $(tail -n 1 logs/c1_g18c/$2.out | cut -c1-80)" >> logs/cloud_worker.log ;;
      sat) b=${2%.json}; sb=""; if [ -n "$4" ]; then b="${b}_sb$4"; sb="--symbreak --prefix=$4"; fi; cp "scripts/structures/$2" "scripts/structures/$b.json" 2>/dev/null; mkdir -p logs/sat_local tools
           [ -x tools/cadical/build/cadical ] || (cd tools && rm -rf cadical && git clone -q https://github.com/arminbiere/cadical.git && cd cadical && ./configure > /dev/null 2>&1 && make -j4 > /dev/null 2>&1)
           [ -x tools/drat-trim/drat-trim ] || (cd tools && rm -rf drat-trim && git clone -q https://github.com/marijnheule/drat-trim.git && cd drat-trim && make > /dev/null 2>&1)
           (cd scripts/structures && python3 sat_local.py "$b.json" 0 cadical195 --dimacs $sb > "../../logs/sat_local/$b.encode.out" 2>&1)
           cnf="scripts/structures/$b.cnf"; pr="/tmp/$b.drat"
           timeout $3 tools/cadical/build/cadical -q "$cnf" "$pr" > "logs/sat_local/$b.cadical.out" 2>&1; rc=$?
           if [ $rc = 20 ]; then tools/drat-trim/drat-trim "$cnf" "$pr" > "logs/sat_local/$b.drattrim.out" 2>&1
             echo "SAT $2: UNSAT, drat-trim $(grep -o 's VERIFIED\|s NOT VERIFIED' logs/sat_local/$b.drattrim.out | head -1), cnf sha $(sha256sum "$cnf" | cut -c1-16), proof sha $(sha256sum "$pr" | cut -c1-16) $(du -h "$pr" | cut -f1)" >> logs/cloud_worker.log
           elif [ $rc = 10 ]; then echo "SAT $2: SATISFIABLE (witness in logs/sat_local/$b.cadical.out)" >> logs/cloud_worker.log
           else echo "SAT $2: UNKNOWN (cadical rc $rc after $3 s)" >> logs/cloud_worker.log; fi
           true ;;
      g18t1) mkdir -p logs/c1_g18t1 && (cd scripts/c1 && Q7_POINTWISE=1 python3 q7_t1_cpsat.py $2 $3 $4 > ../../logs/c1_g18t1/$2.out 2>&1) && echo "G18T1 $2: $(tail -n 1 logs/c1_g18t1/$2.out | cut -c1-80)" >> logs/cloud_worker.log ;;
      tier) mkdir -p logs/tiers tools && t=${2%.json} && n=${t}_$(printf %04d $3) && cd_=/tmp/tiers_cnf/$t && cnf=$cd_/type_$(printf %04d $3).cnf
           [ -x tools/cadical/build/cadical ] || (cd tools && rm -rf cadical && git clone -q https://github.com/arminbiere/cadical.git && cd cadical && ./configure > /dev/null 2>&1 && make -j4 > /dev/null 2>&1)
           (cd scripts/c1_tiers && SAT_LOCAL_DIR=../structures python3 sat_local_batch.py data/$2 $3 $(( $3 + 1 )) 0 --prefix 6 --dimacs $cd_ > ../../logs/tiers/$n.encode.out 2>&1)
           if [ "$5" = proof ]; then
             [ -x tools/drat-trim/drat-trim ] || (cd tools && rm -rf drat-trim && git clone -q https://github.com/marijnheule/drat-trim.git && cd drat-trim && make > /dev/null 2>&1)
             timeout $4 tools/cadical/build/cadical -q "$cnf" /tmp/$n.drat > logs/tiers/$n.cadical.out 2>&1; rc=$?
             [ $rc = 20 ] && tools/drat-trim/drat-trim "$cnf" /tmp/$n.drat -t 200000 > logs/tiers/$n.drattrim.out 2>&1
             pv=" proof $(du -h /tmp/$n.drat 2>/dev/null | cut -f1), drat-trim $(grep -o 's VERIFIED\|s NOT VERIFIED' logs/tiers/$n.drattrim.out 2>/dev/null | head -1), cnf sha $(sha256sum "$cnf" | cut -c1-16)"; rm -f /tmp/$n.drat
           else timeout $4 tools/cadical/build/cadical -q "$cnf" > logs/tiers/$n.cadical.out 2>&1; rc=$?; pv=""; fi
           if [ $rc = 20 ]; then echo "TIER $n: UNSAT$pv ($(head -n 1 logs/tiers/$n.encode.out | cut -c1-60))" >> logs/cloud_worker.log
           elif [ $rc = 10 ]; then python3 scripts/c1_tiers/decode_model.py scripts/c1_tiers/data/$2 $3 logs/tiers/$n.cadical.out "$cnf" logs/tiers/${n}_satsol.json > logs/tiers/$n.decode.out 2>&1
             echo "TIER $n: SAT -> logs/tiers/${n}_satsol.json ($(cat logs/tiers/$n.decode.out | tail -n 1))" >> logs/cloud_worker.log
           else echo "TIER $n: UNKNOWN (cadical rc $rc after $4 s)" >> logs/cloud_worker.log; fi
           gzip -f logs/tiers/$n.cadical.out; true ;;
      tier2) mkdir -p logs/tiers tools && t=${2%.json} && n=${t}_$(printf %04d $3)_enc2 && cd_=/tmp/tiers_cnf/$t && cnf=$cd_/type_$(printf %04d $3)_enc2.cnf
           [ -x tools/cadical/build/cadical ] || (cd tools && rm -rf cadical && git clone -q https://github.com/arminbiere/cadical.git && cd cadical && ./configure > /dev/null 2>&1 && make -j4 > /dev/null 2>&1)
           (cd scripts/c1_tiers && SAT_LOCAL_DIR=../structures python3 enc2_dimacs.py data/$2 $3 $cd_ > ../../logs/tiers/$n.encode.out 2>&1)
           if [ "$5" = proof ]; then
             [ -x tools/drat-trim/drat-trim ] || (cd tools && rm -rf drat-trim && git clone -q https://github.com/marijnheule/drat-trim.git && cd drat-trim && make > /dev/null 2>&1)
             timeout $4 tools/cadical/build/cadical -q "$cnf" /tmp/$n.drat > logs/tiers/$n.cadical.out 2>&1; rc=$?
             [ $rc = 20 ] && tools/drat-trim/drat-trim "$cnf" /tmp/$n.drat -t 200000 > logs/tiers/$n.drattrim.out 2>&1
             pv=" proof $(du -h /tmp/$n.drat 2>/dev/null | cut -f1), drat-trim $(grep -o 's VERIFIED\|s NOT VERIFIED' logs/tiers/$n.drattrim.out 2>/dev/null | head -1), cnf sha $(sha256sum "$cnf" | cut -c1-16)"; rm -f /tmp/$n.drat
           else timeout $4 tools/cadical/build/cadical -q "$cnf" > logs/tiers/$n.cadical.out 2>&1; rc=$?; pv=""; fi
           if [ $rc = 20 ]; then echo "TIER $n: UNSAT$pv ($(head -n 1 logs/tiers/$n.encode.out | cut -c1-60))" >> logs/cloud_worker.log
           elif [ $rc = 10 ]; then python3 scripts/c1_tiers/decode_model.py scripts/c1_tiers/data/$2 $3 logs/tiers/$n.cadical.out "$cnf" logs/tiers/${n}_satsol.json --enc2 > logs/tiers/$n.decode.out 2>&1
             echo "TIER $n: SAT -> logs/tiers/${n}_satsol.json ($(cat logs/tiers/$n.decode.out | tail -n 1))" >> logs/cloud_worker.log
           else echo "TIER $n: UNKNOWN (cadical rc $rc after $4 s)" >> logs/cloud_worker.log; fi
           gzip -f logs/tiers/$n.cadical.out; true ;;
      satp) mkdir -p logs/sat_local && b=${2%.json} && (cd scripts/structures && timeout $(( $3 + 60 )) python3 sat_local.py $2 $3 cadical195 > ../../logs/sat_local/$b.pysat.out 2>&1; [ $? = 124 ] && echo "status UNKNOWN (killed at the time limit)" >> ../../logs/sat_local/$b.pysat.out; cp ${b}_satsol.json ../../logs/sat_local/ 2>/dev/null; true) && echo "SATP $2: $(grep -h '^status' logs/sat_local/$b.pysat.out | tail -n 1 | cut -c1-100)$( [ -f logs/sat_local/${b}_satsol.json ] && echo ' LOCAL SOLUTION saved')" >> logs/cloud_worker.log ;;
      gp)  mkdir -p logs/c1_gplus && (cd scripts/c1_gplus && Q7_POINTWISE=1 Q7_ILP_TL=600 python3 q7_gplus_test.py $2 1000000 > ../../logs/c1_gplus/gplus_$2.out 2>&1; cp gplus_$2_results.json ../../logs/c1_gplus/ 2>/dev/null; true) && echo "GPLUS $2: $(grep -h 'g+ =' logs/c1_gplus/gplus_$2.out | tail -n 1 | cut -c1-120)" >> logs/cloud_worker.log ;;
      rt)  SMOOTH=0.6 LOWMEM=1 TL=6000 python3 scripts/root_job.py $2 $3 >> logs/cloud_worker.log 2>&1 ;;
      g18r) mkdir -p logs/c1_g18_regimes && (cd scripts/c1 && Q7_POINTWISE=1 python3 q7_g18_regimes.py $2 $3 $4 100000 >> ../../logs/cloud_worker.log 2>&1 && mv g18_regimes_*.json ../../logs/c1_g18_regimes/) ;;
    esac && touch "logs/cloud_done/$key"
  }
  export -f one
  grep -E '^(gen|adr) ' "$T" | xargs -P "$P" -I{} bash -c 'one "{}"'
  grep -v -E '^(gen|adr) ' "$T" | grep -v '^$' | xargs -P "$P" -I{} bash -c 'one "{}"'
  echo "WORKER DONE $T" >> logs/cloud_worker.log ;;
status)
  T=$1; tot=$(grep -c . "$T"); done=0; while read -r l; do [ -n "$l" ] && [ -f "logs/cloud_done/$(echo "$l" | tr ' /' '__')" ] && done=$((done+1)); done < "$T"
  echo "$done/$tot $(grep -h -E 'TIER|SATP|GPLUS|ROOT|GEN|RUN|ORIENT|UFIX|CHUNK|alive classes|FB |ADR|ADT|G18C|G18T1|SAT |INFEASIBLE|FEASIBLE|WORKER DONE|Error|Traceback' logs/cloud_worker.log 2>/dev/null | tail -n 1 | cut -c1-150)" ;;
push)
  B=$1; M=${2:-cloud results}
  for p in logs/tiers logs/c1_gplus logs/cloud_stage1 logs/root logs/orbit_certs/root_*.json.gz logs/c1_g18 logs/c1_g18c logs/c1_g18t1 logs/c1_g18_regimes logs/fullblock logs/sat_local logs/ad logs/run_U_*.out logs/orbit_certs/adres_*.json.gz logs/cloud_done logs/cloud_worker.log logs/orient_*.jsonl logs/run_U_fix_*.jsonl logs/orbit_certs/ogen_*.json.gz logs/orbit_certs/*fix*.json.gz; do git add -f "$p" 2>/dev/null; done   # one by one: git add adds nothing if any pathspec is missing
  git -c user.name="amazing-source" -c user.email="206353265+amazing-source@users.noreply.github.com" commit -q -m "$M" && git push -q origin HEAD:"$B" 2>&1 | tail -2; echo "PUSHED $B $(git rev-parse --short HEAD)" ;;
esac
