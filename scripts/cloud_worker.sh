#!/bin/bash
# Cloud worker (note 44): runs a task file on all cores with minimal console output (the agent only polls 'status').
#   setup                       pip install the numeric stack (quiet)
#   task types also: 'adr START END' / 'adt reps logs...' (structures' AD sweep; adr runs in phase 1), 'U json index prefix' (unified model), 'fb script cfg seconds workers' (structures' full-block CP-SAT; run the fb file with P=1), 'g18s START END' (C1's g-=18 t=0 split sweep, recorded) and 'g18r T START END' (C1's torsion regimes)
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
      sat) mkdir -p logs/sat_local && (cd scripts/structures && python3 sat_local.py $2 $3 > "../../logs/sat_local/${2%.json}.out" 2>&1; cp cfg_*satsol*.json ../../logs/sat_local/ 2>/dev/null; true) && echo "SAT $2: $(grep -h -i -E 'unsat|sat|unknown' "logs/sat_local/${2%.json}.out" | tail -n 1 | cut -c1-80)" >> logs/cloud_worker.log ;;
      g18r) mkdir -p logs/c1_g18_regimes && (cd scripts/c1 && Q7_POINTWISE=1 python3 q7_g18_regimes.py $2 $3 $4 100000 >> ../../logs/cloud_worker.log 2>&1 && mv g18_regimes_*.json ../../logs/c1_g18_regimes/) ;;
    esac && touch "logs/cloud_done/$key"
  }
  export -f one
  grep -E '^(gen|adr) ' "$T" | xargs -P "$P" -I{} bash -c 'one "{}"'
  grep -v -E '^(gen|adr) ' "$T" | grep -v '^$' | xargs -P "$P" -I{} bash -c 'one "{}"'
  echo "WORKER DONE $T" >> logs/cloud_worker.log ;;
status)
  T=$1; tot=$(grep -c . "$T"); done=$(ls logs/cloud_done 2>/dev/null | wc -l)
  echo "$done/$tot $(grep -h -E 'GEN|RUN|ORIENT|UFIX|CHUNK|alive classes|FB |ADR|ADT|G18C|SAT |INFEASIBLE|FEASIBLE|WORKER DONE|Error|Traceback' logs/cloud_worker.log 2>/dev/null | tail -n 1 | cut -c1-150)" ;;
push)
  B=$1; M=${2:-cloud results}
  git add -f logs/cloud_stage1 logs/c1_g18 logs/c1_g18c logs/c1_g18_regimes logs/fullblock logs/sat_local logs/ad logs/run_U_*.out logs/orbit_certs/adres_*.json.gz logs/cloud_done logs/cloud_worker.log logs/orient_*.jsonl logs/run_U_fix_*.jsonl logs/orbit_certs/ogen_*.json.gz logs/orbit_certs/*fix*.json.gz 2>/dev/null
  git -c user.name="amazing-source" -c user.email="206353265+amazing-source@users.noreply.github.com" commit -q -m "$M" && git push -q origin HEAD:"$B" 2>&1 | tail -2; echo "PUSHED $B $(git rev-parse --short HEAD)" ;;
esac
