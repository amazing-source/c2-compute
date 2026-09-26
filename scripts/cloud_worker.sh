#!/bin/bash
# Cloud worker (note 44): runs a task file on all cores with minimal console output (the agent only polls 'status').
#   setup                       pip install the numeric stack (quiet)
#   run <taskfile> [P]          phase 1: 'gen ci rep' lines; phase 2: 'run ci rep lo hi' lines (stage 1, cloud_stage1.py);
#                               also 'or ci idx' (orientation root LP) and 'fx json index prefix' (unified block fixing)
#   status <taskfile>           one line: done/total and the last result
#   push <branch> <message>     commit the result files and push them to <branch>
cd "$(dirname "$0")/.."
cmd=$1; shift
case "$cmd" in
setup)
  python3 -m pip install -q --disable-pip-version-check numpy scipy highspy networkx > /tmp/pip.log 2>&1 && echo SETUP OK || { echo SETUP FAILED; tail -3 /tmp/pip.log; } ;;
run)
  T=$1; P=${2:-$(nproc)}; mkdir -p logs/cloud_stage1 logs/cloud_done
  one() {
    set -- $1; key=$(echo "$*" | tr ' /' '__'); [ -f "logs/cloud_done/$key" ] && return
    case "$1" in
      gen) python3 scripts/cloud_stage1.py gen $2 $3 >> logs/cloud_worker.log 2>&1 ;;
      run) MAXBB=0 python3 scripts/cloud_stage1.py run $2 $3 $4 $5 >> logs/cloud_worker.log 2>&1 ;;
      or)  ONLY=$3 SMOOTH=0.6 LOWMEM=1 TL=20000 python3 scripts/orient_campaign.py $2 >> logs/cloud_worker.log 2>&1 ;;
      fx)  SMOOTH=0.6 LOWMEM=1 TL=20000 python3 scripts/run_U_fix.py $2 $3 auto $4 >> logs/cloud_worker.log 2>&1 ;;
    esac && touch "logs/cloud_done/$key"
  }
  export -f one
  grep '^gen ' "$T" | xargs -P "$P" -I{} bash -c 'one "{}"'
  grep -v '^gen ' "$T" | grep -v '^$' | xargs -P "$P" -I{} bash -c 'one "{}"'
  echo "WORKER DONE $T" >> logs/cloud_worker.log ;;
status)
  T=$1; tot=$(grep -c . "$T"); done=$(ls logs/cloud_done 2>/dev/null | wc -l)
  echo "$done/$tot $(grep -h -E 'GEN|RUN|ORIENT|UFIX|WORKER DONE|Error|Traceback' logs/cloud_worker.log 2>/dev/null | tail -n 1 | cut -c1-150)" ;;
push)
  B=$1; M=${2:-cloud results}
  git add -f logs/cloud_stage1 logs/cloud_done logs/cloud_worker.log logs/orient_*.jsonl logs/run_U_fix_*.jsonl logs/orbit_certs/ogen_*.json.gz logs/orbit_certs/*fix*.json.gz 2>/dev/null
  git -c user.name="amazing-source" -c user.email="206353265+amazing-source@users.noreply.github.com" commit -q -m "$M" && git push -q origin HEAD:"$B" 2>&1 | tail -2; echo "PUSHED $B $(git rev-parse --short HEAD)" ;;
esac
