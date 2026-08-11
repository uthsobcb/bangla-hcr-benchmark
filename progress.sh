#!/bin/zsh
# Status of every training run. `./progress.sh watch` refreshes every 60s.
cd ${0:A:h}

show() {
  print "=== $(date '+%H:%M:%S') ==="
  for f in logs/*.log(N); do
    [[ ${f:t} == (run_*|dashboard.log) ]] && continue
    line=$(grep -E "Epoch \[|Test Accuracy" $f 2>/dev/null | tail -1)
    printf "%-16s %s\n" "${f:t:r}" "${line:-(starting)}"
  done
  # ponytail: pgrep, not a stored pid — survives me losing track of the shell that launched it
  pgrep -qf train_arch.py && print "\n[running]" || print "\n[idle — all runs finished or died]"
}

if [[ $1 == watch ]]; then
  while :; do clear; show; sleep 60; done
else
  show
fi
