#!/bin/bash
# pf_status.sh [tag-prefix]: one line a run of the FFN prefill ladder (default pfl_): state, device time
# and rate, and a stall flag. Time is the root chip's DM core trace (the host core writes nothing
# once the window opens, so its trace looks frozen in a healthy run). The time shown is the latest trace time of a sample
# of cores on four chips (not the root DM core, which parks). A run with a watchdog (common/sim_watchdog.py) gets its
# scheduler line below the table: task counters from the bingo_hw_manager monitor, the better progress indicator.
R=$(cd "$(dirname "$0")" && pwd)/runs
P=${1:-pfl_}
last_us() { [ -s "$1" ] && tail -c 3000 "$1" 2>/dev/null | grep -oE "^ *[0-9]+ " | tail -1 | awk '{printf "%d", $1/1000000}'; }
printf "%-18s %-9s %-9s %-8s %s\n" run state "dev us" "us/min" "last event"
for e in $R/${P}*.events; do
  t=$(basename "$e" .events); d=$(ls -d $R/$t/task_0* 2>/dev/null | head -1)
  last=$(tail -1 "$e" | cut -c7- | cut -c1-90)
  st=building; [ -n "$d" ] && [ -f "$d/bin/occamy_chip.vcs" ] && st=staged
  grep -q "SIM START" "$e" && st=running
  grep -q "EXITED" "$e" && st=exited
  grep -q "DONE" "$e" && st=done
  [ -e "$R/$t.PAUSED_DEADLOCK.txt" ] && st=PAUSED
  grep -qE "ASSERTION|CHECK FAIL|L1 OOM|GONE|BUILD ERROR" "$e" && st="$st!"
  us=-; rate=-; flag=""
  L=$d/bin/logs
  if [[ "$st" == running* ]] && [ -d "$L" ]; then
    # the system time: the latest trace time over a sample of cores (one core's trace goes quiet when it parks)
    us=0; for cs in 00:5 00:6 33:1 33:5 42:1 42:5 11:1 11:5; do x=$(last_us $L/trace_chip_${cs%%:*}_hart_0000${cs##*:}.dasm); [ -n "$x" ] && [ "$x" -gt "$us" ] && us=$x; done
    [ "$us" = 0 ] && us=-
    c=0; for h in 1 2 3 5 6 7; do x=$(last_us $L/trace_chip_00_hart_0000$h.dasm); [ -n "$x" ] && [ "$x" -gt "$c" ] && c=$x; done
    root=$(last_us $L/trace_chip_00_hart_00004.dasm)
    [ "$us" != - ] && [ -n "$root" ] && [ $((us - root)) -gt 300 ] && flag=" root-DM-parked ${root}us (a dependency wait, not by itself a hang)"
    first=$(grep -m1 "SIM START" "$e" | cut -c1-5)
    if [ -n "$first" ] && [ "$us" != - ]; then
      m=$(( ( $(date +%s) - $(date -d "$first" +%s) + 86400 ) % 86400 / 60 )); [ $m -gt 0 ] && rate=$(awk -v u=$us -v m=$m 'BEGIN{printf "%.1f", u/m}')
    fi
  fi
  w=""; [ -n "$d" ] && grep -q "SYNC\[hw\]" "$d/bin/uart_chip_0_0.log" 2>/dev/null && w=" [window over]"
  printf "%-18s %-9s %-9s %-8s %s%s%s\n" "$t" "$st" "$us" "$rate" "$last" "$w" "$flag"
done
q=$(quota -u fkong | tail -1 | awk '{print $1}' | tr -d '*')
echo "quota: $((q/1048576)) GB of 150 GB hard ($(( (157286400-q)/1048576 )) GB headroom; soft 120); sim cores busy: $(ps -u $USER -o pcpu=,stat=,args= | awk '/occamy_chip.vcs/ && $2 !~ /T/ {s+=$1} END {printf "%.0f", s/100}')"
