PYTHON=python3.9
ITERS=10

# Defaults
energy_only="false"
phase="test"

usage() {
  printf "Usage: %+18s [-p <train|test>] [-e]\n" "$0" 1>&2
  printf "%+9sp: phase (default: test)\n%+9se: energy agent only (default: false)\n" "-" "-" 1>&2
  exit 1
}

while getopts "p:eh" opt; do
  case ${opt} in
  p)
    phase=$OPTARG
    if [ "$phase" != "train" ] && [ "$phase" != "test" ]; then
      printf "%s: invalid argument for option -- p\n" "$0" 1>&2
      usage
    fi
    ;;
  e)
    energy_only="true"
    ;;
  h | * )
    usage
    ;;
  esac
done

printf "Phase: %s\n\n" "$phase"

plot_opt="-r"
if [ "$phase" == "train" ]; then
  rm -f policy_agent.model energy_agent.model action_agent.model
  plot_opt=${plot_opt}" -l"
fi
rm -f {low_power,high_gflops,policy,action,energy}/*

if [ "$energy_only" = "true" ]; then
  echo "Using energy agent only"
  irmasim -nr ${ITERS} -im energy_agent.model -om energy_agent.model options_energy.json --phase "$phase"
  ${PYTHON} plotter.py -d energy ${plot_opt}
  exit 0
fi

printf 'Low power heuristic\n'
irmasim options_low_power.json
printf '\nHigh gflops heuristic\n'
irmasim options_high_gflops.json

AGENTS=("policy" "action" "energy")
for type in "${AGENTS[@]}"; do
  printf "\n%s agent\n" "$type"
  irmasim -nr ${ITERS} -im "$type"_agent.model -om "$type"_agent.model options_"$type".json --phase "$phase"
  ${PYTHON} plotter.py -d "$type" ${plot_opt}
done
