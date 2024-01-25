PYTHON=python3.9

# Defaults
energy_only="false"
phase="train"
dir=""
plot_opt="-r"
iters=10

# Input handling
usage() {
  printf "Usage: %+18s -d <directory> [-p <train|test>] [-e] [-i <iters>] [-h]\n" "$0" 1>&2
  printf "%+9sp: phase (default: %s)\n" "-" "$phase" 1>&2
  printf "%+9se: energy agent only (default: %s)\n" "-" "$energy_only" 1>&2
  printf "%+9si: number of simulations to run (default: %d)\n" "-" "$iters" 1>&2
  printf "%+9sh: display this message and exit\n" "-" 1>&2
  exit 1
}

while getopts "d:p:i:eh" opt; do
  case ${opt} in
  p)
    phase=$OPTARG
    if [ "$phase" != "train" ] && [ "$phase" != "test" ]; then
      printf "%s: invalid argument for option -- p\n" "$0" 1>&2
      usage
    fi
    ;;
  d)
    dir=$OPTARG
    ;;
  i)
    iters=$OPTARG
    ;;
  e)
    energy_only="true"
    ;;
  h | * )
    usage
    ;;
  esac
done

# dir argument is mandatory
: "${dir:?Missing -d}"

prev_dir=$(pwd)
cd "$dir" || exit 1

# Clean-up
printf "Phase: %s\n\n" "$phase"
if [ "$phase" == "train" ]; then
  rm -f policy_agent.model action_agent.model energy_agent.model
  plot_opt=${plot_opt}" -l"
fi
rm -f {energy,policy,action,low_power,high_gflops,high_cores}/*


# Execution
if [ "$energy_only" != "true" ]; then
  HEURISTICS=("high_gflops" "high_cores" "high_mem" "high_mem_bw" "random")
  AGENTS=("policy" "energy")
else
  HEURISTICS=()
  AGENTS=("energy")
fi

mkdir -p "${HEURISTICS[@]}"
mkdir -p "${AGENTS[@]}"

for type in "${HEURISTICS[@]}"; do
  printf "\n%s heuristic\n" "$type"
  irmasim "$prev_dir"/options_"$type".json
  ${PYTHON} "$prev_dir"/plotter.py -t "$type heuristic" -d "$type" ${plot_opt}
done

for type in "${AGENTS[@]}"; do
  printf "\n%s agent\n" "$type"
  irmasim -nr "$iters" -im "$type"_agent.model -om "$type"_agent.model options_"$type".json --phase "$phase"
  ${PYTHON} "$prev_dir"/plotter.py -t "$type agent" -d "$type" ${plot_opt}
done

cd "$prev_dir" || exit 1
