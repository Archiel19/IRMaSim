# TODO input error handling
PHASE=$1
OPT=""

if [ $PHASE == "train" ]
then
  rm -f policy_agent.model energy_agent.model action_agent.model
  OPT="-l"
fi
rm -f {low_power,high_gflops,policy,action,energy}/*

printf 'Low power heuristic\n'
irmasim options_low_power.json
printf '\nHigh gflops heuristic\n'
irmasim options_high_gflops.json
printf '\nPolicy agent\n'
irmasim -nr 10 -im policy_agent.model -om policy_agent.model options_policy.json --phase $PHASE
printf '\nActionActorCritic agent\n'
irmasim -nr 10 -im action_agent.model -om action_agent.model options_action.json --phase $PHASE
printf '\nEnergy agent\n'
irmasim -nr 10 -im energy_agent.model -om energy_agent.model options_energy.json --phase $PHASE

python3.9 plotter.py -d energy -r $OPT
python3.9 plotter.py -d policy -r $OPT
python3.9 plotter.py -d action -r $OPT
