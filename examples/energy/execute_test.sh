rm -f agent.model test/*
mkdir -p test
irmasim -nr 30 -im agent.model -om agent.model options_test.json
python3.9 plotter.py -d test -r -l
