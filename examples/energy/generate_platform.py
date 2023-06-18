#!/usr/bin/python3

import json
import sys

platform = {
    "platform": {
        "the_platform": {
            "id": "the_platform",
            "model_name": "modelV1",
            "clusters": [
                {"id": "cluster0", "nodes": []},
            ],
        }
    },
    "node": {},
    "processor": {},
}

processors = [
    {
        "cores": 8,
        "clock_rate": 4.2,
        "b": -1.85e-05,
        "c": 32000,
        "da": 1.00,
        "db": 500,
        "dc": 25000,
        "dd": 500,
    },
    {
        "cores": 12,
        "clock_rate": 4.0,
        "b": -1.85e-05,
        "c": 32000,
        "da": 1.00,
        "db": 200,
        "dc": 25000,
        "dd": 400,
    },
    {
        "cores": 16,
        "clock_rate": 3.8,
        "b": -1.85e-05,
        "c": 21000,
        "da": 1.00,
        "db": 169,
        "dc": 25000,
        "dd": 100,
    },
    {
        "cores": 24,
        "clock_rate": 3.6,
        "b": -1.85e-05,
        "c": 21000,
        "da": 1.00,
        "db": 171,
        "dc": 25000,
        "dd": 95,
    },
    {
        "cores": 32,
        "clock_rate": 3.4,
        "b": -1.85e-05,
        "c": 21000,
        "da": 1.00,
        "db": 150,
        "dc": 25000,
        "dd": 20,
    },
    {
        "cores": 48,
        "clock_rate": 3.0,
        "b": -1.85e-05,
        "c": 21000,
        "da": 1.00,
        "db": 100,
        "dc": 25000,
        "dd": 10,
    },
]

for p in processors:
    freq = p["clock_rate"]
    p["dynamic_power"] = 3.313 * (freq / 3.0) ** 2
    p["static_power"] = 35.108 * (freq / 3.0) ** 2

total_cores = 0

if len(sys.argv) < 2:
    print(
        "Error: must supply at least one node specification."
        + "\n\nA node specified with three numbers separated by commas."
        + "The first is the number of nodes, then comes the number of processors and "
        + "finally the number of cores. For example: \n\n"
        + "10,2,48 creates 10 nodes with 2 processors of 48 cores"
    )

nodes = []  # Tamanho num args * 3 (node, proc, core)

for param in sys.argv[1:]:
    nodes.append([int(x) for x in param.split(",")])

for node_count, processor_count, core_count in nodes:
    print(processors[0]["cores"])
    proc = [x for x in processors if x["cores"] == core_count]
    print(proc)
    if proc == []:
        print("Error, invalid number of 'cores'", core_count)
        sys.exit(1)

    freq = proc[0]["clock_rate"]
    total_cores += core_count * processor_count * node_count

    platform["platform"]["the_platform"]["clusters"][0]["nodes"].append(
        {"type": "node_" + str(core_count) + "_" + str(freq), "number": node_count}
    )

    platform["node"]["node_" + str(core_count) + "_" + str(freq)] = {
        "id": "node_" + str(core_count) + "_" + str(freq),
        "processors": [
            {
                "type": "proc_" + str(core_count) + "_" + str(freq),
                "count": int(processor_count),
            }
        ],
    }

    platform["processor"]["proc_" + str(core_count) + "_" + str(freq)] = {
        "id": "proc_" + str(core_count) + "_" + str(freq),
        "cores": proc[0]["cores"],
        "clock_rate": proc[0]["clock_rate"],
        "dpflops_per_cycle": 1,
        "dynamic_power": proc[0]["dynamic_power"],
        "static_power": proc[0]["static_power"],
        "min_power": 0.3959,
        "b": proc[0]["b"],
        "c": proc[0]["c"],
        "da": proc[0]["da"],
        "db": proc[0]["db"],
        "dc": proc[0]["dc"],
        "dd": proc[0]["dd"],
    }

with open("platform.json", "w") as f:
    json.dump(platform, f, indent=2)

print("Total 'cores': ", total_cores)