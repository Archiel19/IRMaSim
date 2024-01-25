import numpy.random as rand
import math
import argparse

OUT = "random_workload.json"
N = 256
TIME_INT_ALPHA = 1.4
TIME_INT_BETA = 7.0
MAX_J_INTERVAL = 20
MIN_NTASKS = 8
MAX_NTASKS = 16
MIN_REQ_OPS = 1e2
MAX_REQ_OPS = 1e3  # Significant digits in this range
REQ_OPS_MAGN = 1e8  # Appended zeroes
MIN_FREQ = 3.0  # Minimum clock rate in the platform
MAX_MEM_VOL_EXP = 9
MAX_POWER = math.inf

parser = argparse.ArgumentParser(prog='Workload Generator',
                                 description='Generate a random workload trace.')
parser.add_argument('-o', '--out', default=OUT,
                    help='Output file name')
parser.add_argument('-n', '--njobs', type=int, default=N,
                    help='Number of jobs to generate')
parser.add_argument('--max_interval', type=int, default=MAX_J_INTERVAL,
                    help='Maximum time interval between consecutive job submissions, in ms')
parser.add_argument('--min_ntasks', type=int, default=MIN_NTASKS,
                    help='Minimum number of tasks per job, should be less than or equal to the maximum '
                         'number of cores per node in the platform')
parser.add_argument('-t', '--max_ntasks', type=int, default=MAX_NTASKS,
                    help='Maximum number of tasks per job, should be less than or equal to the maximum '
                         'number of cores per node in the platform')
parser.add_argument('--min_req_ops', type=int, default=MIN_REQ_OPS,
                    help='Minimum number of significant digits for job req_ops (in exponential notation)')
parser.add_argument('--max_req_ops', type=int, default=MAX_REQ_OPS,
                    help='Maximum number of significant digits for job req_ops (in exponential notation)')
parser.add_argument('--req_ops_magn', type=int, default=REQ_OPS_MAGN,
                    help='Number of non-significant digits for job req_ops (in exponential notation)')
parser.add_argument('--min_freq', type=int, default=MIN_FREQ,
                    help='Frequency of the slowest node in the platform (in GHz)')
parser.add_argument('--max_mem_vol_exp', type=int, default=MAX_MEM_VOL_EXP,
                    help='Maximum exponent for memory contention volume (e.g. 6 -> memory contention = 1e6)')
parser.add_argument('--max_power', type=float, default=MAX_POWER,
                    help='Maximum power consumed during the execution of each job')

args = parser.parse_args()

rng = rand.default_rng()
print("Generating workload...")
with open(args.out, 'w') as f:
  f.write('{\n   "jobs": [\n')
  last_time = 0
  for i in range(args.njobs):
    ntasks = rng.integers(low=args.min_ntasks, high=args.max_ntasks + 1)
    req_ops = rng.integers(low=args.min_req_ops, high=args.max_req_ops + 1) * args.req_ops_magn
    req_time = round(req_ops / (MIN_FREQ * 1e9))
    mem_vol = rng.integers(low=0, high=args.max_mem_vol_exp + 1)
    max_energy = req_time * args.max_power
    f.write(f'      {{ "jobid": "job{i}", ')
    f.write(f'"subtime": {last_time}, ')
    f.write(f'"ntasks": {ntasks}, ')
    f.write(f'"req_time": {req_time}, ')
    f.write(f'"req_ops": {int(req_ops)}, ')  # Int
    f.write(f'"ipc": 1, ')
    f.write(f'"mem": 100, ')
    f.write(f'"mem_vol": 1e{mem_vol}}}')
    # f.write(f'"max_energy": {max_energy}}}')
    last_time = last_time + int(rng.beta(a=TIME_INT_ALPHA, b=TIME_INT_BETA) * args.max_interval) + 1
    if i == N - 1:
      f.write('\n')
    else:
      f.write(',\n')
  f.write('   ]\n}\n')

  print(f"Saved workload to {args.out}")
