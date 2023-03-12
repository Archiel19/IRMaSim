# Energy agent tests

This directory includes a series of experiments designed to compare the performances
of a series of heuristics and intelligent agents when it comes to reducing energy consumption.

More specifically, the following are tested:
* __high_cores__: assigns the next job in the queue to the processor with the highest number of cores.
* __high_gflops__: analogous to the former, but the job is assigned to the processor with the highest gflops.
* __policy__: RL-based agent which attempts to select the best policy for scheduling the given trace. The policy is itself comprised
by both a job selection policy and a node selection policy.
* __action__: RL-based agent which attempts to choose the best job-node assignment at each scheduling
step, without being constrained by a specific policy.
* __energy__: similar to the former, but optimized for energy consumption minimization by altering
the observation, reward function and some aspects of the learning algorithm.
---
### Script
The script `execute_test.sh` runs the example contained in a given directory (e.g. "save"). It accepts the following
arguments:

```
./execute_test.sh -d <example> [-p <test|train>] [-e] [-h]
```
* __-d \<example>__: Mandatory. Selects the example which will be executed. All resulting plots and log files
will be stored in subdirectories inside the specified example directory.
* __-p \<train|test>__: Optional. Specifies whether to train the agents or test the existing (previously trained)
models.
* __-e__: Optional. Enables the sole execution of the energy agent.
* __-h__: Displays usage information.

---
### Energy agent options

Option files for each agent and heuristic can be found in the current directory.
We are particularly interested in `options_energy.json`, which will enable us to fine-tune our energy-focused agent.

But first, a brief introduction concerning the agent's training process during a single simulation shall be given.

1. __Training data is collected__ in a buffer by scheduling `trajectory_length` jobs from the given workload for a total of
`nbtrajectories` times, using a randomly initialized (or existing) agent. To be precise, a set of values is stored for
each *time step*: each time a job is scheduled on a node. 
Consequently, after all the jobs have been scheduled, a total of `trajectory_length * nbtrajectories` samples have been
collected.

2. After some pre-processing of the training data, actor and critic __parameters are optimized__ 
for `train_iters` epochs using a common loss function created by combining both actor and critic losses plus entropy.
During each epoch, all the samples are iterated over after being partitioned into minibatches of size `minibatch_size`.

A brief and informal description of the rest of the training hyperparameters can be found below:

* `max_kl`: maximum Kullback-Leibler divergence between the current policy and the updated policy. Used to cut off training
if the policy's probability distribution starts to change too much. (Added because it seemed to help in training)
* `val_factor`: weight of the critic's loss in the joint loss function.
* `h_factor`: weight of the policy's entropy in the joint loss function. Encourages exploration.
* `lr_pi`: learning rate for the policy (actor) optimizer
* `lr_v`: learning rate for the value (critic) optimizer

The following three hyperparameters are related to the actor's loss function, which follows the PPO algorithm's specification:
* `clipping_factor`: constrains how much a policy can change from one step to the next.
* `gamma`: discount factor, indicates the value given to future rewards. Rewards far in the future should have less influence
in the policy than short-term rewards. From a theoretical standpoint, a discount factor is needed to turn an infinite horizon sum
into a finite one (to explain it badly).
* `lambda`: used in the Generalized Advantage Estimate equation, together with gamma. Can be seen as a bias-variance knob.
A lambda close to 1 reduces variance and increases bias and viceversa.

It is important to notice that this is still a work in progress, so certain aspects of the design might not make
much sense.