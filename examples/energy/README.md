# Energy Scheduler Tests

This directory includes a series of experiments designed to compare the performance of the Energy scheduler with
that of heuristics when it comes to reducing energy consumption or EDP.

More specifically, the Energy scheduler is compared against the low_power, high_gflops and high_cores heuristics and 
the Policy scheduler is employed to find the most suitable heuristic for each experiment.
* __low_power__: assigns the next job in the queue to the processor with the lowest power consumption.
* __high_gflops__: assigns the next job in the queue to the processor with the highest peak compute capability.
* __high_cores__: assigns the next job in the queue to the processor with the most currently available cores.
* __policy__: RL-based agent that attempts to select the best policy for the given trace. The policy is itself comprised
by both a job selection policy and a node selection policy.
* __energy__: similar to the former, but focused on energy consumption by altering the observation, reward function and some aspects of the learning algorithm.
---
### Script
The script `execute_test.sh` runs the example contained in a given directory (e.g. "save"). It accepts the following
arguments:

```
./execute_test.sh -d <example> [-p <test|train>] [-e] [-i <iters>] [-h]
```
* __-d \<example>__: Mandatory. Selects the example which will be executed. All resulting plots and log files
will be stored in subdirectories inside the specified example directory.
* __-p \<train|test>__: Optional. Specifies whether to train the agents or test the existing (previously trained)
models.
* __-e__: Optional. Enables the sole execution of the energy agent.
* __-i \<iters>__: Optional. Number of training simulations.
* __-h__: Displays usage information.

---
### Energy Scheduler Options

Option files for each agent and heuristic can be found in the current directory.
`options_energy.json` can be used to fine-tune the training of the Energy scheduler.

The training process of the agent during a single simulation follows two main steps:

1. __Training data is collected__ in a buffer by scheduling `trajectory_length` jobs from the given workload for a total of
`nbtrajectories` times, using a randomly initialized (or existing) agent. To be precise, a set of values is stored for
each *time step*: each time a job is scheduled on a node. 
Consequently, after all the jobs have been scheduled, a total of `trajectory_length * nbtrajectories` samples have been
collected.

2. After some pre-processing of the training data, actor and critic __parameters are optimized__ 
for `train_iters` epochs, the same number for both actor and critic. The actor's loss function includes an entropy factor.
During each epoch, all the samples are iterated over after being partitioned into minibatches of size `minibatch_size`.

A brief and informal description of the rest of the training hyperparameters can be found below:

* `h_factor`: weight of the policy's entropy in the actor's loss function. May be used to encourage exploration.
* `lr_pi`: learning rate for the policy (actor) optimizer.
* `lr_v`: learning rate for the value (critic) optimizer.

The following three hyperparameters are related to the actor's PPO loss function:
* `clipping_factor`: epsilon. Constrains how much a policy can change from one step to the next.
* `gamma`: discount factor, indicates the decrease in the weight given to future rewards. Rewards far in the future should have less influence
in the policy than short-term rewards.
* `lambda`: used in the Generalized Advantage Estimate equation, together with gamma. Can be seen as a bias-variance knob.
A lambda close to 1 reduces variance and increases bias and viceversa.
