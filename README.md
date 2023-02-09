# PID-Accelerated-TD-Learning
An application of ideas from control theory to hopefully accelerate the dynamics of TD learning.

This builds on the work of Farahmand and Ghavamzadeh [1] in an RL setting.

[1] A.M. Farahmand and Mohammad Ghavamzadeh, “PID Accelerated Value Iteration Algorithm,” International Conference on Machine Learning (ICML), 2021. 

# Implementation TODOs for now:
- [x] Implement various controllers for Value Iteration in a planning setting. This includes P, PI, PD, and PID.
- [x] Write experiments on simple environments, reproducing results from the PAVIA paper.
- [ ] Implement a base for the TD-PID Accelerated Algorithm