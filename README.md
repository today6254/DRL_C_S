# Efficient Crowd Simulation in Complex Environment Using Deep Reinforcement Learning

## Project Introduction

Simulating virtual crowds can bring significant economic benefits to various applications, such as film and television special effects, evacuation planning, and rescue operations. However, the key challenge in crowd simulation is ensuring efficient and reliable autonomous navigation for numerous agents within virtual environments. In recent years, deep reinforcement learning has been used to model agents' steering strategies, including marching and obstacle avoidance. However, most studies have focused on simple, homogeneous scenarios (e.g., intersections, corridors with basic obstacles), making it difficult to generalize the results to more complex settings. In this study, we introduce a new crowd simulation approach that combines deep reinforcement learning with anisotropic fields. This method gives agents global prior knowledge of the high complexity of their environment, allowing them to achieve impressive motion navigation results in complex scenarios without the need to repeatedly compute global path information.

## User Guide
To run the training, fine-tuning, and testing process and save the model, just run 
```
cd src
python main.py
```
The result will be saved in the `saved_model` folder.
### Requirements
```
python            3.9.12
matplotlib        3.7.2
numpy             1.25.1
torch             1.12.0
Gymnasium         0.26.3
pygame            2.5.2
```

## Examples & Visualizations

