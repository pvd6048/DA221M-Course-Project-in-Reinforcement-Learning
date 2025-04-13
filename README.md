## DA221M Course Project

### Description
------------
Reimplementation of [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf) and a deterministic variant of SAC from [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf).


### Setup
Clone this GitHub repo to a location of your choice, setup a virtual environment, preferrably with the latest Python interpreter. Follow the next steps.

### Requirements
Use the below command to install all the required libraries and packages to run the Python scripts.
```
python install -r requirements.txt
```

### Running the Code
For the purpose of ablation studies we had run the scripts in the Humanoid-v4 environment using the alphas- 0.05, 0.04 and 0.02, apart from this, automatic entropy tuning was also done.

#### SAC with different valeus of alpha
```
python main.py --env-name Humanoid-v2 --alpha 0.05
```
```
python main.py --env-name Humanoid-v2 --alpha 0.04
```
```
python main.py --env-name Humanoid-v2 --alpha 0.02
```

#### SAC with automatic entropy tuning
```
python main.py --env-name Humanoid-v2 --automatic_entropy_tuning True
```

### Environment
For the purposes of this paper we have used the Humanoid-v4 environment only.

