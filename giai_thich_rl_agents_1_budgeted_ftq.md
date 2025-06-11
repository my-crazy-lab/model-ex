# 🎯 Giải Thích Chi Tiết: Budgeted FTQ (Budgeted Fitted Q-Iteration)

## 📚 Tổng Quan Thuật Toán

**Budgeted FTQ** là một thuật toán Reinforcement Learning nâng cao được thiết kế để giải quyết **Budgeted Markov Decision Processes (BMDPs)**. Đây là extension của Fitted Q-Iteration để handle các bài toán có **resource constraints** (ràng buộc tài nguyên).

---

## 🎯 Vấn Đề Cần Giải Quyết

### Traditional MDP vs Budgeted MDP

**Traditional MDP:**
```
State → Action → Reward + Next State
Goal: Maximize cumulative reward
```

**Budgeted MDP:**
```
State → Action → Reward + Cost + Next State
Goal: Maximize reward while staying within budget constraint
Constraint: Σ costs ≤ Budget
```

### Ví Dụ Thực Tế:
- **Autonomous Driving**: Maximize safety while minimizing fuel consumption
- **Resource Management**: Maximize profit while staying within budget
- **Energy Systems**: Maximize performance while limiting energy usage

---

## 🏗️ Kiến Trúc Budgeted FTQ

### Core Components:

```python
class BFTQAgent(AbstractAgent):
    def __init__(self, env, config=None):
        # 1. Budgeted Fitted Q-Learning Algorithm
        self.bftq = BudgetedFittedQ(...)
        
        # 2. Exploration Policy với budget awareness
        self.exploration_policy = EpsilonGreedyBudgetedPolicy(...)
        
        # 3. Budget tracking
        self.beta = 0  # Current budget
        
        # 4. Neural Network cho Q-value và Cost estimation
        network = BudgetedMLP(...)
```

---

## 📖 Phân Tích Từng Dòng Code

### Dòng 16-29: Class Initialization
```python
class BFTQAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super(BFTQAgent, self).__init__(config)
        self.batched = True  # Supports batch processing
        if not self.config["epochs"]:
            # Auto-calculate epochs based on discount factor
            self.config["epochs"] = int(1 / np.log(1 / self.config["gamma"]))
        self.env = env
        self.bftq = None  # Will be initialized in reset()
        self.exploration_policy = None
        self.beta = self.previous_beta = 0  # Budget tracking
        self.training = True
        self.previous_state = None
```

**Giải thích:**
- `batched = True`: Agent có thể xử lý multiple experiences cùng lúc
- `epochs`: Số iterations cho fitted Q-iteration, auto-calculated từ gamma
- `beta`: Current budget available cho agent
- `training`: Flag để switch giữa training và evaluation mode

### Dòng 33-80: Default Configuration
```python
@classmethod
def default_config(cls):
    return {
        "gamma": 0.9,        # Discount factor cho rewards
        "gamma_c": 0.9,      # Discount factor cho costs
        "epochs": None,      # Auto-calculated
        "delta_stop": 0.,    # Stopping criterion
        "memory_capacity": 10000,  # Experience replay buffer size
        "beta": 0,           # Default budget
        "betas_for_duplication": "np.arange(0, 1, 0.1)",    # Budget discretization
        "betas_for_discretisation": "np.arange(0, 1, 0.1)",
        "exploration": {
            "temperature": 1.0,        # Initial exploration temperature
            "final_temperature": 0.1,  # Final exploration temperature
            "tau": 5000               # Temperature decay rate
        },
        # ... network architecture config
    }
```

**Key Parameters:**
- **gamma vs gamma_c**: Separate discount factors cho rewards và costs
- **betas_for_discretisation**: Discretize budget space cho tractable computation
- **temperature**: Controls exploration-exploitation trade-off

### Dòng 82-92: Action Selection
```python
def act(self, state):
    """
    Run the exploration policy to pick actions and budgets
    """
    # Choose budget: random during training, fixed during evaluation
    self.beta = self.np_random.uniform() if self.training else self.config["beta"]
    
    state = state.flatten()
    self.previous_state, self.previous_beta = state, self.beta
    
    # Execute policy với current state và budget
    action, self.beta = self.exploration_policy.execute(state, self.beta)
    return action
```

**Giải thích:**
- **Budget Selection**: Random budget during training để explore different budget levels
- **Policy Execution**: Policy takes both state và budget làm input
- **Budget Update**: Policy có thể modify budget (budget allocation)

### Dòng 94-103: Experience Recording
```python
def record(self, state, action, reward, next_state, done, info):
    """
    Record a transition to update the BFTQ policy
    """
    if not self.training:
        return
    
    # Store transition với cost information
    self.bftq.push(state.flatten(), action, reward, 
                   next_state.flatten(), done, info["cost"])
```

**Giải thích:**
- **Cost Tracking**: Record cả reward và cost từ environment
- **Experience Replay**: Store transitions cho batch learning
- **info["cost"]**: Environment phải provide cost information

### Dòng 105-113: Model Update
```python
def update(self):
    """
    Fit a budgeted policy on the batch by running the BFTQ algorithm.
    """
    # Reset và run BFTQ algorithm
    self.bftq.reset()
    network = self.bftq.run()
    
    # Update greedy policy với new network
    self.exploration_policy.pi_greedy.set_network(network)
```

**Giải thích:**
- **BFTQ Algorithm**: Core algorithm để learn Q-values và cost estimates
- **Network Update**: Update policy network với learned parameters
- **Batch Learning**: Update sau khi collect đủ experience

### Dòng 115-134: Agent Reset
```python
def reset(self):
    if not self.np_random:
        self.seed()
    
    # Initialize neural network
    network = BudgetedMLP(size_state=np.prod(self.env.observation_space.shape),
                          n_actions=self.env.action_space.n,
                          **self.config["network"])
    
    # Initialize BFTQ algorithm
    self.bftq = BudgetedFittedQ(value_network=network, 
                                config=self.config, 
                                writer=self.writer)
    
    # Initialize exploration policy
    self.exploration_policy = EpsilonGreedyBudgetedPolicy(
        pi_greedy=PytorchBudgetedFittedPolicy(...),
        pi_random=RandomBudgetedPolicy(...),
        config=self.config["exploration"],
        np_random=self.np_random
    )
```

**Components:**
- **BudgetedMLP**: Neural network architecture cho Q-value và cost prediction
- **BudgetedFittedQ**: Core algorithm implementation
- **EpsilonGreedyBudgetedPolicy**: Exploration strategy với budget awareness

---

## 🧠 Budgeted FTQ Algorithm Deep Dive

### Mathematical Foundation:

**Budgeted Q-Function:**
```
Q^π(s, β, a) = E[Σ γ^t r_t | s_0=s, β_0=β, a_0=a, π]
C^π(s, β, a) = E[Σ γ_c^t c_t | s_0=s, β_0=β, a_0=a, π]

Subject to: C^π(s, β, a) ≤ β
```

**Bellman Equations:**
```
Q*(s, β, a) = r + γ max_{a'} Q*(s', β', a')
C*(s, β, a) = c + γ_c max_{a'} C*(s', β', a')

Where: β' = β - c (remaining budget)
```

### Algorithm Steps:

1. **Experience Collection**: Collect (s, a, r, c, s') tuples
2. **Budget Discretization**: Discretize budget space
3. **Q-Value Fitting**: Fit Q(s, β, a) và C(s, β, a) networks
4. **Policy Extraction**: Extract optimal policy từ Q-values
5. **Convex Hull**: Compute Pareto frontier cho reward-cost trade-offs

---

## 🔄 Training Workflow

### Complete Training Process:

```
1. Environment Interaction
   State s, Budget β → Policy → Action a
                    ↓
   Environment → Reward r, Cost c, Next State s'
                    ↓
2. Experience Storage
   Store (s, β, a, r, c, s') in replay buffer
                    ↓
3. Batch Learning (when buffer full)
   Sample batch → BFTQ Algorithm → Update Q-networks
                    ↓
4. Policy Update
   New Q-networks → Extract Policy → Update Exploration Policy
                    ↓
5. Repeat until convergence
```

---

## 🎯 Key Features của Budgeted FTQ

### 1. **Budget-Aware Learning**
```python
# Policy considers both state và budget
action = policy(state, budget)

# Q-function depends on available budget
q_value = Q(state, budget, action)
cost_estimate = C(state, budget, action)
```

### 2. **Pareto Optimality**
```python
# Find Pareto frontier của reward-cost trade-offs
pareto_policies = compute_convex_hull(q_values, cost_estimates)

# Select policy based on budget constraint
optimal_policy = select_policy(pareto_policies, budget_constraint)
```

### 3. **Dual Discount Factors**
```python
# Separate discounting cho rewards và costs
discounted_reward = gamma ** t * reward
discounted_cost = gamma_c ** t * cost

# Allows different time preferences
```

---

## 💡 Ưu Điểm và Nhược Điểm

### ✅ **Ưu Điểm:**
- **Resource Awareness**: Explicitly handles budget constraints
- **Pareto Optimality**: Finds optimal reward-cost trade-offs
- **Flexible Budgeting**: Supports dynamic budget allocation
- **Theoretical Foundation**: Well-grounded mathematical framework

### ❌ **Nhược Điểm:**
- **Computational Complexity**: Higher than standard Q-learning
- **Budget Discretization**: May lose precision với continuous budgets
- **Environment Requirements**: Needs cost information from environment
- **Hyperparameter Sensitivity**: Many parameters to tune

---

## 🚀 Use Cases

### **Ideal Applications:**
1. **Autonomous Systems**: Energy-constrained robots
2. **Financial Trading**: Risk-constrained portfolio management
3. **Resource Management**: Cloud computing resource allocation
4. **Healthcare**: Treatment planning với cost constraints
5. **Smart Grids**: Energy distribution với budget limits

### **Environment Requirements:**
```python
# Environment must provide cost information
def step(self, action):
    # ... environment logic ...
    info = {"cost": computed_cost}
    return next_state, reward, done, info
```

---

## 🔧 Configuration Tips

### **Key Hyperparameters:**
```python
config = {
    "gamma": 0.9,           # Higher = more future-focused
    "gamma_c": 0.9,         # Can be different from gamma
    "beta": 0.5,            # Target budget level
    "betas_for_discretisation": "np.linspace(0, 1, 21)",  # Finer discretization
    "exploration": {
        "temperature": 1.0,  # Higher = more exploration
        "tau": 5000         # Slower decay = longer exploration
    },
    "network": {
        "layers": [128, 128, 64],  # Larger networks cho complex problems
    }
}
```

### **Performance Tuning:**
- **Budget Discretization**: More points = better precision, higher computation
- **Network Size**: Larger networks cho complex state spaces
- **Exploration**: Longer exploration cho complex environments
- **Memory Capacity**: Larger buffer cho better sample efficiency

**Next Algorithm**: Chúng ta sẽ tiếp tục với **Common Components** - các building blocks được share across multiple RL algorithms!
