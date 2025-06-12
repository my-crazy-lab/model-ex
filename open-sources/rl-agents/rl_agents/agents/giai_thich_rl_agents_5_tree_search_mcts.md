# 🌳 Giải Thích Chi Tiết: Monte Carlo Tree Search (MCTS) - Advanced Planning Algorithm

## 📚 Tổng Quan Monte Carlo Tree Search

**Monte Carlo Tree Search (MCTS)** là thuật toán planning mạnh mẽ kết hợp **tree search** với **Monte Carlo simulation**. MCTS đã đạt breakthrough trong game AI (AlphaGo, AlphaZero) và được ứng dụng rộng rãi trong decision making under uncertainty.

---

## 🎯 Vấn Đề Cần Giải Quyết

### Traditional Planning Limitations:

**Minimax/Alpha-Beta:**
```
Problem: Exponential search space
Solution: Pruning, but still limited depth
Limitation: Need evaluation function
```

**Dynamic Programming:**
```
Problem: Need complete model
Solution: Exact solution
Limitation: Curse of dimensionality
```

### MCTS Advantages:

**Selective Search:**
```
Focus computational resources on promising paths
Asymmetric tree expansion
Balance exploration vs exploitation
```

**Model-Based but Flexible:**
```
Use environment model for simulation
No need for evaluation function
Learn value through rollouts
```

---

## 🏗️ Kiến Trúc MCTS

### Core Components:

```python
class MCTSAgent(AbstractTreeSearchAgent):
    def __init__(self, env, config=None):
        # 1. Environment Model (for simulation)
        self.env = env
        
        # 2. Prior Policy (for expansion)
        self.prior_policy = policy_factory(config["prior_policy"])
        
        # 3. Rollout Policy (for evaluation)
        self.rollout_policy = policy_factory(config["rollout_policy"])
        
        # 4. MCTS Planner
        self.planner = MCTS(env, prior_policy, rollout_policy, config)
```

---

## 🔄 MCTS Algorithm: 4 Phases

### **Phase 1: Selection**
```python
def selection(self, node):
    """Navigate from root to leaf using UCB1"""
    while node.children and not node.is_terminal():
        # UCB1 formula: value + exploration_bonus
        action = node.selection_strategy(temperature=self.config['temperature'])
        node = node.get_child(action)
    return node
```

### **Phase 2: Expansion**
```python
def expansion(self, node, state, observation):
    """Expand leaf node by adding children"""
    if not node.is_terminal():
        # Get available actions from prior policy
        actions, probabilities = self.prior_policy(state, observation)
        node.expand(actions, probabilities)
```

### **Phase 3: Simulation (Rollout)**
```python
def simulation(self, state, observation):
    """Simulate random trajectory to estimate value"""
    total_reward = 0
    for depth in range(self.config["horizon"]):
        # Sample action from rollout policy
        actions, probs = self.rollout_policy(state, observation)
        action = np.random.choice(actions, p=probs)
        
        # Step environment
        observation, reward, terminal, _, _ = self.step(state, action)
        total_reward += (self.config["gamma"] ** depth) * reward
        
        if terminal:
            break
    
    return total_reward
```

### **Phase 4: Backpropagation**
```python
def backpropagation(self, node, value):
    """Update all nodes in path with simulation result"""
    while node is not None:
        node.update(value)  # Update visit count và average value
        node = node.parent
```

---

## 📖 Phân Tích Từng Dòng Code

### Dòng 12-31: MCTSAgent Configuration
```python
class MCTSAgent(AbstractTreeSearchAgent):
    def make_planner(self):
        # Create policies
        prior_policy = MCTSAgent.policy_factory(self.config["prior_policy"])
        rollout_policy = MCTSAgent.policy_factory(self.config["rollout_policy"])
        
        # Create MCTS planner
        return MCTS(self.env, prior_policy, rollout_policy, self.config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "budget": 100,                              # Number of MCTS iterations
            "horizon": None,                            # Planning horizon (auto-calculated)
            "prior_policy": {"type": "random_available"}, # Expansion policy
            "rollout_policy": {"type": "random_available"}, # Simulation policy
            "env_preprocessors": []                     # Environment preprocessing
         })
        return config
```

**Key Parameters:**
- **budget**: Computational budget (number of MCTS simulations)
- **horizon**: Planning depth (auto-calculated từ gamma nếu None)
- **prior_policy**: Policy cho node expansion
- **rollout_policy**: Policy cho Monte Carlo simulation

### Dòng 33-97: Policy Factory
```python
@staticmethod
def policy_factory(policy_config):
    if policy_config["type"] == "random":
        return MCTSAgent.random_policy
    elif policy_config["type"] == "random_available":
        return MCTSAgent.random_available_policy
    elif policy_config["type"] == "preference":
        return partial(MCTSAgent.preference_policy,
                       action_index=policy_config["action"],
                       ratio=policy_config["ratio"])

@staticmethod
def random_available_policy(state, observation):
    """Uniform distribution over available actions"""
    if hasattr(state, 'get_available_actions'):
        available_actions = state.get_available_actions()
    else:
        available_actions = np.arange(state.action_space.n)
    
    probabilities = np.ones(len(available_actions)) / len(available_actions)
    return available_actions, probabilities

@staticmethod
def preference_policy(state, observation, action_index, ratio=2):
    """Biased distribution favoring preferred action"""
    available_actions = state.get_available_actions()
    
    # Create biased probabilities
    probabilities = np.ones(len(available_actions)) / (len(available_actions) - 1 + ratio)
    
    # Increase probability of preferred action
    for i, action in enumerate(available_actions):
        if action == action_index:
            probabilities[i] *= ratio
            break
    
    return available_actions, probabilities
```

**Policy Types:**
- **random**: Uniform over all actions
- **random_available**: Uniform over available actions only
- **preference**: Biased toward specific action

### Dòng 100-127: MCTS Planner Initialization
```python
class MCTS(AbstractPlanner):
    def __init__(self, env, prior_policy, rollout_policy, config=None):
        super().__init__(config)
        self.env = env
        self.prior_policy = prior_policy      # For expansion
        self.rollout_policy = rollout_policy  # For simulation
        
        # Auto-calculate horizon if not specified
        if not self.config["horizon"]:
            self.config["episodes"], self.config["horizon"] = \
                OLOP.allocation(self.config["budget"], self.config["gamma"])

    @classmethod
    def default_config(cls):
        cfg = super(MCTS, cls).default_config()
        cfg.update({
            "temperature": 2 / (1 - cfg["gamma"]),  # UCB exploration parameter
            "closed_loop": False                    # Use observations in tree
        })
        return cfg
```

**Configuration:**
- **temperature**: Controls exploration vs exploitation balance
- **closed_loop**: Whether to use observations trong tree structure

### Dòng 132-159: MCTS Main Loop
```python
def run(self, state, observation):
    """Single MCTS iteration: Selection → Expansion → Simulation → Backpropagation"""
    node = self.root
    total_reward = 0
    depth = 0
    terminal = False
    
    # PHASE 1: SELECTION - Navigate to leaf using UCB1
    while depth < self.config['horizon'] and node.children and not terminal:
        # Select action using UCB1 strategy
        action = node.sampling_rule(temperature=self.config['temperature'])
        
        # Step environment
        observation, reward, terminal, truncated, _ = self.step(state, action)
        total_reward += self.config["gamma"] ** depth * reward
        
        # Move to child node
        node_observation = observation if self.config["closed_loop"] else None
        node = node.get_child(action, observation=node_observation)
        depth += 1

    # PHASE 2: EXPANSION - Add children if leaf node
    if not node.children \
            and depth < self.config['horizon'] \
            and (not terminal or node == self.root):
        node.expand(self.prior_policy(state, observation))

    # PHASE 3: SIMULATION - Rollout from current state
    if not terminal:
        total_reward = self.evaluate(state, observation, total_reward, depth=depth)
    
    # PHASE 4: BACKPROPAGATION - Update all nodes in path
    node.update_branch(total_reward)
```

**MCTS Phases:**
1. **Selection**: UCB1-guided navigation to leaf
2. **Expansion**: Add children using prior policy
3. **Simulation**: Monte Carlo rollout
4. **Backpropagation**: Update statistics

### Dòng 160-177: Monte Carlo Simulation
```python
def evaluate(self, state, observation, total_reward=0, depth=0):
    """Monte Carlo rollout to estimate state value"""
    for h in range(depth, self.config["horizon"]):
        # Sample action from rollout policy
        actions, probabilities = self.rollout_policy(state, observation)
        action = self.np_random.choice(actions, 1, p=np.array(probabilities))[0]
        
        # Step environment
        observation, reward, terminal, truncated, _ = self.step(state, action)
        total_reward += self.config["gamma"] ** h * reward
        
        # Stop if terminal
        if np.all(terminal) or np.all(truncated):
            break
    
    return total_reward
```

**Simulation Process:**
- **Random Policy**: Use rollout policy cho action selection
- **Environment Stepping**: Simulate trajectory
- **Reward Accumulation**: Discounted cumulative reward
- **Terminal Handling**: Stop at episode end

### Dòng 179-184: Planning Interface
```python
def plan(self, state, observation):
    """Run multiple MCTS iterations và return best action"""
    for i in range(self.config['episodes']):
        if (i+1) % 10 == 0:
            logger.debug('{} / {}'.format(i+1, self.config['episodes']))
        
        # Run single MCTS iteration
        self.run(safe_deepcopy_env(state), observation)
    
    # Extract best action from tree
    return self.get_plan()
```

**Planning Process:**
- **Multiple Iterations**: Run MCTS budget times
- **Environment Copying**: Safe simulation without affecting real environment
- **Action Extraction**: Select action với highest visit count

---

## 🌳 MCTSNode Implementation

### Dòng 203-246: Node Structure
```python
class MCTSNode(Node):
    K = 1.0  # Value function filter gain
    
    def __init__(self, parent, planner, prior=1):
        super(MCTSNode, self).__init__(parent, planner)
        self.value = 0      # Average reward estimate
        self.prior = prior  # Prior probability

    def expand(self, actions_distribution):
        """Create child nodes for available actions"""
        actions, probabilities = actions_distribution
        for i in range(len(actions)):
            if actions[i] not in self.children:
                # Create child với prior probability
                self.children[actions[i]] = type(self)(self, self.planner, probabilities[i])
```

### Dòng 275-286: UCB1 Selection Strategy
```python
def selection_strategy(self, temperature):
    """UCB1: Upper Confidence Bound for Trees"""
    if not self.parent:
        return self.get_value()

    # UCB1 formula: exploitation + exploration
    exploitation = self.get_value()
    exploration = temperature * len(self.parent.children) * self.prior / (self.count + 1)
    
    return exploitation + exploration
```

**UCB1 Formula:**
```
UCB1(node) = Q(node) + C * sqrt(ln(N_parent) / N_node)

Where:
- Q(node): Average reward (exploitation)
- C: Exploration constant
- N_parent: Parent visit count
- N_node: Node visit count
```

### Dòng 248-265: Value Updates
```python
def update(self, total_reward):
    """Update node statistics với new sample"""
    self.count += 1
    # Incremental average update
    self.value += self.K / self.count * (total_reward - self.value)

def update_branch(self, total_reward):
    """Backpropagate value up the tree"""
    self.update(total_reward)
    if self.parent:
        self.parent.update_branch(total_reward)
```

**Incremental Average:**
```
V_new = V_old + (1/N) * (sample - V_old)

Benefits:
- Numerically stable
- Constant memory
- Online updates
```

---

## 🎯 MCTS Variants trong Framework

### **1. MCTS with Double Progressive Widening (DPW)**
```python
# File: mcts_dpw.py
class MCTSDPWAgent(MCTSAgent):
    """MCTS với progressive action và state widening"""
    
    def expand_actions(self, node, visit_count):
        """Progressive action widening"""
        max_actions = int(self.config["action_widening"] * visit_count ** self.config["action_alpha"])
        return min(max_actions, len(available_actions))
    
    def expand_states(self, node, visit_count):
        """Progressive state widening"""
        max_states = int(self.config["state_widening"] * visit_count ** self.config["state_alpha"])
        return min(max_states, len(possible_states))
```

### **2. MCTS with Prior Knowledge**
```python
# File: mcts_with_prior.py
class MCTSWithPriorAgent(MCTSAgent):
    """MCTS với neural network prior"""
    
    def __init__(self, env, config=None):
        super().__init__(env, config)
        self.value_network = load_value_network(config["value_network_path"])
        self.policy_network = load_policy_network(config["policy_network_path"])
    
    def neural_prior_policy(self, state, observation):
        """Use neural network cho prior probabilities"""
        policy_logits = self.policy_network(observation)
        probabilities = softmax(policy_logits)
        return available_actions, probabilities[available_actions]
    
    def neural_value_evaluation(self, state, observation):
        """Use neural network cho value estimation"""
        return self.value_network(observation)
```

---

## 💡 Ưu Điểm và Nhược Điểm

### ✅ **Ưu Điểm:**
- **Anytime Algorithm**: Can return best action at any time
- **Asymmetric Search**: Focus on promising regions
- **No Evaluation Function**: Learn values through simulation
- **Proven Performance**: Success trong complex domains (Go, Chess)
- **Handles Uncertainty**: Robust to stochastic environments

### ❌ **Nhược Điểm:**
- **Computational Cost**: Requires many simulations
- **Model Dependency**: Needs environment model
- **Horizon Limitation**: Performance degrades với long horizons
- **Rollout Policy**: Quality depends on rollout policy
- **Memory Usage**: Tree grows với search depth

---

## 🚀 Use Cases và Applications

### **Ideal Applications:**
1. **Game Playing**: Perfect/imperfect information games
2. **Robotics**: Motion planning với uncertainty
3. **Resource Management**: Dynamic resource allocation
4. **Autonomous Driving**: Path planning trong traffic
5. **Financial Trading**: Portfolio optimization

### **Environment Requirements:**
```python
# MCTS requires:
# 1. Simulatable environment
assert hasattr(env, 'step') and hasattr(env, 'reset')

# 2. Finite action space (discrete or discretized)
assert env.action_space.n < float('inf')

# 3. Episodic or finite horizon
assert hasattr(env, '_max_episode_steps') or config["horizon"] is not None

# 4. Deterministic or stochastic dynamics
# (MCTS handles both)
```

---

## 🔧 Configuration Best Practices

### **Key Hyperparameters:**
```python
mcts_config = {
    "budget": 1000,                    # More budget = better performance
    "horizon": 50,                     # Match problem horizon
    "gamma": 0.95,                     # Discount factor
    "temperature": 1.0,                # UCB exploration parameter
    "prior_policy": {
        "type": "random_available"     # Or use domain knowledge
    },
    "rollout_policy": {
        "type": "random_available"     # Or use heuristic policy
    },
    "closed_loop": False               # Use observations trong tree
}
```

### **Performance Tuning:**
- **Budget**: More simulations = better decisions, but slower
- **Temperature**: Higher = more exploration, lower = more exploitation
- **Rollout Policy**: Better policy = better value estimates
- **Prior Policy**: Domain knowledge improves expansion
- **Horizon**: Match problem characteristics

**Next Algorithm**: Chúng ta sẽ tiếp tục với **Cross Entropy Method (CEM)** - optimization-based approach cho RL!
