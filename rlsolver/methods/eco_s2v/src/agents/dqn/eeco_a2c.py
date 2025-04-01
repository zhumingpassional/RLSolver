import torch
import torch.optim as optim
import torch.nn.functional as F
import os, time
import tqdm

class A2C:
    """
    示例 A2C 算法类，用于将你原先的 DQN 迁移到 A2C。
    """
    def __init__(
        self,
        env,
        network,                 # 传入 ActorCriticMPNN 或类似网络结构
        gamma=0.99,
        lr=1e-4,
        value_loss_coef=0.5,    # Critic loss 系数
        entropy_coef=0.01,      # 策略熵系数 (可选)
        max_grad_norm=None,
        n_steps=5,              # A2C 常见用 n-step 回报
        device='cuda',
        ):
        
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.device = device

        # 创建 Actor-Critic 网络
        self.network = network().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        self.global_step = 0

    def act(self, state):
        """
        基于当前策略网络输出动作。
        state: shape (batch_size, num_nodes, num_nodes+7)
        返回:
          action: (batch_size,) 整数动作 (要翻转哪个点)
          log_probs: (batch_size,) 该动作对应的 log π(a|s)
          value: (batch_size,) Critic 值
        """
        state = state.to(self.device)
        with torch.no_grad():
            logits, value = self.network(state.clone())
            # logits: (batch_size, num_nodes)
            # value:  (batch_size, 1)

            # 对 logits 做 softmax，得到对每个节点翻转动作的概率
            probs = F.softmax(logits, dim=-1)  # (batch_size, num_nodes)
            
            # 这里随机采样一个节点翻转 => action
            # 如果想在不可逆环境里只对“未翻转”节点做 softmax，可在 probs 上做一次 mask
            # 再重新归一化，然后采样
            action = []
            log_probs = []
            for i in range(state.shape[0]):
                # 从 probs[i] 这个分布中采样
                act_i = torch.multinomial(probs[i], num_samples=1)
                action.append(act_i.item())
                log_probs.append(torch.log(probs[i][act_i]))
            
            action = torch.tensor(action, dtype=torch.long, device=self.device)
            log_probs = torch.cat(log_probs, dim=0)
            value = value.squeeze(-1)  # => (batch_size,)
        
        return action, log_probs, value

    def compute_returns(self, rewards, dones, last_value):
        """
        计算 n-step 回报 (bootstrap)，类似 A2C/GAE。
        rewards, dones: shape (n_steps, batch_size)
        last_value: shape (batch_size,)
        返回:
          returns: shape (n_steps, batch_size)
        """
        n_steps, batch_size = rewards.shape
        returns = torch.zeros_like(rewards, device=self.device)
        # 对最后一步做 bootstrap
        running_return = last_value
        for t in reversed(range(n_steps)):
            running_return = rewards[t] + self.gamma * running_return * (1.0 - dones[t].float())
            returns[t] = running_return
        return returns

    def learn(self, total_timesteps):
        """
        A2C 主循环。每收集 n_steps，就做一次更新。
        """
        start_time = time.time()
        
        # reset
        state = self.env.reset()  # shape (batch_size, num_nodes, num_nodes+7)
        batch_size = state.shape[0]

        # 用于存储一次 rollout 的序列
        mb_states = []
        mb_actions = []
        mb_log_probs = []
        mb_values = []
        mb_rewards = []
        mb_dones = []

        for step_i in tqdm.tqdm(range(total_timesteps)):
            self.global_step += 1

            # 选动作
            action, log_prob, value = self.act(state)
            
            # 与环境交互
            next_state, reward, done = self.env.step(action)

            # 记录数据
            mb_states.append(state)
            mb_actions.append(action)
            mb_log_probs.append(log_prob)
            mb_values.append(value)
            mb_rewards.append(reward)
            mb_dones.append(done)

            state = next_state

            # 如果回合结束，需要重置环境并清空得分
            if done[0]:
                print(self.env.best_score)
                state = self.env.reset()
            
            # 当我们存满 n_steps 或者 episode 结束时，就做一次训练 (A2C on-policy)
            # 这里简单写：每 n_steps 做一次更新
            if (step_i + 1) % self.n_steps == 0:
                # 需要 bootstrap 最后的状态
                with torch.no_grad():
                    _, next_value = self.network(state.to(self.device))
                    next_value = next_value.squeeze(-1)

                # 把收集到的序列整合到一起
                # (n_steps, batch_size) 的张量
                rewards = torch.stack(mb_rewards, dim=0)  # shape (n_steps, batch_size)
                dones   = torch.stack(mb_dones, dim=0)    # shape (n_steps, batch_size)
                values  = torch.stack(mb_values, dim=0)   # shape (n_steps, batch_size)
                log_probs = torch.stack(mb_log_probs, dim=0) # shape (n_steps, batch_size)

                # 计算 n-step returns
                returns = self.compute_returns(rewards, dones, next_value)

                # 优势 A = returns - values
                advantages = returns - values

                # ========== 计算损失 ==========
                # 1) Policy gradient loss: - logπ(a|s)*A
                policy_loss = -(log_probs * advantages.detach()).mean()

                # 2) Value loss
                value_loss = F.mse_loss(values, returns)

                # 3) (可选) 增加策略熵做探索
                #   为了获得概率分布，需要再计算一次 logits => pi => log pi
                #   但我们已经有 log_probs，可用来估计熵
                #   这里粗略地再 forward 一下演示
                logits, _ = self.network(state.to(self.device))
                dist = F.softmax(logits, dim=-1)
                dist_log = torch.log(dist + 1e-8)
                entropy = -(dist * dist_log).sum(dim=-1).mean()

                # 总损失
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # 反向传播与更新
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # 清空存储
                mb_states.clear()
                mb_actions.clear()
                mb_log_probs.clear()
                mb_values.clear()
                mb_rewards.clear()
                mb_dones.clear()
        self.save()
        print(f"A2C training finished, total time: {time.time() - start_time:.2f} s")

    def save(self, path='network_a2c.pth'):
        folder_path = os.path.dirname(path)
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
