import torch
from torch import nn
from torch.nn import functional as F

N_ACTIONS = 3


class QNet(nn.Module):
    def __init__(self, dim, hidden, n_out=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


class VNet(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def expectile_loss(diff, tau):
    weight = torch.where(diff < 0, 1 - tau, tau)
    return (weight * diff.pow(2)).mean()


def cql_regularizer(q, actions, alpha):
    logsum = torch.logsumexp(q, dim=-1)
    q_sa = q.gather(1, actions.view(-1, 1)).squeeze(1)
    return alpha * (logsum - q_sa).mean()


def awr_loss(q, v, actions, temp):
    adv = (q.gather(1, actions.view(-1, 1)).squeeze(1) - v).detach() / temp
    logp = F.log_softmax(q, dim=-1).gather(1, actions.view(-1, 1)).squeeze(1)
    return -(torch.exp(adv.clamp(max=8)) * logp).mean()


class Agent(nn.Module):
    def __init__(self, dim, hidden, kind, always_visible=False):
        super().__init__()
        self.kind = kind
        self.always_visible = always_visible
        self.q = QNet(dim, hidden)
        self.v = VNet(dim, hidden)
        self.c_miss = QNet(dim, hidden)
        self.c_over = QNet(dim, hidden)
        self.channel = QNet(dim, hidden, n_out=2)

    def q_sa(self, q, actions):
        return q.gather(1, actions.view(-1, 1)).squeeze(1)


def pack_batch(batch, device):
    s, a, r, sp, d, cm, co = batch
    to = lambda x, dtype: torch.tensor(x, dtype=dtype, device=device)
    return (
        to(s, torch.float32),
        to(a, torch.long),
        to(r, torch.float32),
        to(sp, torch.float32),
        to(d, torch.float32),
        to(cm, torch.float32),
        to(co, torch.float32),
    )


def train_step(agent, opt, batch, cfg, dual):
    s, a, r, sp, done, cmiss, cover = pack_batch(batch, next(agent.parameters()).device)
    gamma = cfg["gamma"]
    q = agent.q(s)
    q_sa = agent.q_sa(q, a)
    with torch.no_grad():
        v_sp = agent.v(sp)
        target = r - dual["lam_miss"] * cmiss - dual["lam_over"] * cover + gamma * (1 - done) * v_sp
        if agent.kind == "onestep":
            target = r - dual["lam_miss"] * cmiss - dual["lam_over"] * cover
    v_s = agent.v(s)
    loss_v = expectile_loss(q_sa.detach() - v_s, cfg["expectile"])
    loss_q = F.mse_loss(q_sa, target)
    if agent.kind == "cql":
        loss_q = loss_q + cql_regularizer(q, a, cfg["cql_alpha"])
    if agent.kind == "bc":
        loss = F.cross_entropy(q, a)
    elif agent.kind == "onestep":
        loss = F.cross_entropy(q, a) + loss_q
    elif agent.kind == "rda_no_support":
        loss = loss_q + loss_v
    else:
        loss = loss_q + loss_v + awr_loss(q, v_s.detach(), a, cfg["awr_temp"])
    use_cost = agent.kind in {"rda", "rda_no_channel_mask", "rda_no_support"}
    if use_cost:
        ch = agent.channel(s)
        next_ch = (sp[:, 0] > 0.5).long()
        loss = loss + F.cross_entropy(ch, next_ch)
        miss_q = agent.q_sa(agent.c_miss(s), a)
        over_q = agent.q_sa(agent.c_over(s), a)
        with torch.no_grad():
            miss_t = cmiss + gamma * (1 - done) * agent.q_sa(agent.c_miss(sp), a)
            over_t = cover + gamma * (1 - done) * agent.q_sa(agent.c_over(sp), a)
        loss = loss + F.mse_loss(miss_q, miss_t) + F.mse_loss(over_q, over_t)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if use_cost:
        with torch.no_grad():
            dual["lam_miss"] = max(0.0, dual["lam_miss"] + 0.01 * (cmiss.mean().item() - cfg["kappa_miss"]))
            dual["lam_over"] = max(0.0, dual["lam_over"] + 0.01 * (cover.mean().item() - cfg["kappa_over"]))
    return {"loss": float(loss.detach()), "lam_miss": dual["lam_miss"], "lam_over": dual["lam_over"]}


def act_greedy(agent, states):
    with torch.no_grad():
        q = agent.q(states)
        if agent.kind.startswith("rda") and agent.kind != "rda_no_cost":
            q = q - agent.c_miss(states) - agent.c_over(states)
        return q.argmax(dim=-1)


def behavior_rates(steps):
    n = max(len(steps), 1)
    return {
        "WatchTBP": sum(s["action"] == 0 for s in steps) / n,
        "AcquireDerm": sum(s["action"] == 1 for s in steps) / n,
        "ReferExcise": sum(s["action"] == 2 for s in steps) / n,
        "return": sum(s["reward"] for s in steps) / n,
        "cost_miss": sum(s["cost_miss"] for s in steps) / n,
        "cost_over": sum(s["cost_over"] for s in steps) / n,
        "n": len(steps),
    }


def protocol_rates(steps, pred, proto):
    from .trajectories import outcome_under_action

    n = max(len(steps), 1)
    logged = behavior_rates(steps)
    reward = cost_miss = cost_over = 0.0
    for step, action in zip(steps, pred):
        out = outcome_under_action(step, action, proto)
        reward += out["reward"]
        cost_miss += out["cost_miss"]
        cost_over += out["cost_over"]
    pred = [int(a) for a in pred]
    logged_actions = [int(s["action"]) for s in steps]
    agree = sum(int(p == t) for p, t in zip(pred, logged_actions)) / n
    return {
        "WatchTBP": logged["WatchTBP"],
        "AcquireDerm": logged["AcquireDerm"],
        "ReferExcise": logged["ReferExcise"],
        "logged_return": logged["return"],
        "logged_cost_miss": logged["cost_miss"],
        "logged_cost_over": logged["cost_over"],
        "return": reward / n,
        "cost_miss": cost_miss / n,
        "cost_over": cost_over / n,
        "policy_WatchTBP": pred.count(0) / n,
        "policy_AcquireDerm": pred.count(1) / n,
        "policy_ReferExcise": pred.count(2) / n,
        "action_agree": agree,
        "ess": ess(pred, logged_actions),
        "eval_kind": "protocol_counterfactual_one_step",
        "n": len(steps),
    }


def greedy_rates(agent, steps, proto, always_visible=False):
    from .dataset import encode_batch

    s, *_ = encode_batch(steps, proto, always_visible=always_visible)
    device = next(agent.parameters()).device
    pred = act_greedy(agent, torch.tensor(s, dtype=torch.float32, device=device)).cpu().tolist()
    return protocol_rates(steps, pred, proto)


def ess(pred, logged):
    # one-step IS effective sample size under greedy vs logged Bernoulli match
    n = len(logged)
    if n == 0:
        return 0.0
    p = sum(int(a == b) for a, b in zip(pred, logged)) / n
    p = min(max(p, 1e-6), 1.0)
    w = [p if a == b else (1 - p) for a, b in zip(pred, logged)]
    s = sum(w)
    s2 = sum(x * x for x in w)
    return float(s * s / s2) if s2 else 0.0
