import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Callable, Iterable, Tuple



class CosineAnnealingScheduler(keras.callbacks.Callback):
    """
    Cosine annealing learning rate scheduler.
    """
    def __init__(self, base_lr, min_lr, epochs, verbose=1):
        super().__init__()
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        p = epoch / max(1, self.epochs - 1)
        lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * p))
        keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose and (epoch < 1 or (epoch + 1) % 5 == 0):
            print(f"> [LR Scheduler] epoch {epoch+1}: lr={lr:.6f}")


class TempScheduler(keras.callbacks.Callback):
    """
    Cosine/linear schedules for router exploration & sharpening.

    Schedules any subset of:
      - route_temp:   start → end   (e.g., 2.0 → 0.5)
      - explore_eps:  start → end   (e.g., 0.20 → 0.00)
      - ent_weight:   start → end   (e.g., 1e-2 → 0)
      - lb_weight:    start → end   (e.g., 1e-2 → 0)  # if you add a load-balance term

    Usage example:
      TempScheduler(
          layer_name="adaptive_router",
          epochs=150, mode="cosine",
          route=(1.5, 0.7),
          eps=(0.2, 0.0),
          ent=(1e-2, 0.0),
          lb=(0.0, 0.0),
          verbose=1
      )
    """
    def __init__(self, layer_name="adaptive_router",
                 epochs=150, mode="cosine",
                 route=None,   # tuple (start, end) or None
                 eps=None,     # tuple (start, end) or None
                 ent=None,     # tuple (start, end) or None
                 lb=None,      # tuple (start, end) or None
                 log=True,
                 verbose=1):
        super().__init__()
        self.layer_name = layer_name
        self.E = int(epochs)
        self.mode = mode
        self.verbose = int(verbose)
        self.log = log

        def _pair(x):
            if x is None: return None
            a, b = float(x[0]), float(x[1])
            return a, b

        self.route_pair = _pair(route)
        self.eps_pair   = _pair(eps)
        self.ent_pair   = _pair(ent)
        self.lb_pair    = _pair(lb)

    def _interp(self, e):
        # e is 0-based epoch index
        p = min(1.0, e / max(1, self.E - 1))
        if self.mode == "cosine":
            p = 0.5 * (1.0 - np.cos(np.pi * p))
        return p

    def _apply(self, layer, name, pair, p):
        if pair is None: return None
        if not hasattr(layer, name): return None
        start, end = pair
        val = start + (end - start) * p
        setattr(layer, name, float(val))
        return val

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.get_layer(self.layer_name)
        p = self._interp(epoch)

        vals = {}
        vals["route_temp"] = self._apply(layer, "route_temp", self.route_pair, p)
        vals["explore_eps"] = self._apply(layer, "explore_eps", self.eps_pair, p)
        vals["ent_weight"]  = self._apply(layer, "ent_weight", self.ent_pair, p)
        # if you add a load-balance attr on the layer (e.g., self.lb_weight), this will set it
        vals["lb_weight"]   = self._apply(layer, "lb_weight", self.lb_pair, p)

        if self.verbose and (epoch < 1 or (epoch + 1) % 5 == 0):
            msg = [f"epoch {epoch+1}"]
            for k, v in vals.items():
                if v is not None:
                    msg.append(f"{k}={v:.4f}")
            if self.log: print("> [TempScheduler]", "  ".join(msg))


class RouterStatsCallback(keras.callbacks.Callback):
    def __init__(self, x_val, y_val, layer_name="adaptive_router", batch_size=256, verbose_every=5):
        super().__init__()
        self.xv = x_val
        self.yv = y_val
        self.layer_name = layer_name
        self.bs = int(batch_size)
        self.verbose_every = int(verbose_every)

    def on_epoch_end(self, epoch, logs=None):
        # Run occasionally to avoid slowing training down
        if epoch < 1 or ((epoch + 1) % self.verbose_every == 0):
            layer = self.model.get_layer(self.layer_name)
            K = layer.K
            T = 1  # single routing step

            steps_hist = np.zeros(T + 1, dtype=np.int64)   # [0, 1]
            expert_hist = np.zeros((T, K), dtype=np.int64) # [1, K]
            n_seen = 0

            # Iterate validation set in batches
            for i in range(0, len(self.xv), self.bs):
                xb = self.xv[i:i + self.bs]
                tb = trace_batch(self.model, xb, layer_name=self.layer_name)  # single-step trace

                # shapes: top_indices -> [1, B]
                choices = tb["top_indices"][0]           # [B]
                counts = np.bincount(choices, minlength=K)
                expert_hist[0] += counts

                B = choices.shape[0]
                n_seen += B
                steps_hist[1] += B                       # always one step used

            # With a single step, avg_steps is always 1.0 if we saw any samples
            total_samples = max(1, steps_hist.sum())
            avg_steps = float(np.sum(np.arange(T + 1) * steps_hist) / total_samples)

            print(
                f"> [{self.layer_name}] epoch {epoch + 1}: "
                f"avg_steps={avg_steps:.2f}  "
                f"steps_hist={steps_hist.tolist()}  "
                f"expert_hist={expert_hist.tolist()}"
            )

import numpy as np
import tensorflow as tf
from tensorflow import keras

class RouterStatsMultiStepCallback(keras.callbacks.Callback):
    """
    Logs per-step expert usage and probability metrics for a (multi-step) router layer.
    Expects `trace_batch(model, x, layer_name=...)` to return a dict with:
        "top_indices": np.ndarray [T, B]
        "probs":       np.ndarray [T, B, K]
    """
    def __init__(self,
                 x_val,
                 y_val=None,
                 layer_name="adaptive_router",
                 batch_size=256,
                 verbose_every=5):
        super().__init__()
        self.xv = np.array(x_val)
        self.yv = None if y_val is None else np.array(y_val)
        self.layer_name = layer_name
        self.bs = int(batch_size)
        self.verbose_every = int(verbose_every)

    def on_epoch_end(self, epoch, logs=None):
        # Run occasionally to avoid slowing training
        if epoch >= 1 and ((epoch + 1) % self.verbose_every != 0):
            return

        layer = self.model.get_layer(self.layer_name)
        K = int(layer.K)
        T = int(getattr(layer, "steps", 1))

        # Histograms / accumulators
        steps_hist   = np.zeros(T + 1, dtype=np.int64)     # index t+1 counts samples that reached step t
        expert_hist  = np.zeros((T, K), dtype=np.int64)    # counts of argmax expert per step

        # Probability-based accumulators
        probs_sum    = np.zeros((T, K), dtype=np.float64)  # sum of probs per expert (for mean mass)
        entropy_sum  = np.zeros(T, dtype=np.float64)       # sum of entropies per step
        top1conf_sum = np.zeros(T, dtype=np.float64)       # sum of top-1 confidences per step
        count_sum    = np.zeros(T, dtype=np.int64)         # number of samples seen per step

        # Optional eval
        if self.yv is not None:
            loss, acc = self.model.evaluate(self.xv, self.yv, batch_size=self.bs, verbose=0)
            print(f"\n> [{self.layer_name}] epoch {epoch + 1}  eval: loss={loss:.4f}  acc={acc*100:.2f}%")

        # Iterate validation set in batches
        for i in range(0, len(self.xv), self.bs):
            xb = self.xv[i:i + self.bs]
            tb = trace_batch(self.model, xb, layer_name=self.layer_name)  # expects probs & top_indices
            top_indices = np.asarray(tb["top_indices"])    # [T,B]
            probs       = np.asarray(tb["probs"])          # [T,B,K]

            T_used, B = top_indices.shape[0], top_indices.shape[1]

            # Update expert usage counts and steps histogram
            for t in range(T_used):
                choices = top_indices[t]                                # [B]
                counts  = np.bincount(choices, minlength=K)             # [K]
                expert_hist[t] += counts
                steps_hist[t + 1] += B

            # Probability metrics
            # Entropy per sample: H = -sum_k p_k log p_k
            p_clipped = np.clip(probs, 1e-9, 1.0)
            ent = -np.sum(probs * np.log(p_clipped), axis=-1)           # [T,B]
            top1conf = np.max(probs, axis=-1)                           # [T,B]

            # Accumulate per step
            probs_sum[:T_used]    += probs.sum(axis=1)                  # sum over batch -> [T,K]
            entropy_sum[:T_used]  += ent.sum(axis=1)                    # [T]
            top1conf_sum[:T_used] += top1conf.sum(axis=1)               # [T]
            count_sum[:T_used]    += B

        # Averages
        total_samples = int(steps_hist[1:].sum())  # total samples counted across steps
        avg_steps = 0.0 if total_samples == 0 else float(np.sum(np.arange(1, T + 1) * steps_hist[1:]) / max(steps_hist[1:].max(), 1))

        # Print summary
        print(f"\n> [{self.layer_name}] epoch {epoch + 1}")
        print(f"  samples={int(count_sum.max()) if count_sum.size>0 else 0}  steps={T}  experts={K}")
        print(f"  avg_steps={avg_steps:.2f}")
        print(f"  steps_hist={steps_hist.tolist()}")

        for t in range(T):
            # Expert argmax usage (counts & percentage)
            usage = expert_hist[t]
            total = usage.sum()
            if total == 0:
                print(f"  step {t+1:2d}: total=0")
                continue
            perc = 100.0 * usage / total
            usage_str = "  ".join([f"E{i}:{p:4.1f}%" for i, p in enumerate(perc)])

            # Probability means
            b_t = max(int(count_sum[t]), 1)
            mean_mass = probs_sum[t] / b_t                                # mean prob mass per expert
            mean_entropy = float(entropy_sum[t] / b_t)
            mean_top1 = float(top1conf_sum[t] / b_t)

            # Keep mean_mass compact: show up to first 10 experts; customize as needed
            show_k = min(K, 10)
            mass_preview = " ".join([f"E{i}:{m:.3f}" for i, m in enumerate(mean_mass[:show_k])])
            tail_note = "" if K <= show_k else " ..."

            print(f"  step {t+1:2d}: total={int(total):5d}  {usage_str}")
            print(f"             mean_top1={mean_top1:.3f}  mean_entropy={mean_entropy:.3f}")
            print(f"             mean_prob_mass: {mass_preview}{tail_note}")



class TopNScheduler(tf.keras.callbacks.Callback):
    """
    Generic scheduler for AdaptiveRouter.top_n.
    Provide a callable schedule: epoch (int) -> top_n (int).
    """
    def __init__(self, schedule: Callable[[int], int], verbose: int = 1):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose

    def _iter_routers(self):
        # Walk all submodules to catch routers nested inside other Layers/Models
        for m in self.model.submodules:
            # Replace AdaptiveRouter with your actual class if imported differently
            if m.__class__.__name__ == "AdaptiveRouter":  # avoids import issues
                yield m

    def on_epoch_begin(self, epoch, logs=None):
        new_top_n = int(self.schedule(epoch))
        changed = []
        for router in self._iter_routers():
            # Clamp to valid range [1, K]
            k = int(router.K)
            t = max(1, min(new_top_n, k))
            if getattr(router, "top_n", None) != t:
                router.top_n = t
                changed.append((router.name or "AdaptiveRouter", t, k))
        if self.verbose and changed:
            msg = "; ".join([f"{name}.top_n={t}/{k}" for name, t, k in changed])
            print(f"[TopNScheduler] epoch {epoch}: {msg}")


def linear_topn_schedule(n_start: int, n_end: int, start_epoch: int, end_epoch: int) -> Callable[[int], int]:
    """
    Linear interpolation from n_start to n_end between start_epoch..end_epoch (inclusive).
    Before start_epoch -> n_start; after end_epoch -> n_end.
    """
    assert end_epoch >= start_epoch, "end_epoch must be >= start_epoch"

    def _sched(epoch: int) -> int:
        if epoch <= start_epoch:
            return n_start
        if epoch >= end_epoch:
            return n_end
        # fraction in [0,1]
        t = (epoch - start_epoch) / float(max(1, end_epoch - start_epoch))
        val = n_start + t * (n_end - n_start)
        return int(round(val))
    return _sched


def milestone_topn_schedule(milestones: Iterable[Tuple[int,int]]) -> Callable[[int], int]:
    """
    Piecewise-constant schedule via milestones: list of (epoch, top_n).
    Example: [(0,7), (50,4), (80,2)]
    """
    # sort to be safe
    milestones = sorted(milestones, key=lambda x: x[0])
    def _sched(epoch: int) -> int:
        current = milestones[0][1]
        for e, n in milestones:
            if epoch >= e:
                current = n
            else:
                break
        return current
    return _sched


def router_stats_multistep_once(
    model,
    x_val,
    layer_name="efficient_router",   # name of your (multi-step) router layer
    batch_size=256,
    topk=None,                       # optional: restrict how many top-k indices trace_batch logs
    verbose=True,
):
    """
    Compute and (optionally) print per-step expert usage for a multi-step router layer.
    Also aggregates probability metrics from trace_batch(...)[\"probs\"]:
      - mean probability mass per expert (per step)
      - mean top-1 confidence (per step)
      - mean entropy of routing distribution (per step)

    Returns a dict:
      {
        "steps":         T,
        "experts":       K,
        "samples":       n_seen,
        "avg_steps":     average steps used,
        "steps_hist":    np.ndarray [T+1],
        "expert_hist":   np.ndarray [T,K],
        "mean_prob_mass":np.ndarray [T,K],     # average p(e) per expert and step
        "mean_top1":     np.ndarray [T],       # average max_k p_k per step
        "mean_entropy":  np.ndarray [T],       # average -sum p log p per step
      }
    """
    layer = model.get_layer(layer_name)
    K = int(layer.K)
    T = int(getattr(layer, "steps", 1))

    steps_hist   = np.zeros(T + 1, dtype=np.int64)   # index t+1 counts samples that reached step t
    expert_hist  = np.zeros((T, K), dtype=np.int64)
    probs_sum    = np.zeros((T, K), dtype=np.float64)
    entropy_sum  = np.zeros(T, dtype=np.float64)
    top1_sum     = np.zeros(T, dtype=np.float64)
    count_sum    = np.zeros(T, dtype=np.int64)

    n_seen = 0

    # Iterate validation set in batches and aggregate
    for i in range(0, len(x_val), batch_size):
        xb = x_val[i:i + batch_size]
        tb = trace_batch(model, xb, layer_name=layer_name, topk=topk)  # expects {"top_indices":[T,B], "probs":[T,B,K]}
        top_indices = np.asarray(tb["top_indices"])                     # [T, B]
        probs       = np.asarray(tb["probs"])                           # [T, B, K]

        T_used, B = top_indices.shape[0], top_indices.shape[1]
        n_seen += B

        # Histograms from argmax choices
        for t in range(T_used):
            choices = top_indices[t]                                    # [B]
            counts  = np.bincount(choices, minlength=K)
            expert_hist[t] += counts
            steps_hist[t + 1] += B

        # Probability metrics
        # Entropy per sample: H = -sum_k p_k log p_k
        p_clip = np.clip(probs[:T_used], 1e-9, 1.0)                     # [T_used,B,K]
        ent    = -np.sum(p_clip * np.log(p_clip), axis=-1)              # [T_used,B]
        top1   = np.max(probs[:T_used], axis=-1)                        # [T_used,B]

        probs_sum[:T_used]   += probs[:T_used].sum(axis=1)              # [T_used,K]
        entropy_sum[:T_used] += ent.sum(axis=1)                         # [T_used]
        top1_sum[:T_used]    += top1.sum(axis=1)                        # [T_used]
        count_sum[:T_used]   += B

    total_samples = int(steps_hist[1:].sum())
    avg_steps = 0.0 if total_samples == 0 else float(
        np.sum(np.arange(1, T + 1) * steps_hist[1:]) / max(steps_hist[1:].max(), 1)
    )

    # Compute means safely
    denom = np.maximum(count_sum, 1).astype(np.float64)                 # [T]
    mean_prob_mass = np.divide(probs_sum, denom[:, None])               # [T,K]
    mean_entropy   = np.divide(entropy_sum, denom)                      # [T]
    mean_top1      = np.divide(top1_sum, denom)                         # [T]

    if verbose:
        print(f"\n> [{layer_name}] samples={n_seen}  steps={T}  experts={K}")
        print(f"  avg_steps={avg_steps:.2f}")
        print(f"  steps_hist={steps_hist.tolist()}")
        for t in range(T):
            usage = expert_hist[t]
            total = usage.sum()
            if total == 0:
                print(f"  step {t+1:2d}: total=0")
                continue
            perc = 100.0 * usage / total
            usage_str = "  ".join([f"E{i}:{p:4.1f}%" for i, p in enumerate(perc)])
            print(f"  step {t+1:2d}: total={int(total):5d}  {usage_str}")
            # Compact preview of mean prob mass (show first up to 10 experts)
            show_k = min(K, 10)
            mass_preview = " ".join([f"E{i}:{m:.3f}" for i, m in enumerate(mean_prob_mass[t,:show_k])])
            tail = "" if K <= show_k else " ..."
            print(f"             mean_top1={mean_top1[t]:.3f}  mean_entropy={mean_entropy[t]:.3f}")
            print(f"             mean_prob_mass: {mass_preview}{tail}")

    return {
        "steps":         T,
        "experts":       K,
        "samples":       n_seen,
        "avg_steps":     avg_steps,
        "steps_hist":    steps_hist,
        "expert_hist":   expert_hist,
        "mean_prob_mass":mean_prob_mass,
        "mean_top1":     mean_top1,
        "mean_entropy":  mean_entropy,
    }


# -------- helpers --------

def _unpack_router_output(router_out, x):
    """
    Accepts router outputs in formats:
      - (logits, ctx, attn_map)
      - (logits, ctx)
      - logits
    Returns (logits, attn_map). attn_map defaults to zeros if not provided.
    """
    if isinstance(router_out, (tuple, list)):
        if len(router_out) == 3:
            logits, _, attn_map = router_out
        elif len(router_out) == 2:
            logits, _ = router_out
            attn_map = tf.zeros_like(x[..., :1])
        else:
            logits = router_out[0]
            attn_map = tf.zeros_like(x[..., :1])
    else:
        logits = router_out
        attn_map = tf.zeros_like(x[..., :1])
    return logits, attn_map


def _mix_topn(weights, y_stack, top_n):
    """
    Sparse top-n mixture with renormalization.
    weights: [B,K]
    y_stack: [B,K,H,W,C]
    top_n:   int
    returns: [B,H,W,C]
    """
    K = tf.shape(weights)[1]
    k = tf.minimum(tf.cast(top_n, tf.int32), K)
    topk = tf.math.top_k(weights, k=k)
    mask = tf.reduce_sum(tf.one_hot(topk.indices, K, dtype=weights.dtype), axis=1)  # [B,K]
    sparse_w = tf.where(mask > 0, weights, tf.zeros_like(weights))
    sparse_w = sparse_w / (tf.reduce_sum(sparse_w, axis=-1, keepdims=True) + 1e-12)
    return tf.einsum('bk,bkhwc->bhwc', sparse_w, y_stack), topk.indices  # also return indices for tracing




def trace_and_predict(
    model,
    x_input,
    y_true=None,
    layer_name="efficient_router",   # <-- set to your EfficientMultiStepRouter layer name
    topk=None,                       # how many top-k indices to log per step (None -> K)
):
    """
    Runs the model and traces per-step routing for an EfficientMultiStepRouter.

    Returns:
      {
        "trace": {
          "top_indices":  [T, B],          # top-1 per step
          "topk_indices": [T, B, k_eff],   # top-k per step (k_eff = min(topk or K, K))
          "probs":        [T, B, K],       # softmax weights per step
        },
        "pred_probs":  model(x_input) output (numpy),
        "pred_label":  argmax over last dim (if classification),
        "true_label":  y_true (as provided),
        "layer_info":  {K, steps, route_temp},
        "correct":     (pred_label == y_true) if y_true provided else None,
      }
    """
    layer = model.get_layer(layer_name)
    steps = int(getattr(layer, "steps", 1))
    K = int(layer.K)
    k_eff = K if topk is None else int(min(topk, K))

    # Features BEFORE the router layer
    pre = keras.Model(model.input, layer.input)
    x_in = tf.convert_to_tensor(x_input)
    x = pre(x_in, training=False)                     # [B,H,W,C]
    B = int(x.shape[0])
    temp = tf.cast(layer.route_temp, x.dtype)

    top1_all, topk_all, probs_all = [], [], []
    x_step = x

    for _ in range(steps):
        # Route once (CosineRouter returns logits only)
        logits  = layer.router(x_step, training=False)           # [B,K]
        weights = tf.nn.softmax(logits / temp, axis=-1)          # [B,K]

        # Record stats
        top1 = tf.argmax(weights, axis=-1, output_type=tf.int32) # [B]
        topn = tf.math.top_k(weights, k=k_eff, sorted=True).indices  # [B,k_eff]
        top1_all.append(top1.numpy())
        topk_all.append(topn.numpy())
        probs_all.append(weights.numpy())

        # Compute ALL experts and dense mix (mirror layer)
        y_list  = [br(x_step, training=False) for br in layer.branches]  # K × [B,H,W,C]
        y_stack = tf.stack(y_list, axis=1)                                # [B,K,H,W,C]
        x_step  = tf.einsum('bk,bkhwc->bhwc', weights, y_stack)           # next state

    # Full model predictions
    pred_probs = model(x_in, training=False).numpy()
    pred_label = np.argmax(pred_probs, axis=-1) if pred_probs.ndim >= 2 else (pred_probs > 0.5).astype("int32")

    correct = None
    if y_true is not None:
        y_true_arr = np.asarray(y_true).reshape(-1)
        correct = (pred_label.reshape(-1) == y_true_arr)

    trace = {
        "top_indices":  np.stack(top1_all, axis=0),   # [T,B]
        "topk_indices": np.stack(topk_all, axis=0),   # [T,B,k_eff]
        "probs":        np.stack(probs_all, axis=0),  # [T,B,K]
    }
    return {
        "trace": trace,
        "pred_probs": pred_probs,
        "pred_label": pred_label,
        "true_label": None if y_true is None else np.asarray(y_true),
        "layer_info": {
            "K": K,
            "steps": steps,
            "route_temp": float(layer.route_temp),
        },
        "correct": correct,
    }


# ---------- TRACE (per batch) ----------
def trace_batch(model, x_batch, layer_name="efficient_router", topk=None):
    """
    Traces routing decisions for EfficientMultiStepRouter:
      - Returns per-step top-1 indices, top-k indices, and probabilities.
      - Advances x the same way as the layer: dense mixture over ALL experts.
    Args:
      model: tf.keras.Model
      x_batch: input batch (numpy or Tensor)
      layer_name: name of the EfficientMultiStepRouter layer in the model
      topk: if None, uses K; else restricts top-k logging to min(topk, K)
    Returns dict with:
      {
        "top_indices":  [T, B],          # top-1 per step
        "topk_indices": [T, B, k_eff],   # top-k per step
        "probs":        [T, B, K],       # softmax weights per step
      }
    """
    layer = model.get_layer(layer_name)
    assert hasattr(layer, "router") and hasattr(layer, "branches"), \
        f"Layer '{layer_name}' must be an EfficientMultiStepRouter."

    steps = int(getattr(layer, "steps", 1))
    K = int(layer.K)
    k_eff = K if topk is None else int(min(topk, K))

    # Features BEFORE the router layer
    pre = keras.Model(model.input, layer.input)
    x_in = tf.convert_to_tensor(x_batch)
    x = pre(x_in, training=False)

    B = int(x.shape[0])
    temp = tf.cast(layer.route_temp, x.dtype)

    top1_all, topk_all, probs_all = [], [], []

    x_step = x
    for _ in range(steps):
        # Route once (CosineRouter returns logits only)
        logits  = layer.router(x_step, training=False)            # [B,K]
        weights = tf.nn.softmax(logits / temp, axis=-1)           # [B,K]

        # Record stats
        top1 = tf.argmax(weights, axis=-1, output_type=tf.int32)  # [B]
        topn = tf.math.top_k(weights, k=k_eff, sorted=True).indices  # [B,k_eff]

        top1_all.append(top1.numpy())
        topk_all.append(topn.numpy())
        probs_all.append(weights.numpy())

        # Compute ALL experts, dense mixture (mirror layer)
        y_list  = [br(x_step, training=False) for br in layer.branches]  # K × [B,H,W,C]
        y_stack = tf.stack(y_list, axis=1)                               # [B,K,H,W,C]
        x_step  = tf.einsum('bk,bkhwc->bhwc', weights, y_stack)          # [B,H,W,C]

    return {
        "top_indices":  np.stack(top1_all, axis=0),    # [T,B]
        "topk_indices": np.stack(topk_all, axis=0),    # [T,B,k_eff]
        "probs":        np.stack(probs_all, axis=0),   # [T,B,K]
    }


# ---------- AGGREGATE / PRINT STATS ----------
def print_router_stats(model, x, y, layer_name="efficient_router", batch_size=512, topk=None):
    """
    Evaluates the model and prints per-step expert usage for EfficientMultiStepRouter.
    """
    # Eval once for accuracy
    loss, acc = model.evaluate(x, y, batch_size=batch_size, verbose=0)

    layer = model.get_layer(layer_name)
    K = int(layer.K)
    T = int(getattr(layer, "steps", 1))
    k_eff = K if topk is None else int(min(topk, K))

    steps_hist  = np.zeros(T + 1, dtype=np.int64)   # index t+1 counts samples that reached step t
    expert_hist = np.zeros((T, K), dtype=np.int64)
    n_seen = 0

    # Aggregate per-step counts
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        tb = trace_batch(model, xb, layer_name=layer_name, topk=k_eff)
        top1 = tb["top_indices"]  # [T,B]
        T_used, B = top1.shape
        n_seen += B

        for t in range(T_used):
            counts = np.bincount(top1[t], minlength=K)
            expert_hist[t] += counts
            steps_hist[t + 1] += B

    # Pretty print
    print("")
    print(f"================== {layer_name} ====================")
    print(f"Test acc: {acc*100:.2f}%  |  loss: {loss:.4f}  |  samples: {n_seen}")
    print("Steps histogram (index t+1 = #samples that reached step t):",
          steps_hist.tolist())

    for t in range(T):
        usage = expert_hist[t]
        total = usage.sum()
        if total == 0:
            print(f"Step {t+1:2d}: total=0")
            continue
        perc = 100.0 * usage / total
        usage_str = "  ".join([f"E{i}:{p:4.1f}%" for i, p in enumerate(perc)])
        print(f"Step {t+1:2d}: total={int(total):5d}  {usage_str}")


# ---------- TRACE + PREDICT (for sample ranges) ----------
def trace_and_predict(model, x, y_true=None, layer_name="efficient_router"):
    """
    Runs a forward pass to get prediction, and traces router for the same sample(s).
    Returns:
      {
        "pred":       model logits/probs (as returned by model(x)),
        "pred_label": argmax labels (if classification),
        "true_label": provided y_true (if given),
        "trace":      output of trace_batch(...)
      }
    """
    pred = model.predict(x, verbose=0)
    if pred.ndim >= 2:
        pred_label = np.argmax(pred, axis=-1)
    else:
        pred_label = (pred > 0.5).astype("int32")

    trace = trace_batch(model, x, layer_name=layer_name)

    return {
        "pred": pred,
        "pred_label": pred_label,
        "true_label": y_true if y_true is not None else None,
        "trace": trace,
    }


def print_trace_for_samples(model, x, y, layer_name="efficient_router", start=0, end=10):
    """
    Prints routing trace and prediction for samples in the given range.
    """
    print("")
    print(f"================== {layer_name} ====================")
    end = min(end, len(x))
    for i in range(start, end):
        xi = x[i:i+1]
        yi = y[i:i+1] if y is not None else None
        res = trace_and_predict(model, xi, y_true=yi, layer_name=layer_name)
        trace = res["trace"]
        T, B = trace["top_indices"].shape
        if T > 0 and B > 0:
            print(" > pred label:", int(res["pred_label"][0]),
                  "true label:", int(yi[0]) if yi is not None else None)
            # Show top-1 per step for this sample
            seq = trace["top_indices"][:, 0].tolist()
            print("   experts per step:", seq)
