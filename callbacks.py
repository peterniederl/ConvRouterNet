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


class RouterStatsMultiStepCallback(keras.callbacks.Callback):
    """
    Logs per-step expert usage for an AdaptiveRouterMultiStep layer.
    Expects `trace_batch(model, x, layer_name=...)` to return
    a dict with key "top_indices": list/array of shape [steps, batch].
    """
    def __init__(self, x_val, y_val,
                 layer_name="adaptive_router",
                 batch_size=256,
                 verbose_every=5):
        super().__init__()
        self.xv = np.array(x_val)
        self.yv = np.array(y_val)
        self.layer_name = layer_name
        self.bs = int(batch_size)
        self.verbose_every = int(verbose_every)

    def on_epoch_end(self, epoch, logs=None):
        # Run occasionally
        if epoch < 1 or ((epoch + 1) % self.verbose_every == 0):
            layer = self.model.get_layer(self.layer_name)
            K = layer.K
            T = getattr(layer, "steps", 1)  # support both single-step and multi-step

            steps_hist = np.zeros(T + 1, dtype=np.int64)
            expert_hist = np.zeros((T, K), dtype=np.int64)
            n_seen = 0

            # Iterate validation set in batches
            for i in range(0, len(self.xv), self.bs):
                xb = self.xv[i:i + self.bs]
                # trace_batch() should return { "top_indices": np.ndarray [T,B] }
                tb = trace_batch(self.model, xb, layer_name=self.layer_name)
                top_indices = np.array(tb["top_indices"])  # shape [T, B]

                B = top_indices.shape[1]
                n_seen += B

                for t in range(T):
                    choices = top_indices[t]  # [B]
                    counts = np.bincount(choices, minlength=K)
                    expert_hist[t] += counts
                    steps_hist[t + 1] += B  # mark that B samples reached step t

            total_samples = max(1, steps_hist.sum())
            avg_steps = float(np.sum(np.arange(T + 1) * steps_hist) / total_samples)

            # Print per-step expert usage
            print(f"\n> [{self.layer_name}] epoch {epoch + 1}")
            print(f"  avg_steps={avg_steps:.2f}")
            for t in range(T):
                usage = expert_hist[t]
                total = usage.sum()
                perc = 100 * usage / (total + 1e-6)
                usage_str = "  ".join([f"E{i}:{p:4.1f}%" for i, p in enumerate(perc)])
                print(f"  step {t+1:2d}: total={total:5d}  {usage_str}")



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


# -------- tracing (single sample or batch) --------

def trace_and_predict(
    model,
    x_input,
    y_true=None,
    layer_name="adaptive_router",
    force_full=False,   # kept for API compatibility; unused
):
    """
    Runs the model, and traces per-step expert usage at `layer_name`.
    Returns:
      trace: {
        "top_indices":  [T, B]          (top-1 per step),
        "topk_indices": [T, B, k]       (top-n per step; k = min(top_n, K)),
        "probs":        [T, B, K]       (softmax per step)
      }
      + predictions & some layer info
    """
    layer = model.get_layer(layer_name)
    steps = int(getattr(layer, "steps", 1))
    K = int(layer.K)
    top_n = int(getattr(layer, "top_n", K))

    # features BEFORE the router layer
    pre = keras.Model(model.input, layer.input)
    x_in = tf.convert_to_tensor(x_input)
    x = pre(x_in, training=False)

    B = int(x.shape[0])
    dtype = x.dtype
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]

    # per-step containers
    top1_all   = []
    topk_all   = []
    probs_all  = []

    x_step = x
    for t in range(steps):
        router_out = layer.router(x_step, training=False)
        logits, attn_map = _unpack_router_output(router_out, x_step)

        weights = tf.nn.softmax(logits / layer.route_temp, axis=-1)  # [B,K]
        # Optional exploration (if present)
        explore_eps = float(getattr(layer, "explore_eps", 0.0) or 0.0)
        if explore_eps > 0.0:
            uni = tf.ones_like(weights) / tf.cast(tf.shape(weights)[1], weights.dtype)
            eps = tf.cast(explore_eps, weights.dtype)
            weights = (1.0 - eps) * weights + eps * uni

        # apply attention gain if layer exposes alpha
        scale = tf.cast(H * W, dtype)
        if hasattr(layer, "alpha"):
            alpha = layer.alpha  # use as-is to mirror your layer
            x_mod = x_step * (1.0 + attn_map * scale * alpha)
        else:
            x_mod = x_step

        # run all branches on the modified input
        y_list = [br(x_mod, training=False) for br in layer.branches]   # list of [B,H,W,C]
        y_stack = tf.stack(y_list, axis=1)                              # [B,K,H,W,C]

        if top_n < K:
            y_sel, topk_idx = _mix_topn(weights, y_stack, top_n)        # [B,H,W,C], [B,k]
        else:
            y_sel = tf.einsum('bk,bkhwc->bhwc', weights, y_stack)
            # produce a consistent top-k set (argmax when k==1)
            k_eff = tf.minimum(tf.shape(weights)[1], tf.cast(top_n, tf.int32))
            topk_idx = tf.math.top_k(weights, k=k_eff).indices

        # record trace
        top1 = tf.argmax(weights, axis=-1, output_type=tf.int32)        # [B]
        top1_all.append(top1.numpy())
        topk_all.append(topk_idx.numpy())
        probs_all.append(weights.numpy())

        # next step input
        x_step = y_sel

    # model predictions (full forward)
    pred_probs = model(x_in, training=False).numpy()
    pred_label = pred_probs.argmax(axis=-1).astype(np.int32)
    correct = None
    if y_true is not None:
        y_true_arr = np.asarray(y_true).reshape(-1)
        correct = (pred_label == y_true_arr)

    trace = {
        "top_indices":  np.stack(top1_all, axis=0),   # [T, B]
        "topk_indices": np.stack(topk_all, axis=0),   # [T, B, k]
        "probs":        np.stack(probs_all, axis=0),  # [T, B, K]
    }
    return {
        "trace": trace,
        "pred_probs": pred_probs,
        "pred_label": pred_label,
        "true_label": None if y_true is None else np.asarray(y_true),
        "layer_info": {
            "K": layer.K,
            "steps": steps,
            "top_n": top_n,
            "route_temp": getattr(layer, "route_temp", None),
        },
        "correct": correct,
    }


# def trace_batch(model, x_batch, layer_name="adaptive_router", force_full=False):
#     """
#     Same as trace_and_predict but only returns the trace for a batch.
#     """
#     layer = model.get_layer(layer_name)
#     steps = int(getattr(layer, "steps", 1))
#     K = int(layer.K)
#     top_n = int(getattr(layer, "top_n", K))

#     pre = keras.Model(model.input, layer.input)

#     x_in = tf.convert_to_tensor(x_batch)
#     x = pre(x_in, training=False)

#     B = int(x.shape[0])
#     dtype = x.dtype
#     H = tf.shape(x)[1]
#     W = tf.shape(x)[2]

#     top1_all  = []
#     topk_all  = []
#     probs_all = []

#     x_step = x
#     for t in range(steps):
#         router_out = layer.router(x_step, training=False)
#         logits, attn_map = _unpack_router_output(router_out, x_step)

#         weights = tf.nn.softmax(logits / layer.route_temp, axis=-1)    # [B,K]
#         explore_eps = float(getattr(layer, "explore_eps", 0.0) or 0.0)
#         if explore_eps > 0.0:
#             uni = tf.ones_like(weights) / tf.cast(tf.shape(weights)[1], weights.dtype)
#             eps = tf.cast(explore_eps, weights.dtype)
#             weights = (1.0 - eps) * weights + eps * uni

#         scale = tf.cast(H * W, dtype)
#         if hasattr(layer, "alpha"):
#             alpha = layer.alpha
#             x_mod = x_step * (1.0 + attn_map * scale * alpha)
#         else:
#             x_mod = x_step

#         y_stack = tf.stack([br(x_mod, training=False) for br in layer.branches], axis=1)  # [B,K,H,W,C]
#         if top_n < K:
#             y_sel, topk_idx = _mix_topn(weights, y_stack, top_n)
#         else:
#             y_sel = tf.einsum('bk,bkhwc->bhwc', weights, y_stack)
#             k_eff = tf.minimum(tf.shape(weights)[1], tf.cast(top_n, tf.int32))
#             topk_idx = tf.math.top_k(weights, k=k_eff).indices

#         top1 = tf.argmax(weights, axis=-1, output_type=tf.int32)      # [B]
#         top1_all.append(top1.numpy())
#         topk_all.append(topk_idx.numpy())
#         probs_all.append(weights.numpy())

#         x_step = y_sel

#     return {
#         "top_indices":  np.stack(top1_all, axis=0),   # [T, B]
#         "topk_indices": np.stack(topk_all, axis=0),   # [T, B, k]
#         "probs":        np.stack(probs_all, axis=0),  # [T, B, K]
#     }

def trace_batch(model, x_batch, layer_name="adaptive_router", force_full=False):
    layer = model.get_layer(layer_name)
    steps = int(getattr(layer, "steps", 1))
    K = int(layer.K)

    # features BEFORE the router layer
    pre = tf.keras.Model(model.input, layer.input)
    x_in = tf.convert_to_tensor(x_batch)
    x = pre(x_in, training=False)

    B = int(x.shape[0])
    H = tf.shape(x)[1]; W = tf.shape(x)[2]
    dtype = x.dtype

    top1_all, topk_all, probs_all = [], [], []

    # init mask
    allowed = tf.ones([B, K], dtype=tf.float32)
    last_idx = tf.constant(K - 1, tf.int32)

    x_step = x
    for t in range(steps):
        router_out = layer.router(x_step, training=False)
        # unpack
        if isinstance(router_out, (tuple, list)):
            if len(router_out) == 3:
                logits, _, attn_map = router_out
            elif len(router_out) == 2:
                logits, _ = router_out
                attn_map = tf.zeros_like(x_step[..., :1])
            else:
                logits = router_out[0]
                attn_map = tf.zeros_like(x_step[..., :1])
        else:
            logits = router_out
            attn_map = tf.zeros_like(x_step[..., :1])

        # HARD mask
        neg_inf = tf.constant(-1e9, dtype=logits.dtype)
        masked_logits = tf.where(allowed > 0.5, logits, neg_inf)

        # softmax + optional exploration
        weights = tf.nn.softmax(masked_logits / layer.route_temp, axis=-1)
        explore_eps = float(getattr(layer, "explore_eps", 0.0) or 0.0)
        if explore_eps > 0.0:
            uni = tf.ones_like(weights) / tf.cast(tf.shape(weights)[1], weights.dtype)
            weights = (1.0 - explore_eps) * weights + explore_eps * uni

        top_idx = tf.argmax(weights, axis=-1, output_type=tf.int32)
        k_eff = tf.minimum(tf.shape(weights)[1], tf.cast(getattr(layer, "top_n", K), tf.int32))
        topk_idx = tf.math.top_k(weights, k=k_eff).indices

        # record
        top1_all.append(top_idx.numpy())
        topk_all.append(topk_idx.numpy())
        probs_all.append(weights.numpy())

        # attention gain (same as layer)
        scale = tf.cast(H * W, dtype)
        alpha = layer.alpha if hasattr(layer, "alpha") else tf.constant(0.0, dtype)
        x_mod = x_step * (1.0 + attn_map * scale * alpha)

        # advance x by running ONLY the selected expert (sparse) or your dense path
        # (if your layer is sparse, mirror it here; otherwise, you can do a dense mix)

        # update allowed: disallow chosen unless it's last
        not_last = tf.cast(tf.not_equal(top_idx, last_idx), allowed.dtype)
        idx = tf.stack([tf.range(B), top_idx], axis=1)
        delta = tf.scatter_nd(idx, not_last, tf.shape(allowed))
        allowed = tf.clip_by_value(allowed - delta, 0.0, 1.0)
        # always keep last allowed
        last_vec = tf.one_hot(tf.fill([B], last_idx), depth=K, dtype=allowed.dtype)
        allowed = tf.maximum(allowed, last_vec)

        # If you want to fully mirror the layer’s x update, compute y_sel here as well.
        # For stats only, you can skip advancing x_step; masking still reflects the rule.
        x_step = x_step  # or update like the layer

    return {
        "top_indices":  np.stack(top1_all, axis=0),   # [T, B]
        "topk_indices": np.stack(topk_all, axis=0),   # [T, B, k]
        "probs":        np.stack(probs_all, axis=0),  # [T, B, K]
    }




def print_router_stats(model, x, y, layer_name="adaptive_router", batch_size=512):
    """
    Prints accuracy and per-step expert usage for a (multi-step) router layer.
    Requires `trace_batch` to return:
        {
          "top_indices":  [T, B],   # top-1 expert per step
          "topk_indices": [T, B, k],
          "probs":        [T, B, K]
        }
    """
    # Eval once for accuracy
    loss, acc = model.evaluate(x, y, batch_size=batch_size, verbose=0)

    layer = model.get_layer(layer_name)
    K = int(layer.K)
    T = int(getattr(layer, "steps", 1))

    steps_hist  = np.zeros(T + 1, dtype=np.int64)  # index t+1 counts samples that reached step t
    expert_hist = np.zeros((T, K), dtype=np.int64)
    n_seen = 0

    # Aggregate per-step counts
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        tb = trace_batch(model, xb, layer_name=layer_name)  # uses your multi-step trace
        top1 = tb["top_indices"]  # [T, B]
        T_used, B = top1.shape
        n_seen += B

        for t in range(T_used):
            choices = top1[t]  # [B]
            counts = np.bincount(choices, minlength=K)
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
        perc = 100 * usage / total
        usage_str = "  ".join([f"E{i}:{p:4.1f}%" for i, p in enumerate(perc)])
        print(f"Step {t+1:2d}: total={int(total):5d}  {usage_str}")




def print_trace_for_samples(model, x, y, layer_name="adaptive_router", start=0, end=10):
    """
    Prints routing trace and prediction for samples in the given range (single step).
    """
    print("")
    print(f"================== {layer_name} ====================")
    for i in range(start, end):
        res = trace_and_predict(model, x[i:i+1], y_true=y[i:i+1], layer_name=layer_name)
        trace = res["trace"]
        if trace["top_indices"].shape[0] > 0:
            print(" > pred label:", res["pred_label"][0], "true label:", int(res["true_label"][0]))
            print("   expert (single step):", trace["top_indices"][0, 0])

   