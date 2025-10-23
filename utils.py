import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Conv2DTranspose, MaxPool2D, Add, Conv2D, Dense, Flatten, Dropout, LayerNormalization, 
    DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D
)

def channel_shuffle(x, groups):
    _, width, height, channels = x.shape
    group_ch = channels // groups
    x = layers.Reshape([width, height, group_ch, groups])(x)
    x = layers.Permute([1, 2, 4, 3])(x)
    x = layers.Reshape([width, height, channels])(x)
    return x



class SparseAdaptiveRouterMultiStep(layers.Layer):
    """
    Multi-step router with hard top-n routing per step.
    Sparse execution: only the selected experts are run per sample.
    Constraint: After each step, ONLY the step's top-1 expert is disallowed for later steps,
                EXCEPT the LAST expert (index K-1) which may be reused across steps.
    """
    def __init__(
        self,
        branches,
        steps=3,
        top_n=2,                    # run top-n experts per step
        router_reg=None,
        router_settings=None,
        return_all=False,
        dense_for_diversity=False,  # set True if you keep diversity_tau>0.0
        name=None,
    ):
        super().__init__(name=name)
        router_reg = router_reg or {}
        router_settings = router_settings or {"heads": 2, "dim_head": 64, "mlp_hidden": 0}

        self.branches = branches
        self.K = len(branches)
        self.steps = int(steps)
        self.top_n = int(top_n)
        self.return_all = bool(return_all)
        self.dense_for_diversity = bool(dense_for_diversity)

        if self.top_n < 1 or self.top_n > self.K:
            raise ValueError(f"top_n must be in [1, {self.K}], got {self.top_n}")

        # Router
        self.router = AttnPoolRouter(
            K=self.K,
            dim_head=router_settings.get("dim_head", 64),
            mlp_hidden=router_settings.get("mlp_hidden", 0),
        )

        # Regularization knobs
        self._route_temp   = float(router_reg.get("route_temp",   1.5))
        self.diversity_tau = float(router_reg.get("diversity_tau", 0.0))
        self.explore_eps   = float(router_reg.get("explore_eps",   0.0))
        self.ent_weight    = float(router_reg.get("ent_weight",    0.0))
        self.lb_weight     = float(router_reg.get("lb_weight",     0.0))

        # Learnable scale for attn map
        self.alpha = self.add_weight(
            name="attn_log_alpha",
            shape=(),
            dtype=self.dtype,
            initializer=tf.constant_initializer(0.0),
            trainable=True,
        )

    # Properties
    @property
    def route_temp(self): return self._route_temp
    @route_temp.setter
    def route_temp(self, v: float): self._route_temp = float(v)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            steps=self.steps,
            top_n=self.top_n,
            route_temp=self.route_temp,
            diversity_tau=self.diversity_tau,
            explore_eps=self.explore_eps,
            ent_weight=self.ent_weight,
            lb_weight=self.lb_weight,
            router_settings={
                "heads": self.router.heads,
                "dim_head": self.router.dim_head,
                "mlp_hidden": self.router.mlp_hidden
            },
            K=self.K,
            return_all=self.return_all,
            dense_for_diversity=self.dense_for_diversity,
        ))
        return cfg

    # --- helper: diversity on (optionally) dense outputs ---
    def _diversity_loss(self, y_list):
        """Cosine-sim diversity across experts; y_list is K x [B,H,W,C]."""
        B = tf.shape(y_list[0])[0]
        Y = tf.stack([tf.reshape(y, [B, -1]) for y in y_list], axis=1)   # [B, K, D]
        Y = tf.nn.l2_normalize(Y, axis=-1)
        sims = tf.matmul(Y, Y, transpose_b=True)                         # [B, K, K]
        K_t = tf.shape(Y)[1]
        mask = 1.0 - tf.eye(K_t, dtype=Y.dtype)
        off  = sims * mask
        denom = tf.cast(K_t * (K_t - 1), Y.dtype)
        return tf.reduce_mean(tf.reduce_sum(tf.square(off), axis=[1, 2]) / (denom + 1e-6))

    # --- helper: sparse per-expert execution for top-n (gather → run → weighted scatter-add) ---
    def _run_selected_experts_topn(self, x_mod, topk_idx, weights_masked, training):
        """
        x_mod:         [B,H,W,C]
        topk_idx:      [B, n] int32 indices of selected experts (descending prob)
        weights_masked:[B,K]   soft weights masked to top-n (zeros elsewhere), NOT yet straight-through
        Returns:
            y_mix [B,H,W,C] = sum_e ( w_st[b,e] * expert_e(x_mod[b]) ), but only for selected experts.
        """
        B = tf.shape(x_mod)[0]
        y_out = tf.zeros_like(x_mod)

        # Renormalize masked weights to sum to 1 over the selected set
        eps = tf.constant(1e-9, dtype=weights_masked.dtype)
        denom = tf.reduce_sum(weights_masked, axis=-1, keepdims=True) + eps
        weights_sel = weights_masked / denom  # [B,K]

        # Straight-through on the top-n mask (NOT on the soft values): hard-mask + soft grads
        mask_oh = tf.reduce_sum(tf.one_hot(topk_idx, depth=self.K, dtype=weights_sel.dtype), axis=1)  # [B,K]
        weights_hard = weights_sel * mask_oh
        weights_st = weights_hard + tf.stop_gradient(weights_sel - weights_hard)

        # --- Static loop over all experts; guard with a tf.cond so it works in graph mode ---
        for e, br in enumerate(self.branches):
            e_const = tf.constant(e, dtype=tf.int32)

            # Which samples selected expert e?
            sel_e = mask_oh[:, e_const]  # [B]
            idx = tf.where(sel_e > 0.5)[:, 0]           # [n_e]
            n_e = tf.size(idx)

            def do_expert():
                x_e = tf.gather(x_mod, idx)            # [n_e,H,W,C]
                y_e = br(x_e, training=training)
                w_e = tf.gather(weights_st[:, e_const], idx)  # [n_e]
                w_e = tf.reshape(w_e, [-1, 1, 1, 1])
                y_e = tf.cast(w_e, y_e.dtype) * y_e
                return tf.tensor_scatter_nd_add(
                    y_out,
                    indices=tf.expand_dims(idx, 1),     # [n_e,1] on batch dim
                    updates=y_e
                )
            y_out = tf.cond(n_e > 0, do_expert, lambda: y_out)

        return y_out, weights_sel, weights_st


    # --- forward ---
    def call(self, x, training=None):
        x_dtype = x.dtype
        outputs_all = []

        # Regularizer accumulators (averaged over steps)
        div_acc = tf.constant(0.0, dtype=x_dtype)
        ent_acc = tf.constant(0.0, dtype=x_dtype)
        lb_acc  = tf.constant(0.0, dtype=x_dtype)

        allowed = None  # [B,K] float mask
        last_idx = tf.cast(self.K - 1, tf.int32)

        for t in range(self.steps):
            # Route
            router_out = self.router(x, training=training)
            # Unpack variants
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

            # Initialize allowed on first step
            if allowed is None:
                B = tf.shape(logits)[0]
                allowed = tf.ones([B, self.K], dtype=logits.dtype)

            # Mask disallowed experts
            neg_inf = tf.constant(-1e9, dtype=logits.dtype)
            masked_logits = tf.where(allowed > 0.5, logits, neg_inf)

            # Soft weights
            weights = tf.nn.softmax(masked_logits / self.route_temp, axis=-1)  # [B,K]
            if self.explore_eps > 0.0 and training:
                uni = tf.ones_like(weights) / tf.cast(tf.shape(weights)[1], weights.dtype)
                eps = tf.cast(self.explore_eps, weights.dtype)
                weights = (1.0 - eps) * weights + eps * uni

            # Top-n selection per sample
            k = tf.minimum(self.top_n, tf.shape(weights)[1])
            top_vals, top_idx = tf.math.top_k(weights, k=k, sorted=True)       # [B,n], [B,n]

            # Build masked weights (zero outside top-n)
            mask_oh = tf.reduce_sum(tf.one_hot(top_idx, depth=self.K, dtype=weights.dtype), axis=1)  # [B,K]
            weights_masked = weights * mask_oh                                                     # [B,K]

            # Attention gain
            scale = tf.cast(tf.shape(x)[1] * tf.shape(x)[2], x_dtype)
            alpha = tf.nn.softplus(self.alpha)
            x_mod = x * (1.0 + attn_map * scale * alpha)

            # === Sparse execution: run top-n experts and mix ===
            y_mix, weights_sel, weights_st = self._run_selected_experts_topn(
                x_mod, top_idx, weights_masked, training=training
            )  # [B,H,W,C], [B,K], [B,K]

            # Advance state
            x = y_mix
            if self.return_all:
                outputs_all.append(x)

            # Update allowed-mask:
            # Disallow ONLY the current step's top-1 expert for each sample,
            # EXCEPT the last expert (index K-1), which remains allowed.
            B = tf.shape(top_idx)[0]
            top1_idx = top_idx[:, 0]  # [B]
            not_last = tf.cast(tf.not_equal(top1_idx, last_idx), allowed.dtype)  # 1 if we should block it
            indices = tf.stack([tf.range(B), top1_idx], axis=1)                  # [B,2]
            delta = tf.scatter_nd(indices, not_last, tf.shape(allowed))
            allowed = tf.clip_by_value(allowed - delta, 0.0, 1.0)
            # Always (re)allow last expert
            last_vec = tf.one_hot(tf.fill([B], last_idx), depth=self.K, dtype=allowed.dtype)
            allowed = tf.maximum(allowed, last_vec)

            # --- Regularizers ---
            if training:
                if self.ent_weight > 0.0:
                    ent = -tf.reduce_mean(tf.reduce_sum(
                        weights * tf.math.log(tf.clip_by_value(weights, 1e-8, 1.0)), axis=-1))
                    ent_acc += ent
                if self.lb_weight > 0.0:
                    p_mean = tf.reduce_mean(weights, axis=0)              # [K]
                    uni = tf.fill(tf.shape(p_mean), 1.0 / self.K)
                    lb_loss = tf.reduce_sum((p_mean - uni) ** 2)
                    lb_acc += lb_loss
                if self.diversity_tau > 0.0:
                    if self.dense_for_diversity:
                        y_list = [br(x_mod, training=training) for br in self.branches]
                        y_stack = tf.stack(y_list, axis=1)  # [B,K,H,W,C]
                        y_st = tf.einsum('bk,bkhwc->bhwc', weights_st, y_stack)
                        x = x + (y_st - tf.stop_gradient(y_st))  # surrogate gradient path
                        div_acc += self._diversity_loss(y_list)
                    else:
                        div_acc += 0.0  # cheap approx: skip

        # Add averaged losses
        if training:
            steps_f = tf.cast(self.steps, x_dtype)
            if self.diversity_tau > 0.0:
                self.add_loss(self.diversity_tau * (div_acc / steps_f))
            if self.ent_weight > 0.0:
                self.add_loss(self.ent_weight * (ent_acc / steps_f))
            if self.lb_weight > 0.0:
                self.add_loss(self.lb_weight * (lb_acc / steps_f))

        return outputs_all if self.return_all else x



class AdaptiveRouterMultiStep(layers.Layer):
    """
    Multi-step router with hard top-n routing per step (dense compute).
    Forward uses only the selected top-n experts (renormalized); all experts
    are still computed each step to support the diversity loss.
    No constraint on reusing experts across steps.
    """
    def __init__(
        self,
        branches,
        steps=3,
        top_n=2,
        router_reg=None,
        router_settings=None,
        return_all=False,
        name=None
    ):
        super().__init__(name=name)
        router_reg = router_reg or {}
        router_settings = router_settings or {"heads": 2, "dim_head": 64, "mlp_hidden": 0}

        self.branches = branches
        self.K = len(branches)
        self.steps = int(steps)
        self.top_n = int(top_n)
        if self.top_n < 1 or self.top_n > self.K:
            raise ValueError(f"top_n must be in [1, {self.K}], got {self.top_n}")
        self.return_all = bool(return_all)

        # Router
        if router_settings.get("sr_ratio", -1) < 0:
            self.router = AttnPoolRouter(
                K=self.K,
                dim_head=router_settings.get("dim_head", 64),
                mlp_hidden=router_settings.get("mlp_hidden", 0),
            )
        else: 
            self.router = EffAttnPoolRouter(
                K=self.K,
                dim_head=router_settings.get("dim_head", 64),
                mlp_hidden=router_settings.get("mlp_hidden", 0),
                sr_rationo=router_settings.get("sr_ratio", 1),
            )

        # Regularization knobs
        self._route_temp   = float(router_reg.get("route_temp",   1.5))
        self.diversity_tau = float(router_reg.get("diversity_tau", 0.0))

        # Learnable scale for attn map
        self.alpha = self.add_weight(
            name="attn_log_alpha",
            shape=(),
            dtype=self.dtype,
            initializer=tf.constant_initializer(0.0),
            trainable=True,
        )

    @property
    def route_temp(self): return self._route_temp
    @route_temp.setter
    def route_temp(self, v: float): self._route_temp = float(v)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            steps=self.steps,
            top_n=self.top_n,
            route_temp=self.route_temp,
            diversity_tau=self.diversity_tau,
            router_settings={
                "heads": self.router.heads,
                "dim_head": self.router.dim_head,
                "mlp_hidden": self.router.mlp_hidden
            },
            K=self.K,
            return_all=self.return_all
        ))
        return cfg

    # -------- helpers ----------
    def _diversity_loss(self, y_list):
        """Cosine-sim diversity across branches."""
        B = tf.shape(y_list[0])[0]
        Y = tf.stack([tf.reshape(y, [B, -1]) for y in y_list], axis=1)  # [B, K, D]
        Y = tf.nn.l2_normalize(Y, axis=-1)
        sims = tf.matmul(Y, Y, transpose_b=True)          # [B, K, K]
        K_t = tf.shape(Y)[1]
        mask = 1.0 - tf.eye(K_t, dtype=Y.dtype)
        off  = sims * mask
        denom = tf.cast(K_t * (K_t - 1), Y.dtype)
        return tf.reduce_mean(tf.reduce_sum(tf.square(off), axis=[1, 2]) / (denom + 1e-6))

    # -------- forward ----------
    def call(self, x, training=None):
        x_dtype = x.dtype
        outputs_all = []

        # Regularizer accumulators (averaged over steps)
        div_acc = tf.constant(0.0, dtype=x_dtype)

        for _ in range(self.steps):
            # Route (supports variants; default-zeros attn_map if not provided)
            logits, _, attn_map = self.router(x, training=training)

            # Soft weights (+ exploration)
            weights = tf.nn.softmax(logits / self.route_temp, axis=-1)  # [B,K]

            # Top-n selection per sample
            k = tf.minimum(self.top_n, tf.shape(weights)[1])
            _, top_idx = tf.math.top_k(weights, k=k, sorted=True)       # [B,n]

            # Build masked weights (zero outside top-n), then renormalize over selected set
            top_mask = tf.reduce_sum(tf.one_hot(top_idx, depth=self.K, dtype=weights.dtype), axis=1)  # [B,K]
            weights_masked = weights * top_mask
            eps = tf.constant(1e-9, dtype=weights.dtype)
            denom = tf.reduce_sum(weights_masked, axis=-1, keepdims=True) + eps
            weights_sel = weights_masked / denom  # [B,K], sums to 1 over selected experts

            # Straight-through on the selection mask (forward: masked; backward: soft)
            weights_st = weights_sel + tf.stop_gradient(weights - weights_sel)

            # Attention gain
            scale = tf.cast(tf.shape(x)[1] * tf.shape(x)[2], x_dtype)
            alpha = tf.nn.softplus(self.alpha)
            x_mod = x * (1.0 + attn_map * scale * alpha)

            # Compute ALL branch outputs (dense) for diversity + optional ST path
            y_list  = [br(x_mod, training=training) for br in self.branches]   # K × [B,H,W,C]
            y_stack = tf.stack(y_list, axis=1)                                  # [B,K,H,W,C]

            # Forward uses only top-n via masked/renormalized weights; grads pass through soft weights
            y_mix = tf.einsum('bk,bkhwc->bhwc', weights_st, y_stack)

            # Next state
            x = y_mix
            if self.return_all:
                outputs_all.append(x)

            # ---- Regularizers (accumulate per-step) ----
            if training:
                if self.diversity_tau > 0.0:
                    div_acc += self._diversity_loss(y_list)

        # Add averaged losses
        if training:
            steps_f = tf.cast(self.steps, x_dtype)
            if self.diversity_tau > 0.0:
                self.add_loss(self.diversity_tau * (div_acc / steps_f))

        return outputs_all if self.return_all else x




class AdaptiveRouter(layers.Layer):
    def __init__(
        self,
        branches,
        top_n=1,
        router_reg={},
        router_settings={"heads": 2, "dim_head": 64, "mlp_hidden": 0},
        name=None
    ):
        super().__init__(name=name)
        self.branches = branches
        self.K = len(branches)
        self.top_n = top_n if top_n > 0 else len(branches)
        self.router = AttnPoolRouter(
            K=self.K,
            dim_head=router_settings.get("dim_head", 64),
            mlp_hidden=router_settings.get("mlp_hidden", 0),
        )
        self._route_temp  = router_reg.get("route_temp", 1.5) 
        self.diversity_tau = router_reg.get("diversity_tau", 0.0) 
        self.explore_eps   = router_reg.get("explore_eps", 0.0) 
        self.ent_weight    = router_reg.get("ent_weight", 0.0) 
        self.alpha = self.add_weight(
            name="attn_log_alpha",
            shape=(),
            dtype=self.dtype,
            initializer=tf.constant_initializer(0),
            trainable=True,
        )
        
    @property
    def route_temp(self): return self._route_temp
    @route_temp.setter
    def route_temp(self, v: float): self._route_temp = float(v)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            route_temp=self.route_temp,
            diversity_tau=self.diversity_tau,
            explore_eps=self.explore_eps,      # ← new
            ent_weight=self.ent_weight,        # ← new
            router_settings={
                "heads": self.router.heads,
                "dim_head": self.router.dim_head,
                "mlp_hidden": self.router.mlp_hidden
            },
            K=self.K,
            top_n=self.top_n
        ))
        return cfg

    def call(self, x, training=None):
        logits, ctx, attn_map = self.router(x, training=training)
        weights = tf.nn.softmax(logits / self.route_temp, axis=-1)  
        scale = tf.cast(tf.shape(x)[1] * tf.shape(x)[2], x.dtype)      
        x = x * (1.0 + attn_map * scale * self.alpha)
        
        # Branches
        y_list  = [br(x, training=training) for br in self.branches]  # K x [B,H,W,C]

        B = tf.shape(x)[0]
        Y = tf.stack([tf.reshape(y, [B, -1]) for y in y_list], axis=1)   # [B, K, D]
        Y = tf.nn.l2_normalize(Y, axis=-1)                               # unit vectors along D
        sims = tf.matmul(Y, Y, transpose_b=True)
        K_t = tf.shape(Y)[1]
        mask = 1.0 - tf.eye(K_t, dtype=Y.dtype)                          # [K, K]
        off = sims * mask                                                 # zeros on diag
        denom = tf.cast(K_t * (K_t - 1), Y.dtype)                        # number of off-diagonal pairs
        div_loss = tf.reduce_mean(tf.reduce_sum(tf.square(off), axis=[1, 2]) / (denom + 1e-6))

        y_stack = tf.stack(y_list, axis=1) 

        if self.top_n < self.K:
            K = tf.shape(weights)[1]
            k = tf.minimum(tf.constant(self.top_n, dtype=tf.int32), K)         # choose 2
            topk = tf.math.top_k(weights, k=k)
            mask = tf.reduce_sum(tf.one_hot(topk.indices, K, dtype=weights.dtype), axis=1)  # [B,K]
            sparse_w = tf.where(mask > 0, weights, tf.zeros_like(weights))
            sparse_w = sparse_w / tf.reduce_sum(sparse_w, axis=-1, keepdims=True)
            y_sel = tf.einsum('bk,bkhwc->bhwc', sparse_w, y_stack)
        else:                                
            y_sel = tf.einsum('bk,bkhwc->bhwc', weights, y_stack)             # soft mixture

        # regularizers
        if training:
            if self.diversity_tau > 0.0:
                self.add_loss(self.diversity_tau * div_loss)
            if getattr(self, "ent_weight", 0.0) > 0.0:
                ent = -tf.reduce_mean(tf.reduce_sum(weights * tf.math.log(tf.clip_by_value(weights, 1e-8, 1.0)), axis=-1))
                self.add_loss(self.ent_weight * ent)
            if getattr(self, "lb_weight", 0.0) > 0.0:
                p_mean = tf.reduce_mean(weights, axis=0)
                uni = tf.fill(tf.shape(p_mean), 1.0 / self.K)
                lb_loss = tf.reduce_sum((p_mean - uni)**2)
                self.add_loss(self.lb_weight * lb_loss)

        return y_sel


# ---------------------------------------------------------
# Softmax Router (no Gumbel). Optional hard mode at inference.
# ---------------------------------------------------------
class SoftmaxRouter(layers.Layer):
    def __init__(self, num_choices, hard_at_inference=False, **kwargs):
        super().__init__(**kwargs)
        self.num_choices = num_choices
        self.hard_at_inference = hard_at_inference
        self.logits_layer = Dense(num_choices)

    def call(self, features, training=None):
        logits = self.logits_layer(GlobalAveragePooling2D()(features))  # (B, K)
        if training or not self.hard_at_inference:
            probs = tf.nn.softmax(logits, axis=-1)                      # (B, K)
        else:
            idx = tf.argmax(logits, axis=-1)
            probs = tf.one_hot(idx, depth=self.num_choices, dtype=tf.float32)
        return probs  # (B, K)


class ResBlock(layers.Layer):
    def __init__(self, layers, **kw):
        super().__init__(**kw)
        self.layers = layers
        self.add = Add()

    def call(self, x, training=None):
        shortcut = x
        for layer in self.layers:
            x = layer(x)
        x = self.add([shortcut, x])
        return x


# ---------------------------
# Tiny conv stem
# ---------------------------
class ConvStem(layers.Layer):
    def __init__(self, out_ch, kernel_size, groups=1, **kw):
        super().__init__(**kw)
        self.conv = GroupConv2D(3, out_ch, kernel_size, groups=groups)
        #layers.Conv2D(out_ch, kernel_size, padding="same", use_bias=True)
        self.norm = layers.LayerNormalization()
        self.act  = layers.Activation("swish")

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.norm(x, training=training)
        x = self.act(x)
        return x


class DepthwiseConvStem(layers.Layer):
    def __init__(self, out_ch, kernel_size, **kw):
        super().__init__(**kw)
        self.conv = DepthwiseConv2D(kernel_size, padding="same",  use_bias=True, depth_multiplier=4)
        self.pointwise = layers.Conv2D(out_ch, 1, padding="same", use_bias=True)
        self.norm = layers.LayerNormalization()
        self.act  = layers.Activation("swish")

    def call(self, x, training=None):
        x = self.pointwise(self.conv(x))
        x = self.norm(x, training=training)
        x = self.act(x)
        return x

class PatchConvStem(layers.Layer):
    def __init__(self, filters, **kw):
        super().__init__(**kw)
        self.conv = layers.Conv2D(filters, 4, padding="same", stride=4, use_bias=True)
        self.norm = layers.LayerNormalization()
        self.act  = layers.Activation("swish")

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.norm(x, training=training)
        x = self.act(x)
        return x
    

class EffAttnPoolRouter(layers.Layer):
    def __init__(
        self,
        K,
        heads=2,
        dim_head=64,
        mlp_hidden=64,
        sr_ratio=1,            # NEW: spatial reduction (>=1). 1 = no downsample.
        use_sdp=True,          # NEW: try tf.nn.scaled_dot_product_attention
        **kw
    ):
        super().__init__(**kw)
        self.K = int(K)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.mlp_hidden = int(mlp_hidden)
        self.sr_ratio = int(sr_ratio)
        self.use_sdp = bool(use_sdp)

        # Per-head query vectors (learned), shape [H, D]
        self.q = self.add_weight(
            name="queries",
            shape=(self.heads, self.dim_head),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Fused key/value projection: one 1x1 conv, then split -> [B,H,W,2*H*D]
        self.kv_proj = layers.Conv2D(
            filters=2 * self.heads * self.dim_head,
            kernel_size=1,
            use_bias=False,
            kernel_initializer="glorot_uniform",
        )

        # Optional spatial reduction before attention
        if self.sr_ratio > 1:
            self.sr = layers.AveragePooling2D(pool_size=self.sr_ratio, strides=self.sr_ratio, padding="valid")
        else:
            self.sr = None

        # Head aggregator -> K logits
        if self.mlp_hidden > 0:
            self.head_mlp = keras.Sequential([
                layers.Dense(self.mlp_hidden, activation="swish", use_bias=False, kernel_initializer="glorot_uniform"),
                layers.Dense(self.K, use_bias=False, kernel_initializer="glorot_uniform"),
            ])
        else:
            self.head_mlp = layers.Dense(self.K, use_bias=False, kernel_initializer="glorot_uniform")

    def _attend_manual(self, q_hd, k_bhld, v_bhld, x_dtype):
        """
        Manual attention path:
          q_hd:    [H, D]
          k_bhld:  [B, H, L, D]
          v_bhld:  [B, H, L, D]
        Returns:
          pooled:  [B, H, D]
          attn_w:  [B, H, L]
        """
        scale = tf.math.rsqrt(tf.cast(self.dim_head, x_dtype))        # 1/sqrt(D)
        # scores: [B, H, L] = (k · q) / sqrt(D)
        # einsum 'hd,bhld->bhl' computes dot(k,q) per head and location
        scores = tf.einsum('hd,bhld->bhl', q_hd, k_bhld) * scale
        attn_w = tf.nn.softmax(scores, axis=-1)                       # [B,H,L]
        # pooled per head: sum_l w * v  => [B, H, D]
        pooled = tf.einsum('bhl,bhld->bhd', attn_w, v_bhld)
        return pooled, attn_w

    def _attend_sdp(self, q_hd, k_bhld, v_bhld, x_dtype):
        """
        Flash/SDPA path if available (TF >= 2.12 w/ CUDA kernels).
        We still need the weights for attn_map, so we compute scores manually for that;
        SDPA gives the output efficiently.
        """
        B = tf.shape(k_bhld)[0]
        L = tf.shape(k_bhld)[2]
        # Build batched query: [B, H, 1, D] by broadcasting learned q
        q_bh1d = tf.broadcast_to(q_hd[None, :, None, :], [B, self.heads, 1, self.dim_head])
        # SDPA output: [B, H, 1, D]
        pooled = tf.nn.scaled_dot_product_attention(q_bh1d, k_bhld, v_bhld, is_causal=False)
        pooled = tf.squeeze(pooled, axis=2)  # [B,H,D]

        # For attn_map we still need weights: compute light scores (cheap)
        scale = tf.math.rsqrt(tf.cast(self.dim_head, x_dtype))
        scores = tf.einsum('hd,bhld->bhl', q_hd, k_bhld) * scale      # [B,H,L]
        attn_w = tf.nn.softmax(scores, axis=-1)                       # [B,H,L]
        return pooled, attn_w

    def call(self, x, training=None):
        x_dtype = x.dtype
        B = tf.shape(x)[0]
        Hs = tf.shape(x)[1]
        Ws = tf.shape(x)[2]

        # 1) Fused K/V projection
        kv = self.kv_proj(x)                            # [B,Hs,Ws, 2*H*D]
        k, v = tf.split(kv, num_or_size_splits=2, axis=-1)  # [B,Hs,Ws,H*D] each

        # 2) Optional spatial reduction
        if self.sr is not None:
            k = self.sr(k)                              # [B,Hs',Ws',H*D]
            v = self.sr(v)
        Hr = tf.shape(k)[1]
        Wr = tf.shape(k)[2]
        L = Hr * Wr                                     # sequence length per head

        # 3) Reshape to [B, H, L, D] without extra transposes
        #    NHWC -> [B, L, H, D] -> [B, H, L, D]
        def _reshape_bhld(t):
            t = tf.reshape(t, [B, Hr * Wr, self.heads, self.dim_head])   # [B,L,H,D]
            return tf.transpose(t, [0, 2, 1, 3])                         # [B,H,L,D]
        k_bhld = _reshape_bhld(k)
        v_bhld = _reshape_bhld(v)

        # 4) Choose attention path
        q_hd = self.q  # [H, D]
        if self.use_sdp:
            try:
                pooled, attn_w = self._attend_sdp(q_hd, k_bhld, v_bhld, x_dtype)
            except Exception:
                pooled, attn_w = self._attend_manual(q_hd, k_bhld, v_bhld, x_dtype)
        else:
            pooled, attn_w = self._attend_manual(q_hd, k_bhld, v_bhld, x_dtype)

        # 5) Aggregate heads -> logits
        # pooled: [B,H,D] -> flatten heads
        pooled_flat = tf.reshape(pooled, [B, self.heads * self.dim_head])   # [B, H*D]
        logits = self.head_mlp(pooled_flat, training=training)              # [B, K]

        # 6) Return spatial attention map averaged over heads, upsample/back-project to input HxW
        # attn_w: [B,H,L] over reduced grid -> average heads -> [B,L] -> [B,Hr,Wr,1]
        attn_map = tf.reduce_mean(attn_w, axis=1, keepdims=False)           # [B,L]
        attn_map = tf.reshape(attn_map, [B, Hr, Wr, 1])                     # [B,Hr,Wr,1]
        if self.sr is not None and (self.sr_ratio > 1):
            # Simple nearest/bilinear upsample back to input spatial size
            attn_map = tf.image.resize(attn_map, size=[Hs, Ws], method="bilinear")

        return logits, pooled_flat, attn_map


# ---------------------------
# Multi-head attention pooling router
# Produces logits over K experts
# ---------------------------
class AttnPoolRouter(layers.Layer):
    def __init__(self, K, heads=2, dim_head=64, mlp_hidden=64, **kw):
        super().__init__(**kw)
        self.K = int(K)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.mlp_hidden = int(mlp_hidden)

        self.q = self.add_weight(
            name="queries", shape=(self.heads, self.dim_head),
            initializer="glorot_uniform", trainable=True)

        self.key_proj = layers.Conv2D(self.heads*self.dim_head, 1, use_bias=False, kernel_initializer="glorot_uniform")
        self.val_proj = layers.Conv2D(self.heads*self.dim_head, 1, use_bias=False, kernel_initializer="glorot_uniform")

        # Head aggregator -> K logits
        if self.mlp_hidden > 0:
            self.head_mlp = keras.Sequential([
                layers.Dense(self.mlp_hidden, activation="swish", use_bias=False, kernel_initializer="glorot_uniform"),
                layers.Dense(self.K, use_bias=False, kernel_initializer="glorot_uniform")
            ])
        else:
            self.head_mlp = layers.Dense(self.K, use_bias=False, kernel_initializer="glorot_uniform")

    def call(self, x, training=None):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        k = self.key_proj(x)                                 # [B,H,W,heads*dim]
        v = self.val_proj(x)                                 # [B,H,W,heads*dim]
        k = tf.reshape(k, [B, H*W, self.heads, self.dim_head])
        v = tf.reshape(v, [B, H*W, self.heads, self.dim_head])
        k = tf.transpose(k, [0,2,1,3])                       # [B,heads,HW,dim]
        v = tf.transpose(v, [0,2,1,3])                       # [B,heads,HW,dim]

        q = tf.expand_dims(tf.expand_dims(self.q, 0), 2)     # [1,heads,1,dim] -> [B,heads,1,dim] broadcast
        attn = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.dim_head, x.dtype))  # [B,heads,1,HW]
        attn = tf.nn.softmax(attn, axis=-1)                  # per-head spatial weights

        pooled = tf.matmul(attn, v)                          # [B,heads,1,dim]
        pooled = tf.squeeze(pooled, axis=2)                  # [B,heads,dim]
        pooled_flat = tf.reshape(pooled, [B, self.heads*self.dim_head])

        logits = self.head_mlp(pooled_flat, training=training)  # [B,K]

        # --- new: spatial attention map (averaged across heads) ---
        attn_map = tf.reduce_mean(attn, axis=1)              # [B,1,HW]
        attn_map = tf.reshape(attn_map, [B, H, W, 1])        # [B,H,W,1]
        return logits, pooled_flat, attn_map


class FeatureModulator(layers.Layer):
    def __init__(self, channels, hidden=128):
        super().__init__()
        self.fc = keras.Sequential([
            layers.Dense(hidden, activation="swish"),
            layers.Dense(channels * 2)  # gamma and beta
        ])
    def call(self, ctx, x):
        # ctx: [B, heads*dim_head]
        B = tf.shape(x)[0]
        C = tf.shape(x)[-1]
        gb = self.fc(ctx)                          # [B, 2C]
        gamma, beta = tf.split(gb, 2, axis=-1)     # [B,C], [B,C]
        gamma = tf.reshape(gamma, [B,1,1,C])
        beta  = tf.reshape(beta,  [B,1,1,C])
        return x * (1.0 + gamma) + beta


class PoolingLayer(keras.layers.Layer):

    def __init__(self, filters, frac_ratio=1.0 ,groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups
        self.frac_ratio = frac_ratio

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        config.update({"frac_ratio": self.frac_ratio})
        return config

    def build(self, input_shape):
        self.depth = input_shape[-1]
        if self.frac_ratio == 2.0: self.pool = MaxPool2D(pool_size=(2, 2)) #AveragePooling2D(pool_size=(2, 2))
        self.channel_up_conv = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.channel_up_norm = LayerNormalization()
        self.channel_up_swish = layers.Activation("swish")

    def call(self, input):
        x = input
        if self.frac_ratio < 2.0 and self.frac_ratio > 0.0: x = tf.nn.fractional_max_pool(value=x, pooling_ratio=[1, self.frac_ratio, self.frac_ratio, 1], pseudo_random=True, overlapping=False)[0]
        elif self.frac_ratio == 2.0: x = self.pool(x) 
        x = self.channel_up_conv(x)
        x = self.channel_up_norm(x)
        x = self.channel_up_swish(x)
        
        return x
    

class MaxPoolingLayer(keras.layers.Layer):

    def __init__(self, filters,groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        return config

    def build(self, input_shape):
        self.pool = MaxPool2D(pool_size=(2, 2)) 
        self.channel_up_conv = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.channel_up_norm = LayerNormalization()
        self.channel_up_swish = layers.Activation("swish")

    def call(self, x):
        x = self.pool(x) 
        x = self.channel_up_conv(x)
        x = self.channel_up_norm(x)
        x = self.channel_up_swish(x)
        return x
    

class AvgPoolingLayer(keras.layers.Layer):

    def __init__(self, filters,groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        return config

    def build(self, input_shape):
        self.pool = AveragePooling2D(pool_size=(2, 2)) 
        self.channel_up_conv = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.channel_up_norm = LayerNormalization()
        self.channel_up_swish = layers.Activation("swish")

    def call(self, x):
        x = self.pool(x) 
        x = self.channel_up_conv(x)
        x = self.channel_up_norm(x)
        x = self.channel_up_swish(x)
        return x
    

class Conv3x3PoolingLayer(keras.layers.Layer):

    def __init__(self, filters,groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        return config

    def build(self, input_shape):
        self.depth = input_shape[-1]
        self.channel_up_conv = GroupConv2D(input_channels=self.depth, output_channels=self.filters, kernel_size=[3, 3], padding='same', strides=2, groups=self.groups)
        self.channel_up_norm = LayerNormalization()
        self.channel_up_swish = layers.Activation("swish")

    def call(self, x):
        x = self.channel_up_conv(x)
        x = self.channel_up_norm(x)
        x = self.channel_up_swish(x)
        return x
    

class Conv5x5PoolingLayer(keras.layers.Layer):

    def __init__(self, filters,groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        return config

    def build(self, input_shape):
        self.depth = input_shape[-1]
        self.channel_up_conv = GroupConv2D(input_channels=self.depth, output_channels=self.filters, kernel_size=[5, 5], padding='same', strides=2, groups=self.groups)
        self.channel_up_norm = LayerNormalization()
        self.channel_up_swish = layers.Activation("swish")

    def call(self, x):
        x = self.channel_up_conv(x)
        x = self.channel_up_norm(x)
        x = self.channel_up_swish(x)
        return x
    

class Depthwise3x3ConvPoolingLayer(keras.layers.Layer):

    def __init__(self, filters,groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        return config

    def build(self, input_shape):
        self.depth = input_shape[-1]
        self.depthwise_pool = DepthwiseConv2D(kernel_size=[3, 3], padding='same', strides=2)
        self.channel_up_conv = Conv2D(self.filters, kernel_size=[1, 1], padding='same', groups=self.groups)
        self.channel_up_norm = LayerNormalization()
        self.channel_up_swish = layers.Activation("swish")

    def call(self, x):
        x = self.depthwise_pool(x)
        x = self.channel_up_conv(x)
        x = self.channel_up_norm(x)
        x = self.channel_up_swish(x)
        return x
    

class Depthwise5x5ConvPoolingLayer(keras.layers.Layer):

    def __init__(self, filters, groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        return config

    def build(self, input_shape):
        self.depth = input_shape[-1]
        self.depthwise_pool = DepthwiseConv2D(kernel_size=[5, 5], padding='same', strides=2)
        self.channel_up_conv = Conv2D(self.filters, kernel_size=[1, 1], padding='same', groups=self.groups)
        self.channel_up_norm = LayerNormalization()
        self.channel_up_swish = layers.Activation("swish")

    def call(self, x):
        x = self.depthwise_pool(x)
        x = self.channel_up_conv(x)
        x = self.channel_up_norm(x)
        x = self.channel_up_swish(x)
        return x
    

class Depthwise7x7ConvPoolingLayer(keras.layers.Layer):

    def __init__(self, filters,groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        return config

    def build(self, input_shape):
        self.depth = input_shape[-1]
        self.depthwise_pool = DepthwiseConv2D(kernel_size=[7, 7], padding='same', strides=2)
        self.channel_up_conv = Conv2D(self.filters, kernel_size=[1, 1], padding='same', groups=self.groups)
        self.channel_up_norm = LayerNormalization()
        self.channel_up_swish = layers.Activation("swish")

    def call(self, x):
        x = self.depthwise_pool(x)
        x = self.channel_up_conv(x)
        x = self.channel_up_norm(x)
        x = self.channel_up_swish(x)
        return x

# ---------------------------------------------------------
# Your GroupConv2D (unchanged)
# ---------------------------------------------------------
class GroupConv2D(layers.Layer):
    def __init__(self, input_channels, output_channels, kernel_size=(3,3), padding='same', groups=1, strides=1, **kwargs):
        super().__init__(**kwargs)
        assert input_channels % groups == 0, "in_ch must be divisible by groups"
        assert output_channels % groups == 0, "out_ch must be divisible by groups"
        self.in_ch = input_channels
        self.out_ch = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.convs = []
        self.strides = strides

    def build(self, input_shape):
        for _ in range(self.groups):
            self.convs.append(
                Conv2D(self.out_ch // self.groups, self.kernel_size, padding=self.padding, strides=self.strides)
            )

    def call(self, x):
        splits = tf.split(x, num_or_size_splits=self.groups, axis=-1)
        outs = [conv(s) for conv, s in zip(self.convs, splits)]
        return tf.concat(outs, axis=-1)

# ---------------------------------------------------------
# Your ResidualBlock (as provided; VALID 3x3s and projection)
# ---------------------------------------------------------
class ResidualBlock3x3(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(3, 3), residual=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        self.residual = residual
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        config.update({"residual": self.residual})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = GroupConv2D(input_channels=self.filters//self.block_reduction,
                                 output_channels=self.filters//self.block_reduction, 
                                 kernel_size=[3, 3], 
                                 padding='same',
                                 groups=self.groups)
        self.norm2 = LayerNormalization()
        self.conv3 = GroupConv2D(input_channels=self.filters//self.block_reduction,
                                 output_channels=self.filters//self.block_reduction, 
                                 kernel_size=[3, 3], 
                                 padding='same',
                                 groups=self.groups) 
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        # NOTE: your projection shrinks spatial dims via 5x5 VALID; assumed to match main path.
        #self.projection = DepthwiseConv2D(kernel_size=[5, 5], padding='valid')
        #self.proj_norm = LayerNormalization()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    

class ResidualBlockDepthwise3x3(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = DepthwiseConv2D(kernel_size=[3, 3], padding='same')
        self.norm2 = LayerNormalization()
        self.conv3 = DepthwiseConv2D(kernel_size=[3, 3], padding='same')
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y

class ResidualBlockDepthwise5x5(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(5, 5), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = DepthwiseConv2D(kernel_size=[5, 5], padding='same')
        self.norm2 = LayerNormalization()
        self.conv3 = DepthwiseConv2D(kernel_size=[5, 5], padding='same')
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    
class ResidualBlockDepthwise7x7(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(7, 7), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = DepthwiseConv2D(kernel_size=[7, 7], padding='same')
        self.norm2 = LayerNormalization()
        self.conv3 = DepthwiseConv2D(kernel_size=[7, 7], padding='same')
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    
class ResidualBlockDepthwise9x9(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(9, 9), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = DepthwiseConv2D(kernel_size=[9, 9], padding='same')
        self.norm2 = LayerNormalization()
        self.conv3 = DepthwiseConv2D(kernel_size=[9, 9], padding='same')
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    

class ResidualBlock5x5(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(5, 5), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // (self.block_reduction*2), kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = GroupConv2D(input_channels=self.filters//(self.block_reduction*2),
                                 output_channels=self.filters//(self.block_reduction*2), 
                                 kernel_size=[5, 5], 
                                 padding='same',
                                 groups=self.groups)
        self.norm2 = LayerNormalization()
        self.conv3 = GroupConv2D(input_channels=self.filters//(self.block_reduction*2),
                                 output_channels=self.filters//(self.block_reduction*2), 
                                 kernel_size=[5, 5], 
                                 padding='same',
                                 groups=self.groups) 
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        # NOTE: your projection shrinks spatial dims via 5x5 VALID; assumed to match main path.
        #self.projection = DepthwiseConv2D(kernel_size=[5, 5], padding='valid')
        #self.proj_norm = LayerNormalization()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    
class ResidualBlock7x7(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // (self.block_reduction * 4), kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = GroupConv2D(input_channels=self.filters//(self.block_reduction * 4),
                                 output_channels=self.filters//(self.block_reduction * 4), 
                                 kernel_size=[7, 7], 
                                 padding='same',
                                 groups=self.groups)
        self.norm2 = LayerNormalization()
        self.conv3 = GroupConv2D(input_channels=self.filters//(self.block_reduction * 4),
                                 output_channels=self.filters//(self.block_reduction * 4), 
                                 kernel_size=[7, 7], 
                                 padding='same',
                                 groups=self.groups) 
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        # NOTE: your projection shrinks spatial dims via 5x5 VALID; assumed to match main path.
        #self.projection = DepthwiseConv2D(kernel_size=[5, 5], padding='valid')
        #self.proj_norm = LayerNormalization()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    
 
class ResidualBlock7x7(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // (self.block_reduction * 4), kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = GroupConv2D(input_channels=self.filters//(self.block_reduction * 4),
                                 output_channels=self.filters//(self.block_reduction * 4), 
                                 kernel_size=[7, 7], 
                                 padding='same',
                                 groups=self.groups)
        self.norm2 = LayerNormalization()
        self.conv3 = GroupConv2D(input_channels=self.filters//(self.block_reduction * 4),
                                 output_channels=self.filters//(self.block_reduction * 4), 
                                 kernel_size=[7, 7], 
                                 padding='same',
                                 groups=self.groups) 
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        # NOTE: your projection shrinks spatial dims via 5x5 VALID; assumed to match main path.
        #self.projection = DepthwiseConv2D(kernel_size=[5, 5], padding='valid')
        #self.proj_norm = LayerNormalization()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(
            1, kernel_size, padding="same", use_bias=False,
            kernel_initializer="he_normal"
        )
        self.act = layers.Activation("sigmoid")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernel_size": self.kernel_size})
        return cfg

    def call(self, x):
        # channel-wise pooling → concat → conv → sigmoid
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        maxv = tf.reduce_max(x, axis=-1, keepdims=True)
        attn = tf.concat([avg, maxv], axis=-1)            # [B,H,W,2]
        attn = self.conv(attn)                            # [B,H,W,1]
        attn = self.act(attn)                             # [0,1]
        return x * attn

    
class SpatialSE(layers.Layer):
    """Applies squeeze and excitation to input feature maps as seen in
    https://arxiv.org/abs/1709.01507.

    Args:
        ratio: The ratio with which the feature map needs to be reduced in
        the reduction phase.

    Inputs:
        Convolutional features.

    Outputs:
        Attention modified feature maps.
    """

    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

    def build(self, input_shape):
        self.reduction = GroupConv2D(input_channels=input_shape[-1],
                                output_channels=self.ratio, 
                                kernel_size=[5, 5], 
                                padding='same',
                                groups=self.ratio,
                                use_bias=False)
        self.relu = layers.Activation("relu")
        self.attn = Conv2D(1, kernel_size=(7, 7), padding='same', use_bias=False, activation="sigmoid", kernel_initializer="he_normal")
        self.multiply = layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.relu(self.reduction(x))
        x = self.attn(x)
        x = self.multiply([shortcut, x])
        return x
    
class ChannelSE(layers.Layer):
    """Applies squeeze and excitation to input feature maps as seen in
    https://arxiv.org/abs/1709.01507.

    Args:
        ratio: The ratio with which the feature map needs to be reduced in
        the reduction phase.

    Inputs:
        Convolutional features.

    Outputs:
        Attention modified feature maps.
    """

    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

    def build(self, input_shape):
        filters = input_shape[-1]
        self.squeeze_avg = layers.GlobalAveragePooling2D(keepdims=True)
        self.channel_reduction = layers.Dense(
            units=filters // (self.ratio), activation="relu", use_bias=False, kernel_initializer="he_normal"
        )
        self.channel_excite = layers.Dense(units=filters, activation="sigmoid", use_bias=False, kernel_initializer="he_normal") #TRY: softmax
        self.multiply = layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.squeeze_avg(x)
        x = self.channel_reduction(x)
        x = self.channel_excite(x)
        x = self.multiply([shortcut, x])
        return x
    

class TransposeConvBlock(keras.layers.Layer):

    def __init__(self, filters, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"strides": self.strides})
        return config

    def build(self, input_shape):
        kernel_size = 3 if self.strides == 1 else 4
        self.norm = LayerNormalization()
        self.conv = Conv2D(self.filters, kernel_size=(kernel_size, kernel_size), strides=self.strides, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.relu = layers.Activation("swish")

    def call(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x            
    


class ConvBlock(keras.layers.Layer):

    def __init__(self, filters, pool=False, attn=False, kernel=3, stride=(1, 1), dilation=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.attn = attn
        self.pool = pool
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation

    def get_config(self):
        config = super().get_config()
        config.update({"attn": self.attn})
        config.update({"pool": self.pool})
        config.update({"kernel": self.kernel})
        config.update({"stride": self.stride})
        config.update({"filters": self.filters})
        config.update({"dilation": self.dilation})
        return config

    def build(self, input_shape):

        self.norm = LayerNormalization()
        self.conv = Conv2D(self.filters, kernel_size=[self.kernel, self.kernel], strides=self.stride, dilation_rate=self.dilation, padding='same')
        self.relu = layers.Activation("swish")

    def call(self, input, training):
        x = input
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x            
     

# class GroupConv2D(keras.layers.Layer):

#     def __init__(self, input_channels, output_channels, kernel_size=(3, 3),
#                  padding='same', strides=(1, 1), groups=1, use_bias=True, **kwargs):
#         super(GroupConv2D, self).__init__(**kwargs)
#         if not input_channels % groups == 0:
#             raise ValueError("The input channel must be divisible by the no. of groups")
#         if not output_channels % groups == 0:
#             raise ValueError("The output channel must be divisible by the no. of groups") 
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding
#         self.groups = groups
#         self.group_in_num = input_channels // groups
#         self.group_out_num = output_channels // groups
#         self.conv_list = []
#         self.use_bias = use_bias


#     def build(self, input_shape):
#         for i in range(self.groups):
#             self.conv_list.append(
#             Conv2D(filters=self.group_out_num, kernel_size=self.kernel_size, padding=self.padding, strides=self.strides, use_bias=self.use_bias, kernel_initializer="he_normal"))

#     def call(self, input):
#         feature_map_list = []
#         splits = tf.split(input, self.groups, axis=-1)
#         for split, conv in zip(splits, self.conv_list):
#             feature_map_list.append(conv(split))
#         x = keras.layers.concatenate(feature_map_list, axis=-1)
        
#         return x


class DummyBlock(keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x, training):
        return x
    

class ResidualBlock(keras.layers.Layer):
    
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = GroupConv2D(input_channels=self.filters//self.block_reduction,
                                 output_channels=self.filters//self.block_reduction, 
                                 kernel_size=[3, 3], 
                                 padding='same',
                                groups=self.groups)
        self.norm2 = LayerNormalization()
        self.conv3 = GroupConv2D(input_channels=self.filters//self.block_reduction,
                                 output_channels=self.filters//self.block_reduction, 
                                 kernel_size=[3, 3], 
                                 padding='same',
                                groups=self.groups) 
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        self.relu = layers.Activation("swish")

    def call(self, x, training):
        shortcut = x
        x = self.relu1(self.conv1(x))
        #x = self.norm2(self.conv2(x))
        #x = channel_shuffle(x, self.groups)
        #x = self.relu3(self.norm3(self.conv3(x)))
        x = self.conv4(x)
        x = self.add([shortcut, x])
        x = self.relu(x)
        return x
    

class FullResidualBlock(keras.layers.Layer):
    
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = GroupConv2D(input_channels=self.filters//self.block_reduction,
                                 output_channels=self.filters//self.block_reduction, 
                                 kernel_size=[3, 3], 
                                 padding='same',
                                groups=self.groups)
        self.norm2 = LayerNormalization()
        self.conv3 = GroupConv2D(input_channels=self.filters//self.block_reduction,
                                 output_channels=self.filters//self.block_reduction, 
                                 kernel_size=[3, 3], 
                                 padding='same',
                                groups=self.groups) 
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        self.relu = layers.Activation("swish")

    def call(self, x, training):
        shortcut = x
        x = self.relu1(self.conv1(x))
        x = self.norm2(self.conv2(x))
        x = channel_shuffle(x, self.groups)
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.conv4(x)
        x = self.add([shortcut, x])
        x = self.relu(x)
        return x



class SpatialSE(layers.Layer):
    """Applies squeeze and excitation to input feature maps as seen in
    https://arxiv.org/abs/1709.01507.

    Args:
        ratio: The ratio with which the feature map needs to be reduced in
        the reduction phase.

    Inputs:
        Convolutional features.

    Outputs:
        Attention modified feature maps.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def build(self, input_shape):
        self.reduction = DepthwiseConv2D(kernel_size=[7, 7], padding='same', use_bias=False)
        self.relu = layers.Activation("swish")
        self.attn = Conv2D(1, kernel_size=(7, 7), padding='same', use_bias=False, activation="sigmoid", kernel_initializer="he_normal")
        self.multiply = layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.relu(self.reduction(x))
        x = self.attn(x)
        x = self.multiply([shortcut, x])
        return x
    

class ChannelSE(layers.Layer):
    """Applies squeeze and excitation to input feature maps as seen in
    https://arxiv.org/abs/1709.01507.

    Args:
        ratio: The ratio with which the feature map needs to be reduced in
        the reduction phase.

    Inputs:
        Convolutional features.

    Outputs:
        Attention modified feature maps.
    """

    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

    def build(self, input_shape):
        filters = input_shape[-1]
        self.squeeze_avg = layers.GlobalAveragePooling2D(keepdims=True)
        self.channel_reduction = layers.Dense(
            units=filters // (self.ratio), activation="relu", use_bias=False, kernel_initializer="he_normal"
        )
        self.channel_excite = layers.Dense(units=filters, activation="sigmoid", use_bias=False, kernel_initializer="he_normal") #TRY: softmax
        self.multiply = layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.squeeze_avg(x)
        x = self.channel_reduction(x)
        x = self.channel_excite(x)
        x = self.multiply([shortcut, x])
        return x


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.query = tf.keras.layers.Conv2D(self.channels // 8, kernel_size=1)
        self.key = tf.keras.layers.Conv2D(self.channels // 8, kernel_size=1)
        self.value = tf.keras.layers.Conv2D(self.channels // 2, kernel_size=1)
        self.output_conv = tf.keras.layers.Conv2D(self.channels, kernel_size=1)

    def call(self, x):
        batch_size, height, width, _ = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        q = self.query(x)  # (batch, height, width, channels // 8)
        k = self.key(x)    # (batch, height, width, channels // 8)
        v = self.value(x)  # (batch, height, width, channels // 2)

        q_flatten = tf.reshape(q, [batch_size, -1, self.channels // 8])  # (batch, height*width, channels // 8)
        k_flatten = tf.reshape(k, [batch_size, -1, self.channels // 8])  # (batch, height*width, channels // 8)
        v_flatten = tf.reshape(v, [batch_size, -1, self.channels // 2])  # (batch, height*width, channels // 2)

        attention_weights = tf.nn.softmax(tf.matmul(q_flatten, k_flatten, transpose_b=True))  # (batch, height*width, height*width)
        attention_out = tf.matmul(attention_weights, v_flatten)  # (batch, height*width, channels // 2)

        attention_out = tf.reshape(attention_out, [batch_size, height, width, self.channels // 2])  # Reshape back
        output = self.output_conv(attention_out) + x  # Residual connection

        return output


class GenResidualBlock(keras.layers.Layer):
    
    def __init__(self, filters, strides=(1, 1),  **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"strides": self.strides})
        return config

    def build(self, input_shape):
        self.conv_1 = Conv2DTranspose(self.filters, kernel_size=3, strides=self.strides, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.shortcut_conv = Conv2DTranspose(self.filters, kernel_size=1, strides=self.strides, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.shortcut_norm = LayerNormalization()

        self.norm_1 = LayerNormalization()
        self.relu_1 = layers.Activation("swish")

        self.conv_2 = Conv2D(self.filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.norm_2 = LayerNormalization()
        self.add = Add()
        self.relu_2 = layers.Activation("swish")

    def call(self, x, training):
        shortcut = self.shortcut_norm(self.shortcut_conv(x))
        x = self.relu_1(self.norm_1(self.conv_1(x)))
        x = self.norm_2(self.conv_2(x))
        x = self.add([shortcut, x])
        x = self.relu_2(x)
        return x


class DiscResidualBlock(keras.layers.Layer):
    
    def __init__(self, filters, kernel_size=(3, 3), drop_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"kernel_size": self.kernel_size})
        config.update({"drop_rate": self.drop_rate})
        return config

    def build(self, input_shape):
        self.residual = Conv2D(self.filters, kernel_size=2, strides=2, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.conv1 = layers.SpectralNormalization(Conv2D(filters=self.filters, kernel_size=(4, 4), strides=2, padding='same', kernel_initializer='he_normal'))  #, kernel_constraint=tf.keras.constraints.MaxNorm(2.0)
        self.relu1 = layers.LeakyReLU(alpha=0.2)
        self.dropout1 = layers.Dropout(self.drop_rate)
        self.conv2 = layers.SpectralNormalization(Conv2D(filters=self.filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal'))  #, kernel_constraint=tf.keras.constraints.MaxNorm(2.0)

        
        self.add = Add()

    def call(self, x, training):
        shortcut = self.residual(x)
        x = self.dropout1(self.relu1(self.conv1(x)))
        x = self.conv2(x)
        x = self.add([shortcut, x])
        return x
    

class TransitionLayer(keras.layers.Layer):

    def __init__(self, filters, groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        return config

    def build(self, input_shape):
        self.conv = GroupConv2D(input_channels=input_shape[-1],
                                 output_channels=self.filters, 
                                 kernel_size=[3, 3], 
                                 strides=(2, 2),
                                 padding='same',
                                groups=self.groups) 
        self.norm = LayerNormalization()
        self.relu = layers.Activation("swish")

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        
        return x
    

class PoolingLayer(keras.layers.Layer):

    def __init__(self, filters, frac_ratio=1.0 ,groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups
        self.frac_ratio = frac_ratio

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        config.update({"frac_ratio": self.frac_ratio})
        return config

    def build(self, input_shape):
        self.depth = input_shape[-1]
        if self.frac_ratio == 2.0: self.pool = MaxPool2D(pool_size=(2, 2)) #AveragePooling2D(pool_size=(2, 2))
        self.channel_up_conv = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.channel_up_norm = LayerNormalization()
        self.channel_up_swish = layers.Activation("swish")

    def call(self, input):
        x = input
        if self.frac_ratio < 2.0 and self.frac_ratio > 0.0: x = tf.nn.fractional_max_pool(value=x, pooling_ratio=[1, self.frac_ratio, self.frac_ratio, 1], pseudo_random=True, overlapping=False)[0]
        elif self.frac_ratio == 2.0: x = self.pool(x) 
        x = self.channel_up_conv(x)
        x = self.channel_up_norm(x)
        x = self.channel_up_swish(x)
        
        return x
    
class GlobalAvgPoolLayer(keras.layers.Layer):

    def __init__(self, num_classes, drop_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.drop_rate =  drop_rate

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        config.update({"drop_rate": self.drop_rate})
        return config

    def build(self, input_shape):
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = Dropout(rate=self.drop_rate)
        self.dense = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, x):
        x = self.global_avg_pooling(x)
        if self.drop_rate > 0.0: x = self.dropout(x)
        logits = self.dense(x)
        
        return logits
    

class BaseGenerator(keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        return config
    
    def build(self, input_shape):
        self.conv_trans_1 = Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')
        self.relu_1 = layers.Activation("swish")
        self.norm_1 = LayerNormalization()
        self.conv_trans_2 = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer='he_normal')
        self.relu_2 = layers.Activation("swish")
        self.norm_2 = LayerNormalization()
        self.conv_trans_3 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer='he_normal')
        self.relu_3 = layers.Activation("swish") 
        self.norm_3 = LayerNormalization()
        self.conv_trans_final = Conv2D(3, (3, 3), strides=(1,1), padding='same', use_bias=False, activation='tanh', kernel_initializer='he_normal')


    def call(self, x):
        x = self.norm_1(self.relu_1(self.conv_trans_1(x))) 
        x = self.norm_2(self.relu_2(self.conv_trans_2(x))) 
        x = self.norm_3(self.relu_3(self.conv_trans_3(x))) 
        x = self.conv_trans_final(x) 
        return x



class BaseDiscriminator(keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        return config
    
    def build(self, input_shape): 
        # TODO try: just use BatchNorm
        self.conv_1 = layers.SpectralNormalization(Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='same', kernel_initializer=init))  #, kernel_constraint=tf.keras.constraints.MaxNorm(2.0)
        self.relu_1 = layers.LeakyReLU(alpha=0.2)
        self.dropout_1 = layers.Dropout(0.3)
        self.conv_2 = layers.SpectralNormalization(Conv2D(filters=128, kernel_size=(4, 4), strides=2, padding='same', kernel_initializer=init))
        self.relu_2 = layers.LeakyReLU(alpha=0.2)
        self.dropout_2 = layers.Dropout(0.3)
        self.conv_3 = layers.SpectralNormalization(Conv2D(filters=256, kernel_size=(4, 4), strides=2, padding='same', kernel_initializer=init))
        self.relu_3 = layers.LeakyReLU(alpha=0.2)
        self.dropout_3 = layers.Dropout(0.3)
        self.flatten = Flatten()
        self.dropout = layers.Dropout(0.3)
        self.dense = Dense(1, activation='sigmoid')

        self.concat = layers.Concatenate()
        self.upsample = tf.keras.layers.UpSampling2D(size=(8,8))
    
    def call(self, input, disc_train=True):
        x, label = input[0], input[1]
        label = self.upsample(label)
        x = self.concat([x, label])
        x = self.dropout_1(x)
        x = self.relu_1(self.conv_1(x))
        x = self.dropout_2(x)
        x = self.relu_2(self.conv_2(x))
        x = self.dropout_3(x)
        x = self.relu_3(self.conv_3(x))
        x = self.flatten(x)
        x = self.dropout(x)

        x = self.dense(x)
        return x
    