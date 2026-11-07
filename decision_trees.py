"""ID3 and C4.5 decision tree implementations (NumPy + Pandas only).

ID3   : categorical features, multiway splits, Information Gain.
C4.5  : categorical + continuous features, Gain Ratio, binary threshold
        splits for numeric attributes, and Quinlan's "average gain" filter.
"""
import itertools
from collections import Counter

import graphviz
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Information theory helpers
# --------------------------------------------------------------------------
def entropy(y):
    """Shannon entropy (base 2) of a label array."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def gain_and_split_info(y, base_entropy, parts):
    """Information gain and split information for a partition of y."""
    n = len(y)
    gain = base_entropy
    split_info = 0.0
    for part in parts:
        w = len(part) / n
        gain -= w * entropy(part)
        split_info -= w * np.log2(w)
    return gain, split_info


# --------------------------------------------------------------------------
# Tree structure
# --------------------------------------------------------------------------
class Node:
    def __init__(self, counts, label):
        self.counts = counts          # {class: count}
        self.label = label            # majority class at this node
        self.feature = None
        self.threshold = None         # only for numeric splits
        self.kind = None              # "cat" | "num" | None (leaf)
        self.children = {}            # value -> Node, or {"<=": Node, ">": Node}

    @property
    def is_leaf(self):
        return self.feature is None

    @property
    def n_samples(self):
        return sum(self.counts.values())


class BaseTree:
    name = "base"
    supports_numeric = False

    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain

    # ---- training ---------------------------------------------------------
    def fit(self, X, y):
        X = X.reset_index(drop=True)
        self.feature_names_ = list(X.columns)
        self.numeric_ = (
            [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            if self.supports_numeric
            else []
        )
        self.root_ = self._build(X, np.asarray(y), list(X.columns), 0)
        return self

    def _build(self, X, y, features, depth):
        counts = Counter(y.tolist())
        label = counts.most_common(1)[0][0]
        node = Node(dict(counts), label)

        if (
            len(counts) == 1
            or not features
            or len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        split = self._best_split(X, y, features)
        if split is None:
            return node

        feat, thr = split
        node.feature = feat
        if thr is None:  # categorical multiway split
            node.kind = "cat"
            remaining = [f for f in features if f != feat]
            for value, idx in X.groupby(feat, sort=False).indices.items():
                node.children[value] = self._build(
                    X.iloc[idx].reset_index(drop=True), y[idx], remaining, depth + 1
                )
        else:  # numeric binary split
            node.kind = "num"
            node.threshold = thr
            mask = (X[feat] <= thr).values
            node.children["<="] = self._build(
                X[mask].reset_index(drop=True), y[mask], features, depth + 1
            )
            node.children[">"] = self._build(
                X[~mask].reset_index(drop=True), y[~mask], features, depth + 1
            )
        return node

    def _best_split(self, X, y, features):
        raise NotImplementedError

    # ---- inference --------------------------------------------------------
    def _predict_one(self, row):
        node = self.root_
        while not node.is_leaf:
            v = row[node.feature]
            if node.kind == "num":
                node = node.children["<=" if v <= node.threshold else ">"]
            elif v in node.children:
                node = node.children[v]
            else:  # unseen category: fall back to this node's majority class
                break
        return node.label

    def predict(self, X):
        return np.array([self._predict_one(r) for r in X.to_dict("records")])

    # ---- stats ------------------------------------------------------------
    def depth(self):
        def d(n):
            return 0 if n.is_leaf else 1 + max(d(c) for c in n.children.values())
        return d(self.root_)

    def n_nodes(self):
        def c(n):
            return 1 + sum(c(ch) for ch in n.children.values())
        return c(self.root_)

    def n_leaves(self):
        def c(n):
            return 1 if n.is_leaf else sum(c(ch) for ch in n.children.values())
        return c(self.root_)


# --------------------------------------------------------------------------
# ID3
# --------------------------------------------------------------------------
class ID3Tree(BaseTree):
    """ID3: choose the categorical attribute with the highest information gain."""

    name = "ID3"
    supports_numeric = False

    def _best_split(self, X, y, features):
        base = entropy(y)
        best_gain, best_feat = self.min_gain, None
        for f in features:
            groups = X.groupby(f, sort=False).indices
            if len(groups) < 2:
                continue
            gain, _ = gain_and_split_info(y, base, [y[idx] for idx in groups.values()])
            if gain > best_gain:
                best_gain, best_feat = gain, f
        return None if best_feat is None else (best_feat, None)


# --------------------------------------------------------------------------
# C4.5
# --------------------------------------------------------------------------
class C45Tree(BaseTree):
    """C4.5: gain ratio, continuous attributes via best binary threshold."""

    name = "C4.5"
    supports_numeric = True

    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-7,
                 max_thresholds=64):
        super().__init__(max_depth, min_samples_split, min_gain)
        self.max_thresholds = max_thresholds

    def _best_split(self, X, y, features):
        base = entropy(y)
        candidates = []  # (gain, gain_ratio, feature, threshold)

        for f in features:
            if f in self.numeric_:
                x = X[f].values.astype(float)
                vals = np.unique(x)
                if len(vals) < 2:
                    continue
                mids = (vals[:-1] + vals[1:]) / 2
                if len(mids) > self.max_thresholds:
                    pick = np.linspace(0, len(mids) - 1, self.max_thresholds).astype(int)
                    mids = np.unique(mids[pick])
                best = None
                for t in mids:
                    m = x <= t
                    gain, si = gain_and_split_info(y, base, [y[m], y[~m]])
                    if si > 0 and (best is None or gain > best[0]):
                        best = (gain, gain / si, f, float(t))
                if best:
                    candidates.append(best)
            else:
                groups = X.groupby(f, sort=False).indices
                if len(groups) < 2:
                    continue
                gain, si = gain_and_split_info(y, base, [y[i] for i in groups.values()])
                if si > 0:
                    candidates.append((gain, gain / si, f, None))

        candidates = [c for c in candidates if c[0] > self.min_gain]
        if not candidates:
            return None
        # Quinlan's heuristic: only attributes with at-least-average gain compete
        avg_gain = np.mean([c[0] for c in candidates])
        eligible = [c for c in candidates if c[0] >= avg_gain - 1e-12]
        best = max(eligible, key=lambda c: c[1])
        return best[2], best[3]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def to_graphviz(tree, class_names):
    """Return a graphviz.Digraph of the fitted tree."""
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB")
    counter = itertools.count()

    def add(node):
        nid = str(next(counter))
        dist = ", ".join(f"{class_names[k]}: {v}" for k, v in sorted(node.counts.items()))
        if node.is_leaf:
            dot.node(nid, f"{class_names[node.label]}\nn={node.n_samples}\n[{dist}]",
                     shape="box", style="filled,rounded", fillcolor="#c7e9c0")
        else:
            dot.node(nid, f"{node.feature}\nn={node.n_samples}",
                     shape="ellipse", style="filled", fillcolor="#bdd7ee")
            for key, child in node.children.items():
                cid = add(child)
                label = str(key) if node.kind == "cat" else f"{key} {node.threshold:.4g}"
                dot.edge(nid, cid, label=label)
        return nid

    add(tree.root_)
    return dot


def to_text(tree, class_names):
    """Plain-text rendering of the tree."""
    lines = []

    def rec(node, indent):
        if node.is_leaf:
            lines.append(f"{indent}-> {class_names[node.label]} (n={node.n_samples})")
            return
        for key, child in node.children.items():
            cond = f"{node.feature} = {key}" if node.kind == "cat" \
                else f"{node.feature} {key} {node.threshold:.4g}"
            lines.append(f"{indent}[{cond}]")
            rec(child, indent + "    ")

    rec(tree.root_, "")
    return "\n".join(lines)