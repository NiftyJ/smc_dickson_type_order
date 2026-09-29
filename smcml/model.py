"""
THE JUDGE: a model that gives every candidate a score.
Higher score = more likely to reach TARGET_R before the stop.

Two options (choose with MODEL in config.py):

  "gbm"  Gradient-boosted trees on the feature table. START HERE.
         Trees are the best tool for small tables of numbers like this one.
         Rare winners are handled with class weights: each winner counts as much as
         all the losers together, so the model cannot just say "loss" every time.

  "cnn"  A small convolutional network that LOOKS at the chart picture
         (chart_images.py) and also reads the feature table. Trained with FOCAL LOSS
         (Lin et al., 2017). Focal loss was invented for object detection, where
         almost every candidate box is empty background; it turns down the loss on
         easy, obvious examples so the rare positives are not drowned out. That is the
         same shape as your problem: about 1 setup in 20 is a 20R winner.
         Needs PyTorch. Only worth trying once "gbm" works.

Both have the same two methods: fit(X, y, images) and score(X, images).
"""
import numpy as np


def balanced_weights(y):
    """Weights so that winners and losers count equally in total."""
    y = np.asarray(y, dtype=float)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return np.ones(len(y))
    return np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))


class TreeModel:
    def __init__(self, seed=0):
        self.const = None
        try:
            import lightgbm as lgb
            self.m = lgb.LGBMClassifier(
                n_estimators=250, learning_rate=0.03, num_leaves=8, min_child_samples=30,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                random_state=seed, verbose=-1)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.m = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=30,
                l2_regularization=2.0, random_state=seed)

    def fit(self, X, y, images=None):
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:              # nothing to learn from
            self.const = float(y.mean()) if len(y) else 0.0
            return self
        self.const = None
        self.m.fit(X, y, sample_weight=balanced_weights(y))
        return self

    def score(self, X, images=None):
        if self.const is not None:
            return np.full(len(X), self.const)
        return self.m.predict_proba(X)[:, 1]

    def importance(self, columns):
        """Which features the trees used most (only for LightGBM / when available)."""
        imp = getattr(self.m, "feature_importances_", None)
        if imp is None:
            return None
        return dict(sorted(zip(columns, imp), key=lambda kv: -kv[1]))


# ------------------------------------------------------------------------ CNN (optional)
def focal_loss(logits, y, gamma=2.0, alpha=0.75):
    """Focal loss for yes/no labels.

    Ordinary cross-entropy, multiplied by (1 - p_correct)^gamma: examples the model
    already gets right with confidence contribute almost nothing, so training time is
    spent on the hard, rare cases. alpha gives the positive class extra weight.
    """
    import torch
    import torch.nn.functional as F
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    p_correct = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - p_correct) ** gamma * ce).mean()


class CNNModel:
    def __init__(self, seed=0, epochs=25, lr=1e-3, gamma=2.0, alpha=0.75, batch=64):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("MODEL='cnn' needs PyTorch: pip install torch") from e
        self.seed, self.epochs, self.lr, self.gamma, self.alpha, self.batch = seed, epochs, lr, gamma, alpha, batch

    def _tab(self, X):
        Z = (np.asarray(X, dtype=np.float32) - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X, y, images):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Xa = np.asarray(X, dtype=np.float32)
        self.mu = np.nanmean(Xa, axis=0)
        self.sd = np.nanstd(Xa, axis=0) + 1e-6
        self.mu = np.nan_to_num(self.mu)
        tab = torch.tensor(self._tab(Xa))
        img = torch.tensor(images, dtype=torch.float32) / 255.0
        yt = torch.tensor(np.asarray(y, dtype=np.float32))

        class Net(nn.Module):
            def __init__(self, n_tab):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
                self.head = nn.Sequential(nn.Linear(32 + n_tab, 32), nn.ReLU(),
                                          nn.Dropout(0.2), nn.Linear(32, 1))

            def forward(self, im, tb):
                return self.head(torch.cat([self.conv(im).flatten(1), tb], 1)).squeeze(1)

        self.net = Net(tab.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        n = len(yt)
        self.net.train()
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch):
                b = torch.tensor(order[s:s + self.batch])
                loss = focal_loss(self.net(img[b], tab[b]), yt[b], self.gamma, self.alpha)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def score(self, X, images):
        import torch
        self.net.eval()
        with torch.no_grad():
            tab = torch.tensor(self._tab(X))
            img = torch.tensor(images, dtype=torch.float32) / 255.0
            return torch.sigmoid(self.net(img, tab)).numpy()


def make_model(kind, seed=0):
    if kind == "gbm":
        return TreeModel(seed)
    if kind == "cnn":
        return CNNModel(seed)
    raise ValueError("MODEL must be 'gbm' or 'cnn'")
