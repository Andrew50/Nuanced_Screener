from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .normalization import StandardBatch
from .paths import DataPaths
from .stack6_features import Stack6FeatureSpec, stack6_required_window_size, stack6_tabular_features


def _require_sklearn():  # noqa: ANN001
    try:
        import sklearn  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "scikit-learn is required for Stack 6 classic models.\n"
            "Install with: pip install -e '.[ml_classic]'\n"
            f"Import error: {type(e).__name__}: {e}"
        ) from e


def _require_lightgbm():  # noqa: ANN001
    try:
        import lightgbm  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "LightGBM is required for lgbm_stack6.\n"
            "Install with: pip install -e '.[ml_classic]'\n"
            f"Import error: {type(e).__name__}: {e}"
        ) from e


def _require_hmmlearn():  # noqa: ANN001
    try:
        import hmmlearn  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "hmmlearn is required for hmm_regime.\n"
            "Install with: pip install -e '.[ml_classic]'\n"
            f"Import error: {type(e).__name__}: {e}"
        ) from e


def _infer_repo_root_from_run_dir(run_dir: Path) -> Path:
    """
    Try to infer repo_root given a run_dir like:
      <repo_root>/data/models/<model_type>/<setup>/<run_id>
    """
    p = run_dir.resolve()
    # Best-effort: find the nearest ancestor that contains data/models.
    for parent in [p] + list(p.parents):
        if (parent / "data" / "models").exists():
            return parent
        if parent.name == "data" and (parent / "models").exists():
            return parent.parent
    # Fallback: assume old layout.
    if len(p.parents) >= 5:
        return p.parents[4]
    return p.parent


def _read_weights(batch: StandardBatch, *, weight_key: str = "weight") -> np.ndarray | None:
    w = batch.meta.get(weight_key)
    if w is None:
        return None
    try:
        arr = np.asarray(w).astype(float)
    except Exception:
        return None
    arr = np.where(np.isfinite(arr), arr, 1.0).astype(float)
    arr = np.clip(arr, 0.0, 1e6)
    return arr


def _safe_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.where(np.isfinite(p), p, np.nan)
    return np.clip(p, 0.0, 1.0)


@dataclass
class _MedianImputer:
    med: np.ndarray  # (D,)

    @classmethod
    def fit(cls, X: np.ndarray) -> "_MedianImputer":
        X = np.asarray(X, dtype=float)
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        return cls(med=med.astype(float))

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = X.copy()
        mask = ~np.isfinite(out)
        if mask.any():
            out[mask] = np.take(self.med, np.where(mask)[1])
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(float, copy=False)


class _ProbCalibrator:
    """
    Lightweight probability calibration wrapper.

    - platt: logistic regression on a single feature (raw prob)
    - isotonic: isotonic regression mapping raw prob -> calibrated prob
    """

    def __init__(self, method: str, model):  # noqa: ANN001
        self.method = str(method)
        self.model = model

    @classmethod
    def fit(cls, method: str, p_raw: np.ndarray, y_true: np.ndarray, sample_weight: np.ndarray | None = None) -> "_ProbCalibrator":
        _require_sklearn()
        method = str(method).strip().lower()
        p = _safe_prob(p_raw)
        y = np.asarray(y_true).astype(int)
        ok = np.isfinite(p) & np.isfinite(y)
        p = p[ok]
        y = y[ok]
        w = sample_weight[ok] if sample_weight is not None else None
        if p.size == 0:
            raise ValueError("No finite probabilities to calibrate")

        if method == "platt":
            from sklearn.linear_model import LogisticRegression

            X = p.reshape(-1, 1)
            lr = LogisticRegression(solver="lbfgs", max_iter=500)
            lr.fit(X, y, sample_weight=w)
            return cls(method=method, model=lr)
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y, sample_weight=w)
            return cls(method=method, model=iso)
        raise ValueError("method must be platt|isotonic")

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p = _safe_prob(p_raw)
        if self.method == "platt":
            X = p.reshape(-1, 1)
            out = self.model.predict_proba(X)[:, 1]
            return _safe_prob(out)
        if self.method == "isotonic":
            out = self.model.transform(p)
            return _safe_prob(out)
        return _safe_prob(p)


@dataclass(frozen=True)
class Stack6Artifact:
    runner: str
    feature_names: list[str]
    spec: dict
    has_hmm_feature: bool
    model_path: str
    imputer_path: str
    calibrator_path: str | None


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_pickle(path: Path, obj) -> None:  # noqa: ANN001
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=4)


def _load_pickle(path: Path):  # noqa: ANN001
    with path.open("rb") as f:
        return pickle.load(f)


def _stack6_Xy(
    batch: StandardBatch,
    *,
    spec: Stack6FeatureSpec,
    use_hmm_feature: bool,
    hmm_helper: "RegimeHMM | None",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    hmm_p = None
    if use_hmm_feature and hmm_helper is not None:
        hmm_p = hmm_helper.p_trend_for_batch(batch)
    X, names = stack6_tabular_features(batch, spec=spec, hmm_p_trend=hmm_p)
    y = np.asarray(batch.y).astype(int)
    return X, y, names


class RegimeHMM:
    """
    2-state regime HMM fit on contiguous per-ticker daily returns.
    Emissions default to [r, |r|] where r is log close-to-close return.
    """

    def __init__(self, *, repo_root: Path, hmm, trend_state: int, tickers: list[str]):  # noqa: ANN001
        self.repo_root = Path(repo_root)
        self.hmm = hmm
        self.trend_state = int(trend_state)
        self.tickers = list(tickers)

    @staticmethod
    def _load_ticker_df(paths: DataPaths, ticker: str) -> pd.DataFrame | None:
        p = paths.raw_ticker_parquet(ticker)
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
        except Exception:
            return None
        if df.empty:
            return None
        if "date" not in df.columns or "close" not in df.columns:
            return None
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        out = out.dropna(subset=["date", "close"]).sort_values(["date"]).reset_index(drop=True)
        return out

    @staticmethod
    def _returns_features(df: pd.DataFrame) -> tuple[np.ndarray, list[date]]:
        close = df["close"].to_numpy(dtype=float)
        dates = df["date"].tolist()
        prev = np.roll(close, 1)
        prev[0] = np.nan
        ok = np.isfinite(close) & np.isfinite(prev) & (close > 0) & (prev > 0)
        r = np.full_like(close, np.nan, dtype=float)
        r[ok] = np.log(close[ok] / prev[ok])
        # emissions: [r, abs(r)]
        X = np.stack([r, np.abs(r)], axis=1)
        return X, dates

    @classmethod
    def fit(
        cls,
        *,
        run_dir: Path,
        tickers: list[str],
        min_asof: date,
        max_asof: date,
        seed: int = 1337,
    ) -> "RegimeHMM":
        _require_hmmlearn()
        from hmmlearn.hmm import GaussianHMM

        repo_root = _infer_repo_root_from_run_dir(run_dir)
        paths = DataPaths(repo_root)

        Xs: list[np.ndarray] = []
        lengths: list[int] = []
        used_tickers: list[str] = []

        # Fit on data up to max_asof (decision date), using only bars strictly before it.
        # For each ticker, take the full available history subset within a broad range.
        # (HMM benefits from contiguous sequences; missing days are fine, we just keep observed points.)
        for tkr in sorted(set(str(t).upper() for t in tickers)):
            df = cls._load_ticker_df(paths, tkr)
            if df is None:
                continue
            df = df[(df["date"] >= min_asof) & (df["date"] < max_asof)].reset_index(drop=True)
            if len(df) < 50:
                continue
            Xi, _dates = cls._returns_features(df)
            ok = np.isfinite(Xi).all(axis=1)
            Xi = Xi[ok]
            if Xi.shape[0] < 50:
                continue
            Xs.append(Xi.astype(float))
            lengths.append(int(Xi.shape[0]))
            used_tickers.append(tkr)

        if not Xs:
            raise RuntimeError("No ticker sequences available to fit HMM (missing raw data or too little history).")

        X_all = np.concatenate(Xs, axis=0)
        hmm = GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=200,
            random_state=int(seed),
        )
        hmm.fit(X_all, lengths=lengths)
        means = np.asarray(hmm.means_, dtype=float)
        trend_state = int(np.nanargmax(means[:, 0]))  # higher mean return
        return cls(repo_root=repo_root, hmm=hmm, trend_state=trend_state, tickers=used_tickers)

    def p_trend_for_batch(self, batch: StandardBatch) -> np.ndarray:
        paths = DataPaths(self.repo_root)

        tickers = np.asarray(batch.meta.get("ticker")).astype(str)
        asof = np.asarray(batch.meta.get("asof_date"))
        asof_dates = pd.to_datetime(asof).dt.date.to_numpy()

        out = np.full((int(len(tickers)),), np.nan, dtype=float)

        cache: dict[str, tuple[list[date], np.ndarray]] = {}
        for i, (tkr, d) in enumerate(zip(tickers.tolist(), asof_dates.tolist(), strict=True)):
            tkr_u = str(tkr).upper()
            if tkr_u not in cache:
                df = self._load_ticker_df(paths, tkr_u)
                if df is None or df.empty:
                    cache[tkr_u] = ([], np.empty((0,)))
                else:
                    Xi, dates = self._returns_features(df)
                    ok = np.isfinite(Xi).all(axis=1)
                    Xi2 = Xi[ok]
                    dates2 = [dd for dd, okk in zip(dates, ok.tolist(), strict=True) if okk]
                    if Xi2.shape[0] == 0:
                        cache[tkr_u] = (dates2, np.empty((0,)))
                    else:
                        post = self.hmm.predict_proba(Xi2)
                        p = post[:, self.trend_state].astype(float)
                        cache[tkr_u] = (dates2, p)

            dates2, p = cache[tkr_u]
            if not dates2 or p.size == 0:
                continue
            # Find last observation date strictly before asof_date (decision is at open).
            idx = None
            for j in range(len(dates2) - 1, -1, -1):
                if dates2[j] < d:
                    idx = j
                    break
            if idx is None:
                continue
            out[i] = float(p[idx])

        return _safe_prob(out)


class Stack6LogRegRunner:
    name: str = "logreg_stack6"
    task_type: str = "binary_classification"

    def supports_warm(self) -> bool:  # noqa: D401
        return False

    def warm(self, batches: Iterable[StandardBatch], *, run_dir: Path):  # noqa: ANN001
        raise RuntimeError(f"{self.name} does not support warm()")

    def train(self, batches: Iterable[StandardBatch], *, run_dir: Path):  # noqa: ANN001
        from .model_registry import TrainedArtifact

        _require_sklearn()
        from sklearn.linear_model import LogisticRegression

        run_dir.mkdir(parents=True, exist_ok=True)
        batches = list(batches)
        if not batches:
            raise ValueError("No batches provided")
        if len(batches) != 1:
            raise ValueError("Stack6 runners currently expect a single StandardBatch")
        b = batches[0]

        spec = Stack6FeatureSpec()
        if int(b.x_seq.shape[1]) < stack6_required_window_size(spec):
            raise ValueError("window_size is too small for Stack 6 horizons")

        # Optional internal HMM feature (auto if hmmlearn is installed).
        hmm_helper = None
        try:
            _require_hmmlearn()
            tickers = np.asarray(b.meta.get("ticker")).astype(str).tolist()
            asof = pd.to_datetime(np.asarray(b.meta.get("asof_date"))).dt.date.tolist()
            hmm_helper = RegimeHMM.fit(
                run_dir=run_dir,
                tickers=tickers,
                min_asof=min(asof),
                max_asof=max(asof),
                seed=1337,
            )
        except Exception:
            hmm_helper = None

        X, y, names = _stack6_Xy(b, spec=spec, use_hmm_feature=(hmm_helper is not None), hmm_helper=hmm_helper)
        imp = _MedianImputer.fit(X)
        X2 = imp.transform(X)

        w = _read_weights(b)
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
        clf.fit(X2, y, sample_weight=w)

        model_path = run_dir / "model.pkl"
        imputer_path = run_dir / "imputer.pkl"
        _save_pickle(model_path, clf)
        _save_pickle(imputer_path, imp)

        # Persist optional HMM to allow consistent inference when used as a feature.
        has_hmm_feature = hmm_helper is not None
        if hmm_helper is not None:
            _save_pickle(run_dir / "hmm.pkl", hmm_helper)

        artifact_payload = Stack6Artifact(
            runner=self.name,
            feature_names=list(names),
            spec=spec.__dict__,
            has_hmm_feature=bool(has_hmm_feature),
            model_path=str(model_path.name),
            imputer_path=str(imputer_path.name),
            calibrator_path=None,
        )
        artifact_path = run_dir / "artifact.json"
        _write_json(artifact_path, artifact_payload.__dict__)
        return TrainedArtifact(runner_name=self.name, path=artifact_path)

    def train_with_val(self, train_batches: Iterable[StandardBatch], val_batches: Iterable[StandardBatch], *, run_dir: Path):  # noqa: ANN001
        # Logistic regression doesn't benefit from val-aware training; keep simple.
        return self.train(train_batches, run_dir=run_dir)

    def fit_probability_calibration(  # noqa: ANN001
        self,
        calib_batches: Iterable[StandardBatch],
        *,
        artifact,
        run_dir: Path,
        method: str,
    ) -> None:
        _require_sklearn()
        payload = _read_json(Path(artifact.path))
        model = _load_pickle(run_dir / payload["model_path"])
        imp = _load_pickle(run_dir / payload["imputer_path"])
        hmm_helper = _load_pickle(run_dir / "hmm.pkl") if bool(payload.get("has_hmm_feature")) and (run_dir / "hmm.pkl").exists() else None

        calib_batches = list(calib_batches)
        if not calib_batches:
            return
        if len(calib_batches) != 1:
            raise ValueError("Calibration currently expects a single batch")
        b = calib_batches[0]

        spec = Stack6FeatureSpec(**(payload.get("spec") or {}))
        X, y, _names = _stack6_Xy(b, spec=spec, use_hmm_feature=(hmm_helper is not None), hmm_helper=hmm_helper)
        X2 = imp.transform(X)
        p_raw = model.predict_proba(X2)[:, 1]
        w = _read_weights(b)

        cal = _ProbCalibrator.fit(method=method, p_raw=p_raw, y_true=y, sample_weight=w)
        cal_path = run_dir / "calibrator.pkl"
        _save_pickle(cal_path, cal)
        payload["calibrator_path"] = str(cal_path.name)
        _write_json(Path(artifact.path), payload)

    def predict(self, batches: Iterable[StandardBatch], *, artifact):  # noqa: ANN001
        payload = _read_json(Path(artifact.path))
        model = _load_pickle(Path(artifact.path).parent / payload["model_path"])
        imp = _load_pickle(Path(artifact.path).parent / payload["imputer_path"])
        hmm_helper = (
            _load_pickle(Path(artifact.path).parent / "hmm.pkl")
            if bool(payload.get("has_hmm_feature")) and (Path(artifact.path).parent / "hmm.pkl").exists()
            else None
        )
        cal = (
            _load_pickle(Path(artifact.path).parent / payload["calibrator_path"])
            if payload.get("calibrator_path")
            else None
        )

        rows = []
        for b in batches:
            spec = Stack6FeatureSpec(**(payload.get("spec") or {}))
            X, _y, _names = _stack6_Xy(b, spec=spec, use_hmm_feature=(hmm_helper is not None), hmm_helper=hmm_helper)
            X2 = imp.transform(X)
            p = model.predict_proba(X2)[:, 1]
            if cal is not None:
                p = cal.transform(p)
            p = _safe_prob(p)

            sample_ids = np.asarray(b.meta.get("sample_id"))
            setups = np.asarray(b.meta.get("setup"))
            tickers = np.asarray(b.meta.get("ticker"))
            asof_dates = np.asarray(b.meta.get("asof_date"))
            y_true = np.asarray(b.y).astype(int)

            for sid, sc, yy, ss, tkr, d in zip(
                sample_ids.tolist(),
                p.tolist(),
                y_true.tolist(),
                (setups.tolist() if setups is not None else [None] * len(y_true)),
                (tickers.tolist() if tickers is not None else [None] * len(y_true)),
                (asof_dates.tolist() if asof_dates is not None else [None] * len(y_true)),
                strict=True,
            ):
                rows.append(
                    {
                        "sample_id": str(sid),
                        "setup": str(ss) if ss is not None else None,
                        "ticker": str(tkr) if tkr is not None else None,
                        "asof_date": d,
                        "y_true": int(yy),
                        "score": float(sc),
                    }
                )

        return pd.DataFrame(rows)


class Stack6LightGBMRunner:
    name: str = "lgbm_stack6"
    task_type: str = "binary_classification"

    def supports_warm(self) -> bool:  # noqa: D401
        return False

    def warm(self, batches: Iterable[StandardBatch], *, run_dir: Path):  # noqa: ANN001
        raise RuntimeError(f"{self.name} does not support warm()")

    def _fit_model(  # noqa: ANN001
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        w_train: np.ndarray | None,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        w_val: np.ndarray | None,
        seed: int = 1337,
    ):
        _require_lightgbm()
        import lightgbm as lgb

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = (n_neg / n_pos) if (n_pos > 0 and w_train is None) else 1.0

        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=int(seed),
            n_jobs=-1,
            scale_pos_weight=float(scale_pos_weight),
        )

        if X_val is not None and y_val is not None:
            try:
                cb = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            except Exception:
                cb = None
            clf.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=([w_val] if w_val is not None else None),
                eval_metric="binary_logloss",
                callbacks=cb,
            )
        else:
            clf.fit(X_train, y_train, sample_weight=w_train)
        return clf

    def train(self, batches: Iterable[StandardBatch], *, run_dir: Path):  # noqa: ANN001
        from .model_registry import TrainedArtifact

        _require_lightgbm()
        run_dir.mkdir(parents=True, exist_ok=True)
        batches = list(batches)
        if not batches:
            raise ValueError("No batches provided")
        if len(batches) != 1:
            raise ValueError("Stack6 runners currently expect a single StandardBatch")
        b = batches[0]

        spec = Stack6FeatureSpec()
        if int(b.x_seq.shape[1]) < stack6_required_window_size(spec):
            raise ValueError("window_size is too small for Stack 6 horizons")

        hmm_helper = None
        try:
            _require_hmmlearn()
            tickers = np.asarray(b.meta.get("ticker")).astype(str).tolist()
            asof = pd.to_datetime(np.asarray(b.meta.get("asof_date"))).dt.date.tolist()
            hmm_helper = RegimeHMM.fit(
                run_dir=run_dir,
                tickers=tickers,
                min_asof=min(asof),
                max_asof=max(asof),
                seed=1337,
            )
        except Exception:
            hmm_helper = None

        X, y, names = _stack6_Xy(b, spec=spec, use_hmm_feature=(hmm_helper is not None), hmm_helper=hmm_helper)
        imp = _MedianImputer.fit(X)
        X2 = imp.transform(X)
        w = _read_weights(b)

        clf = self._fit_model(
            X_train=X2,
            y_train=y,
            w_train=w,
            X_val=None,
            y_val=None,
            w_val=None,
        )

        model_path = run_dir / "model.pkl"
        imputer_path = run_dir / "imputer.pkl"
        _save_pickle(model_path, clf)
        _save_pickle(imputer_path, imp)
        if hmm_helper is not None:
            _save_pickle(run_dir / "hmm.pkl", hmm_helper)

        artifact_payload = Stack6Artifact(
            runner=self.name,
            feature_names=list(names),
            spec=spec.__dict__,
            has_hmm_feature=bool(hmm_helper is not None),
            model_path=str(model_path.name),
            imputer_path=str(imputer_path.name),
            calibrator_path=None,
        )
        artifact_path = run_dir / "artifact.json"
        _write_json(artifact_path, artifact_payload.__dict__)
        return TrainedArtifact(runner_name=self.name, path=artifact_path)

    def train_with_val(self, train_batches: Iterable[StandardBatch], val_batches: Iterable[StandardBatch], *, run_dir: Path):  # noqa: ANN001
        from .model_registry import TrainedArtifact

        _require_lightgbm()
        run_dir.mkdir(parents=True, exist_ok=True)
        train_batches = list(train_batches)
        val_batches = list(val_batches)
        if len(train_batches) != 1 or len(val_batches) != 1:
            raise ValueError("Stack6 runners currently expect single train and val batches")
        b_tr = train_batches[0]
        b_va = val_batches[0]

        spec = Stack6FeatureSpec()
        hmm_helper = None
        try:
            _require_hmmlearn()
            tickers = np.asarray(b_tr.meta.get("ticker")).astype(str).tolist()
            asof = pd.to_datetime(np.asarray(b_tr.meta.get("asof_date"))).dt.date.tolist()
            hmm_helper = RegimeHMM.fit(
                run_dir=run_dir,
                tickers=tickers,
                min_asof=min(asof),
                max_asof=max(asof),
                seed=1337,
            )
        except Exception:
            hmm_helper = None

        Xtr, ytr, names = _stack6_Xy(b_tr, spec=spec, use_hmm_feature=(hmm_helper is not None), hmm_helper=hmm_helper)
        Xva, yva, _ = _stack6_Xy(b_va, spec=spec, use_hmm_feature=(hmm_helper is not None), hmm_helper=hmm_helper)
        imp = _MedianImputer.fit(Xtr)
        Xtr2 = imp.transform(Xtr)
        Xva2 = imp.transform(Xva)
        wtr = _read_weights(b_tr)
        wva = _read_weights(b_va)

        clf = self._fit_model(
            X_train=Xtr2,
            y_train=ytr,
            w_train=wtr,
            X_val=Xva2,
            y_val=yva,
            w_val=wva,
        )

        model_path = run_dir / "model.pkl"
        imputer_path = run_dir / "imputer.pkl"
        _save_pickle(model_path, clf)
        _save_pickle(imputer_path, imp)
        if hmm_helper is not None:
            _save_pickle(run_dir / "hmm.pkl", hmm_helper)

        artifact_payload = Stack6Artifact(
            runner=self.name,
            feature_names=list(names),
            spec=spec.__dict__,
            has_hmm_feature=bool(hmm_helper is not None),
            model_path=str(model_path.name),
            imputer_path=str(imputer_path.name),
            calibrator_path=None,
        )
        artifact_path = run_dir / "artifact.json"
        _write_json(artifact_path, artifact_payload.__dict__)
        return TrainedArtifact(runner_name=self.name, path=artifact_path)

    def fit_probability_calibration(  # noqa: ANN001
        self,
        calib_batches: Iterable[StandardBatch],
        *,
        artifact,
        run_dir: Path,
        method: str,
    ) -> None:
        _require_sklearn()
        payload = _read_json(Path(artifact.path))
        model = _load_pickle(run_dir / payload["model_path"])
        imp = _load_pickle(run_dir / payload["imputer_path"])
        hmm_helper = _load_pickle(run_dir / "hmm.pkl") if bool(payload.get("has_hmm_feature")) and (run_dir / "hmm.pkl").exists() else None

        calib_batches = list(calib_batches)
        if not calib_batches:
            return
        if len(calib_batches) != 1:
            raise ValueError("Calibration currently expects a single batch")
        b = calib_batches[0]
        spec = Stack6FeatureSpec(**(payload.get("spec") or {}))
        X, y, _names = _stack6_Xy(b, spec=spec, use_hmm_feature=(hmm_helper is not None), hmm_helper=hmm_helper)
        X2 = imp.transform(X)
        p_raw = model.predict_proba(X2)[:, 1]
        w = _read_weights(b)
        cal = _ProbCalibrator.fit(method=method, p_raw=p_raw, y_true=y, sample_weight=w)
        cal_path = run_dir / "calibrator.pkl"
        _save_pickle(cal_path, cal)
        payload["calibrator_path"] = str(cal_path.name)
        _write_json(Path(artifact.path), payload)

    def predict(self, batches: Iterable[StandardBatch], *, artifact):  # noqa: ANN001
        payload = _read_json(Path(artifact.path))
        model = _load_pickle(Path(artifact.path).parent / payload["model_path"])
        imp = _load_pickle(Path(artifact.path).parent / payload["imputer_path"])
        hmm_helper = (
            _load_pickle(Path(artifact.path).parent / "hmm.pkl")
            if bool(payload.get("has_hmm_feature")) and (Path(artifact.path).parent / "hmm.pkl").exists()
            else None
        )
        cal = (
            _load_pickle(Path(artifact.path).parent / payload["calibrator_path"])
            if payload.get("calibrator_path")
            else None
        )

        rows = []
        for b in batches:
            spec = Stack6FeatureSpec(**(payload.get("spec") or {}))
            X, _y, _names = _stack6_Xy(b, spec=spec, use_hmm_feature=(hmm_helper is not None), hmm_helper=hmm_helper)
            X2 = imp.transform(X)
            p = model.predict_proba(X2)[:, 1]
            if cal is not None:
                p = cal.transform(p)
            p = _safe_prob(p)

            sample_ids = np.asarray(b.meta.get("sample_id"))
            setups = np.asarray(b.meta.get("setup"))
            tickers = np.asarray(b.meta.get("ticker"))
            asof_dates = np.asarray(b.meta.get("asof_date"))
            y_true = np.asarray(b.y).astype(int)

            for sid, sc, yy, ss, tkr, d in zip(
                sample_ids.tolist(),
                p.tolist(),
                y_true.tolist(),
                (setups.tolist() if setups is not None else [None] * len(y_true)),
                (tickers.tolist() if tickers is not None else [None] * len(y_true)),
                (asof_dates.tolist() if asof_dates is not None else [None] * len(y_true)),
                strict=True,
            ):
                rows.append(
                    {
                        "sample_id": str(sid),
                        "setup": str(ss) if ss is not None else None,
                        "ticker": str(tkr) if tkr is not None else None,
                        "asof_date": d,
                        "y_true": int(yy),
                        "score": float(sc),
                    }
                )

        return pd.DataFrame(rows)


class HMMRegimeRunner:
    """
    Unsupervised 2-state HMM regime model.

    Output score is p(trend_state) at decision time.
    """

    name: str = "hmm_regime"
    task_type: str = "binary_classification"

    def supports_warm(self) -> bool:  # noqa: D401
        return False

    def warm(self, batches: Iterable[StandardBatch], *, run_dir: Path):  # noqa: ANN001
        raise RuntimeError(f"{self.name} does not support warm()")

    def train(self, batches: Iterable[StandardBatch], *, run_dir: Path):  # noqa: ANN001
        from .model_registry import TrainedArtifact

        _require_hmmlearn()
        run_dir.mkdir(parents=True, exist_ok=True)
        batches = list(batches)
        if not batches:
            raise ValueError("No batches provided")
        if len(batches) != 1:
            raise ValueError("hmm_regime currently expects a single StandardBatch")
        b = batches[0]
        tickers = np.asarray(b.meta.get("ticker")).astype(str).tolist()
        asof = pd.to_datetime(np.asarray(b.meta.get("asof_date"))).dt.date.tolist()
        if not asof:
            raise ValueError("Batch missing asof_date meta")

        hmm = RegimeHMM.fit(
            run_dir=run_dir,
            tickers=tickers,
            min_asof=min(asof),
            max_asof=max(asof),
            seed=1337,
        )
        _save_pickle(run_dir / "hmm.pkl", hmm)
        artifact_path = run_dir / "artifact.json"
        _write_json(
            artifact_path,
            {
                "runner": self.name,
                "hmm_path": "hmm.pkl",
                "trend_state": int(hmm.trend_state),
                "tickers_used": list(hmm.tickers),
            },
        )
        return TrainedArtifact(runner_name=self.name, path=artifact_path)

    def train_with_val(self, train_batches: Iterable[StandardBatch], val_batches: Iterable[StandardBatch], *, run_dir: Path):  # noqa: ANN001
        # Unsupervised; ignore val.
        return self.train(train_batches, run_dir=run_dir)

    def predict(self, batches: Iterable[StandardBatch], *, artifact):  # noqa: ANN001
        payload = _read_json(Path(artifact.path))
        hmm = _load_pickle(Path(artifact.path).parent / payload.get("hmm_path", "hmm.pkl"))
        rows = []
        for b in batches:
            p = hmm.p_trend_for_batch(b)
            sample_ids = np.asarray(b.meta.get("sample_id"))
            setups = np.asarray(b.meta.get("setup"))
            tickers = np.asarray(b.meta.get("ticker"))
            asof_dates = np.asarray(b.meta.get("asof_date"))
            y_true = np.asarray(b.y).astype(int)
            for sid, sc, yy, ss, tkr, d in zip(
                sample_ids.tolist(),
                p.tolist(),
                y_true.tolist(),
                (setups.tolist() if setups is not None else [None] * len(y_true)),
                (tickers.tolist() if tickers is not None else [None] * len(y_true)),
                (asof_dates.tolist() if asof_dates is not None else [None] * len(y_true)),
                strict=True,
            ):
                rows.append(
                    {
                        "sample_id": str(sid),
                        "setup": str(ss) if ss is not None else None,
                        "ticker": str(tkr) if tkr is not None else None,
                        "asof_date": d,
                        "y_true": int(yy),
                        "score": float(sc),
                    }
                )
        return pd.DataFrame(rows)

