from __future__ import annotations

import numpy as np
from typing import Tuple, Any, Dict
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scipy.stats import loguniform
import umap


class S21AgePipelineManager:
    def __init__(self, random_state = 42, n_iter = 10) -> None:
        self.random_state = random_state
        self.n_iter = n_iter

        # will show up after calling fit_*
        self.linear_estimator_ = None
        self.linear_params_ = None
        self.forest_estimator_ = None
        self.forest_params_ = None


    # returns ridge pipeline
    def _linear_pipeline(self, reduce) -> Pipeline:
        if reduce is None:
            return Pipeline([
                ("scaler", StandardScaler(with_mean=False)),  # false for compatibility without dimensionality reduction
                ("model", Ridge()),
            ])

        key = reduce.lower() if isinstance(reduce, str) else None

        if key == "pca":
            return Pipeline([
                ("to_dense", FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)),
                ("scaler", StandardScaler(with_mean=True)),
                ("reduce", PCA()),
                ("model", Ridge()),
            ])

        elif key == "umap":
            return Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("reduce", umap.UMAP()),
                ("model", Ridge()),
            ])

        else:
            raise ValueError("Enter valid reduce (None, PCA or UMAP)")


    # returns forest pipeline
    def _forest_pipeline(self, reduce) -> Pipeline:
        if reduce is None:
            return Pipeline([
                ("model", RandomForestRegressor(n_jobs=-1)),
            ])

        key = reduce.lower() if isinstance(reduce, str) else None

        if key == "pca":
            return Pipeline([
                ("to_dense", FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)),
                ("scaler", StandardScaler(with_mean=True)),
                ("reduce", PCA()),
                ("model", RandomForestRegressor(n_jobs=-1)),
            ])

        elif key == "umap":
            return Pipeline([
                ("reduce", umap.UMAP()),
                ("model", RandomForestRegressor(n_jobs=-1)),
            ])

        else:
            raise ValueError("Enter valid reduce (None, PCA or UMAP)")


    # updates params in case of using PCA or UMAP
    def _param_updater(self, params: Dict[str, Any], reduce) -> Dict[str, Any]:
        key = reduce.lower() if isinstance(reduce, str) else None

        if key == "pca":
            params.update({
                "reduce__n_components": [8, 16, 32],
            })

        elif key == "umap":
            params.update({
                "reduce__n_components": [16, 32, 64],
                "reduce__n_neighbors": [10, 25, 50],
                "reduce__min_dist": [0.0, 0.3],
            })

        return params


    # searchs for ridge hyperparamethers usirng _linear_pipeline
    def fit_linear(self, X: csr_matrix, y: np.ndarray, reduce=None) -> "S21AgePipelineManager":
        params = {
            "model__alpha": loguniform(1e-3, 1e3),
        }

        params = self._param_updater(params, reduce)

        n = y.shape[0]
        cv_single = [(np.arange(n), np.arange(n))]

        search = RandomizedSearchCV(
            self._linear_pipeline(reduce),
            param_distributions=params,
            n_iter=self.n_iter,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=self.random_state,
            refit=True,
            verbose=0,
            cv=cv_single,
        )

        search.fit(X, y)

        self.linear_estimator_ = search.best_estimator_
        self.linear_params_ = search.best_params_

        return self


    # searchs for ridge hyperparamethers usirng _forest_pipeline
    def fit_forest(self, X: csr_matrix, y: np.ndarray, reduce=None) -> "S21AgePipelineManager":
        params = {
            "model__n_estimators": [150, 300],
            "model__max_depth": [None, 16],
            "model__min_samples_split": [2, 6],
            "model__min_samples_leaf": [1, 3],
            "model__max_features": ["sqrt", 0.4],
            "model__bootstrap": [True],
            "model__max_samples": [None, 0.7],
            "model__min_weight_fraction_leaf": [0.0, 0.01],
            "model__max_leaf_nodes": [None, 750],
            "model__min_impurity_decrease": [0.0, 1e-3],
            "model__ccp_alpha": [0.0, 5e-4],
        }

        params = self._param_updater(params, reduce)

        n = y.shape[0]
        cv_single = [(np.arange(n), np.arange(n))]

        search = RandomizedSearchCV(
            self._forest_pipeline(reduce),
            param_distributions=params,
            n_iter=self.n_iter,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=self.random_state,
            refit=True,
            verbose=0,
            cv=cv_single,
        )

        search.fit(X, y)

        self.forest_estimator_ = search.best_estimator_
        self.forest_params_ = search.best_params_

        return self


    # gets linear model & it's params
    def get_linear(self) -> Tuple[Pipeline, Dict[str, Any]]:
        linear_estimator_, linear_params_ = self.linear_estimator_, self.linear_params_

        if linear_estimator_ is None or linear_params_ is None:
            raise RuntimeError("Call fit_ before calling get_")

        return linear_estimator_, linear_params_


    # gets forest model & it's params
    def get_forest(self) -> Tuple[Pipeline, Dict[str, Any]]:
        forest_estimator_, forest_params_ = self.forest_estimator_, self.forest_params_

        if forest_estimator_ is None or forest_params_ is None:
            raise RuntimeError("Call fit_ before calling get_")

        return forest_estimator_, forest_params_

    
    # shows quality of models
    def evaluate(
        self,
        models: tuple,
        X_train: csr_matrix,
        X_val: csr_matrix,
        y: np.ndarray
    ) -> None:
        if self.linear_estimator_ is None and self.forest_estimator_ is None:
            raise RuntimeError("Call fit_linear/fit_forest before evaluate")

        def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
            return {
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "R2": float(r2_score(y_true, y_pred)),
            }

        for name, est in models:
            y_pred_tr = est.predict(X_train)
            y_pred_val = est.predict(X_val)

            m_tr = _metrics(y, y_pred_tr)
            m_val = _metrics(y, y_pred_val)

            print(f"{name} [train] -> MAE: {m_tr['MAE']:.3f}, RMSE: {m_tr['RMSE']:.3f}, R2: {m_tr['R2']:.3f}")
            print(f"{name} [val]   -> MAE: {m_val['MAE']:.3f}, RMSE: {m_val['RMSE']:.3f}, R2: {m_val['R2']:.3f}")


        return None