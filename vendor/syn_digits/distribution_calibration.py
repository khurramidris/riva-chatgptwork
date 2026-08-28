"""
Distribution Calibration Optimization Algorithm

This module implements optimization algorithms for calibrating digital twin distributions
to match empirical distributions using various divergence measures.

The optimization problem is:
min_{θ ∈ Δ^{n+K-1}} Σ_{j=1}^m D(P_j || A_j θ)

where:
- P_j is the empirical distribution of answers for question j
- A_j is a matrix mapping parameters θ to a synthetic distribution of answers for question j, 
  shape (K, n+K), where entry (k, i) is an indicator of whether the ith digital twin picked 
  the kth possible answer for question j. Note that the last K columns of A_j are the identity matrix.
- θ = [w, v] where w are digital twin weights and v are base distribution weights
- D is a divergence measure (KL, Chi2, Hellinger, Total Variation, etc.)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import os


import io
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from contextlib import redirect_stdout, redirect_stderr

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'outputs')
DISTRIBUTION_CALIBRATION_DIR = os.path.join(OUTPUT_DIR, 'distribution_calibration')
DISTRIBUTION_CALIBRATION_BATCH_DIR = os.path.join(OUTPUT_DIR, 'distribution_calibration_batch')


class DistributionCalibration:
    """
    Calibrating digital twin distributions to match empirical distributions.
    Supports multiple divergence measures and optimization methods.
    """
    
    def __init__(
        self, 
        P: np.ndarray, 
        Y_hat: np.ndarray, 
        dataset_name: str,
        personas: Dict[int, str],
        pids: List[int],
        possible_answers: np.ndarray, 
        divergence: str = 'kl', 
        method: str = 'mirror_descent', 
        reg_w: float = 1e-6, 
        reg_v: float = 1e-6, 
        reg_mse: float = 1e-6, 
        fit_persona_only: bool = False, 
        fit_dummy_only: bool = False, 
        weight_tol: Optional[float] = None,
        max_iter: int = 1000, 
        tol: float = 1e-4, 
        learning_rate: float = 1, 
        train_test_ratio: float = 0.8, 
        random_state: int = 42,
        adaptive_lr: bool = True, 
        max_grad_norm: float = 10.0, 
    ) -> None:
        """
        Initialize the object.
        
        Parameters:
        -----------
        P : np.ndarray, shape (m, K)
            Empirical distributions for m questions, each with K possible answers
        Y_hat : np.ndarray, shape (m, n)
            Digital twin answers for m questions and n twins
        dataset_name : str
            Name of the dataset. Used for saving the results. If None, no results will be saved.
        personas : Dict[int, str]
            Dictionary mapping persona IDs (pids) to persona descriptions/names. 
            Used for printing the top personas in run_full_workflow.
        pids : List[int]
            List of persona IDs corresponding to each digital twin (length n).
            The i-th element is the persona ID for the i-th digital twin.
            Used to map persona indices to persona IDs when printing top personas.
        possible_answers : np.ndarray, shape (K,)
            Possible answers for the questions, assumed to be the same for all questions
        divergence : str
            Divergence measure: 'tv', 'chi2', 'kl', 'hellinger', 'ks', 'l1', 'l2'. Default: 'kl'.
        method : str
            Optimization method: 'mirror_descent' or 'projected_gradient'. Default: 'mirror_descent'.
        reg_w : float
            Regularization parameter for an l2 regularization on the digital twin weights. Default: 1e-6.
        reg_v : float
            Regularization parameter for an l2 regularization on the base distribution weights. Default: 1e-6.
        reg_mse : float
            Regularization parameter for an additional mean squared loss. Default: 1e-6.
        fit_persona_only : bool
            Whether to only fit the persona weights. Default: False.
        fit_dummy_only : bool
            Whether to only fit the dummy weights. Default: False.
        weight_tol : float
            Keep weights greater than weight_tol. Default: None.
        max_iter : int
            Maximum number of iterations. Default: 1000.
        tol : float
            Convergence tolerance. Default: 1e-4.
        learning_rate : float
            Learning rate for gradient-based methods. Default: 1.
        train_test_ratio : float
            Ratio of questions to use for training. Default: 0.8.
        random_state : int
            Random seed for train-test splitting. Default: 42.
        adaptive_lr : bool
            Whether to use adaptive learning rate. Default: True.
        max_grad_norm : float
            Maximum gradient norm for clipping. Default: 10.0.
        """
        self.P = P
        self.K = P.shape[1]
        self.Y_hat = Y_hat
        self.m, self.n = Y_hat.shape

        # Check that m, n, and K are positive
        if self.m == 0 or self.n == 0 or self.K == 0:
            raise ValueError("m, n, and K must be positive")
        
        # Check that the length of P is m
        if self.m != len(P):
            raise ValueError(f"Length of P ({len(P)}) does not match the number of rows in Y_hat ({self.m})")
        
        # Check that the length of Y_hat is m
        if self.m != len(Y_hat):
            raise ValueError(f"Length of Y_hat ({len(Y_hat)}) does not match the number of rows in P ({self.m})")
        
        # Store personas and pids
        self.personas = personas
        self.pids = pids
        
        # Validate pids length matches n
        if len(pids) != self.n:
            raise ValueError(f"Length of pids ({len(pids)}) does not match the number of digital twins ({self.n})")
        
        # Store possible answers as a numpy array
        self.possible_answers = possible_answers if isinstance(possible_answers, np.ndarray) else np.array(possible_answers)

        # Check that the number of possible answers is indeed K
        if self.K != len(possible_answers):
            raise ValueError(f"Number of possible answers ({len(possible_answers)}) does not match the number of columns in P ({self.K})")

        # Store divergence
        self.divergence = divergence
        self.admissible_divergences = ['tv', 'chi2', 'kl', 'hellinger', 'ks', 'l1', 'l2']

        # Check that the divergence given is admissible
        if divergence not in self.admissible_divergences:
            raise ValueError(f"Unsupported divergence measure: {divergence}")

        # Store method
        self.method = method
        if method not in ['mirror_descent', 'projected_gradient']:
            raise ValueError(f"Unknown method: {method}")
        
        # Store fit persona only and fit dummy only
        self.fit_persona_only = fit_persona_only
        self.fit_dummy_only = fit_dummy_only
        if fit_persona_only and fit_dummy_only:
            raise ValueError("Cannot fit-persona-only and fit-dummy-only at the same time")
        
        # Store dataset name
        self.dataset_name = dataset_name
        
        # Store results directory
        if dataset_name is not None:
            self.results_dir = os.path.join(DISTRIBUTION_CALIBRATION_DIR, dataset_name)
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            self.results_figures_dir = os.path.join(self.results_dir, 'figures')
            if not os.path.exists(self.results_figures_dir):
                os.makedirs(self.results_figures_dir)
            self.results_tables_dir = os.path.join(self.results_dir, 'tables')
            if not os.path.exists(self.results_tables_dir):
                os.makedirs(self.results_tables_dir)
        else:
            self.results_dir = None
            self.results_figures_dir = None
            self.results_tables_dir = None
        
        # Store regularization parameters
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.reg_mse = reg_mse
        
        # Store weight tolerance
        self.weight_tol = weight_tol
        
        # Store maximum number of iterations
        self.max_iter = max_iter
        self.tol = tol
        
        # Store learning rate
        self.learning_rate = learning_rate
        
        # Store train-test ratio
        self.train_test_ratio = train_test_ratio
        self.random_state = random_state
        
        # Store adaptive learning rate
        self.adaptive_lr = adaptive_lr
        
        # Store maximum gradient norm
        self.max_grad_norm = max_grad_norm
        
        # Split data into train and test sets
        self._split_train_test()
        
        # Construct matrices A_j for each question (both training and test sets)
        self._construct_A_matrices_train()
        self._construct_A_matrices_test()
        
        # Initialize parameters
        self.theta = None
        self.objective_history = []
        
        # Store train performance
        self.train_performance = None
        self.baseline_train_performance = None

        # Store test performance
        self.test_performance = None
        self.baseline_test_performance = None
        
    def _split_train_test(self):
        """Split questions into training and testing sets."""
        np.random.seed(self.random_state)
        n_train = int(self.m * self.train_test_ratio)
        
        # Randomly shuffle question indices
        all_indices = np.arange(self.m)
        np.random.shuffle(all_indices)
        
        # Split into train and test indices
        self.train_indices = all_indices[:n_train]
        self.test_indices = all_indices[n_train:]
        
        # Create training data
        self.P_train = self.P[self.train_indices]
        self.P_train_mean = self.P_train @ self.possible_answers.T
        self.Y_hat_train = self.Y_hat[self.train_indices]
        self.m_train = self.Y_hat_train.shape[0]
        # Construct a m_train * (n+K) matrix as [Y_hat_train, possible_answers (copied m_train times)] for MSE calculation
        # mask with zeros if fit_persona_only or fit_dummy_only
        if not self.fit_persona_only and not self.fit_dummy_only:
            self.mse_train_matrix = np.hstack([self.Y_hat_train, np.tile(self.possible_answers, (self.m_train, 1))])
        elif self.fit_persona_only:
            self.mse_train_matrix = np.hstack([self.Y_hat_train, np.zeros((self.m_train, self.K))])
        elif self.fit_dummy_only:
            self.mse_train_matrix = np.hstack([np.zeros((self.m_train, self.n)), np.tile(self.possible_answers, (self.m_train, 1))])
        
        # Create test data
        self.P_test = self.P[self.test_indices]
        self.P_test_mean = self.P_test @ self.possible_answers.T
        self.Y_hat_test = self.Y_hat[self.test_indices]
        self.m_test = self.Y_hat_test.shape[0]
        # Construct a m_test * (n+K) matrix as [Y_hat_test, possible_answers (copied m_test times)] for MSE calculation
        # mask with zeros if fit_persona_only or fit_dummy_only
        if not self.fit_persona_only and not self.fit_dummy_only:
            self.mse_test_matrix = np.hstack([self.Y_hat_test, np.tile(self.possible_answers, (self.m_test, 1))])
        elif self.fit_persona_only:
            self.mse_test_matrix = np.hstack([self.Y_hat_test, np.zeros((self.m_test, self.K))])
        elif self.fit_dummy_only:
            self.mse_test_matrix = np.hstack([np.zeros((self.m_test, self.n)), np.tile(self.possible_answers, (self.m_test, 1))])
        
        print(f"Train-test split: {len(self.train_indices)} training questions, {len(self.test_indices)} test questions")
        
    def _construct_A_matrices_train(self):
        """
        Construct the A_j matrices for each question (training set only).
        
        A_j is a matrix mapping parameters θ to a synthetic distribution of answers for question j, 
        shape (K, n+K), where entry (k, i) is an indicator of whether the ith digital twin picked 
        the kth possible answer for question j. Note that the last K columns of A_j are the identity matrix.
        
        Returns:
            None
        
        Stores:
            self.A_matrices_train: A numpy array of shape (m_train, K, n+K) containing the A_j matrices for each training question.
        """
        self.A_matrices_train = []
        
        for j in range(len(self.train_indices)):
            # Indicator matrix for digital twin answers
            Y_indicator = np.zeros((self.K, self.n))
            for i in range(self.n):
                answer_idx = np.where(self.possible_answers == self.Y_hat_train[j, i])[0]
                if len(answer_idx) > 0:
                    Y_indicator[answer_idx[0], i] = 1
            
            # Identity matrix for v parameters
            I_K = np.eye(self.K)
            
            # Combine: A_j = [Y_indicator, I_K]
            A_j = np.hstack([Y_indicator, I_K])
            self.A_matrices_train.append(A_j)
        self.A_matrices_train = np.array(self.A_matrices_train)
    
    def _construct_A_matrices_test(self):
        """
        Construct the A_j matrices for each question (test set only).
        
        A_j is a matrix mapping parameters θ to a synthetic distribution of answers for question j, 
        shape (K, n+K), where entry (k, i) is an indicator of whether the ith digital twin picked 
        the kth possible answer for question j. Note that the last K columns of A_j are the identity matrix.
        
        Returns:
            None
        
        Stores:
            self.A_matrices_test: A numpy array of shape (m_test, K, n+K) containing the A_j matrices for each test question.
        """
        self.A_matrices_test = []
        
        for j in range(len(self.test_indices)):
            # Get the test question index
            test_question_idx = self.test_indices[j]
            
            # Indicator matrix for digital twin answers
            Y_indicator = np.zeros((self.K, self.n))
            for i in range(self.n):
                answer_idx = np.where(self.possible_answers == self.Y_hat[test_question_idx, i])[0]
                if len(answer_idx) > 0:
                    Y_indicator[answer_idx[0], i] = 1
            
            # Identity matrix for v parameters
            I_K = np.eye(self.K)
            
            # Combine: A_j = [Y_indicator, I_K]
            A_j = np.hstack([Y_indicator, I_K])
            self.A_matrices_test.append(A_j)
        self.A_matrices_test = np.array(self.A_matrices_test)
    
    def compute_Q_distributions_train(
        self, 
        theta: np.ndarray
    ) -> np.ndarray:
        """Compute synthetic distributions Q_j for training questions given theta.
        
        Args:
            theta: Parameter vector of shape (n+K,).
            
        Returns:
            Array of shape (m_train, K) containing synthetic distributions for each training question.
        """
        return self.A_matrices_train @ theta
    
    def compute_Q_distributions_test(
        self, 
        theta: np.ndarray
    ) -> np.ndarray:
        """Compute synthetic distributions Q_j for test questions given theta.
        
        Args:
            theta: Parameter vector of shape (n+K,).
            
        Returns:
            Array of shape (m_test, K) containing synthetic distributions for each test question.
        """
        return self.A_matrices_test @ theta
    
    def compute_CDFs(
        self, 
        distributions: np.ndarray
        ) -> np.ndarray:
        """Compute cumulative distribution functions (CDFs) from probability distributions.
        
        Args:
            distributions: Probability distribution array of shape (K,).
            
        Returns:
            CDF array of shape (K,).
        """
        return np.cumsum(distributions)
    
    def compute_divergence(
        self, 
        P: np.ndarray, 
        Q: np.ndarray, 
        div_type: Optional[str] = None
    ) -> float:
        """Compute the specified divergence between probability distributions P and Q.
        
        Args:
            P: True probability distribution of shape (K,).
            Q: Predicted/synthetic probability distribution of shape (K,).
            div_type: Divergence type to compute. If None, uses self.divergence. Default: None.
            
        Returns:
            Divergence value as a float.
            
        Raises:
            ValueError: If div_type is not in admissible_divergences.
        """
        if div_type is None:
            div_type = self.divergence
        elif div_type not in self.admissible_divergences:
            raise ValueError(f"Unsupported divergence measure: {div_type}")

        if div_type == 'tv':
            return 0.5 * np.sum(np.abs(P - Q))
        elif div_type == 'chi2':
            # Clip Q to avoid numerical instability
            eps = 1e-6
            Q_clipped = np.maximum(Q, eps)
            return np.sum(P**2 / Q_clipped) - 1
        elif div_type == 'kl':
            # Clip P and Q to avoid numerical instability
            eps = 1e-6
            P_clipped = np.maximum(P, eps)
            Q_clipped = np.maximum(Q, eps)
            return np.sum(P_clipped * np.log(P_clipped / Q_clipped))
        elif div_type == 'hellinger':
            # Clip P and Q to avoid sqrt of negative values
            eps = 1e-8
            P_clipped = np.maximum(P, eps)
            Q_clipped = np.maximum(Q, eps)
            return 1 - np.sum(np.sqrt(P_clipped * Q_clipped))
        elif div_type in ['ks', 'l1', 'l2']:
            # For ordinal measures, compare CDFs
            F_P = self.compute_CDFs(P)
            F_Q = self.compute_CDFs(Q)
            if div_type == 'ks':
                return np.max(np.abs(F_P - F_Q))
            elif div_type == 'l1':
                return np.sum(np.abs(F_P - F_Q))
            elif div_type == 'l2':
                return np.sum((F_P - F_Q)**2)
        else:
            # This should never be reached due to the check above, but included for completeness
            raise ValueError(f"Unsupported divergence measure: {div_type}")
    
    def compute_train_objective(
        self, 
        theta: np.ndarray
    ) -> float:
        """Compute the total objective value for training data.
        
        The objective includes:
        - Average divergence across training questions
        - L2 regularization on digital twin weights (w)
        - L2 regularization on base distribution weights (v)
        - Mean squared error regularization
        
        Args:
            theta: Parameter vector of shape (n+K,).
            
        Returns:
            Total objective value as a float.
        """
        Q = self.compute_Q_distributions_train(theta)
        total_divergence = 0
        for j in range(len(self.train_indices)):
            total_divergence += self.compute_divergence(self.P_train[j], Q[j])
        reg_w_loss = self.reg_w * theta[:self.n].T @ theta[:self.n] if not self.fit_dummy_only else 0
        reg_v_loss = self.reg_v * theta[self.n:].T @ theta[self.n:] if not self.fit_persona_only else 0
        reg_mse_loss = self.reg_mse * np.mean((self.P_train_mean - self.mse_train_matrix @ theta)**2)
        return total_divergence / len(self.train_indices) + reg_w_loss + reg_v_loss + reg_mse_loss
    
    def compute_gradient(
        self, 
        theta: np.ndarray
    ) -> np.ndarray:
        """Compute the gradient of the objective function for training data.
        
        Args:
            theta: Parameter vector of shape (n+K,).
            
        Returns:
            Gradient vector of shape (n+K,).
        """
        Q = self.compute_Q_distributions_train(theta)
        grad = np.zeros_like(theta)
        
        for j in range(len(self.train_indices)):
            Q_j = Q[j]
            P_j = self.P_train[j]
            
            if self.divergence == 'tv':
                diff = P_j - Q_j
                abs_diff = np.abs(diff)
                delta = 1e-6
                # For small differences, use diff/delta as an approximation of the sign function
                smooth_sign = np.where(abs_diff < delta, 
                                    diff / delta,  # Continuous approximation
                                    np.sign(diff))  # True sign
                grad_j = - self.A_matrices_train[j].T @ smooth_sign
                
                grad += grad_j
                
            elif self.divergence == 'chi2':
                # Gradient for chi2 with clipping
                eps = 1e-6
                Q_j_clipped = np.maximum(Q_j, eps)
                grad_j = - self.A_matrices_train[j].T @ (P_j**2 / (Q_j_clipped**2))
                grad += grad_j
                
            elif self.divergence == 'kl':
                # Gradient for KL with clipping
                eps = 1e-6
                P_j_clipped = np.maximum(P_j, eps)
                Q_j_clipped = np.maximum(Q_j, eps)
                grad_j = -self.A_matrices_train[j].T @ (P_j_clipped / Q_j_clipped)
                grad += grad_j
                
            elif self.divergence == 'hellinger':
                # Gradient for Hellinger with clipping
                eps = 1e-8
                P_j_clipped = np.maximum(P_j, eps)
                Q_j_clipped = np.maximum(Q_j, eps)
                grad_j = -0.5 * self.A_matrices_train[j].T @ (np.sqrt(P_j_clipped / Q_j_clipped))
                grad += grad_j
                
            elif self.divergence in ['ks', 'l1', 'l2']:
                # Gradient for ordinal measures
                F_P = self.compute_CDFs(P_j)
                F_Q = self.compute_CDFs(Q_j)
                diff = F_P - F_Q
                C = np.tril(np.ones((self.K, self.K))) # cumulative sum matrix, a lower triangular matrix of ones
                
                if self.divergence == 'ks':
                    abs_diff = np.abs(diff)
                    alpha = 10.0
                    exp_terms = np.exp(alpha * abs_diff)
                    sum_exp = np.sum(exp_terms)
                    grad_weights = exp_terms / sum_exp
                    delta = 1e-6
                    # For small differences, use diff/delta as an approximation of the sign function
                    smooth_sign = np.where(abs_diff < delta, 
                                        diff / delta,  # Continuous approximation
                                        np.sign(diff))  # True sign
                    weighted_diff = grad_weights * smooth_sign
                    grad_j = - self.A_matrices_train[j].T @ (C.T @ weighted_diff)
                    
                elif self.divergence == 'l1':
                    abs_diff = np.abs(diff)
                    delta = 1e-6
                    # For small differences, use diff/delta as an approximation of the sign function
                    smooth_sign = np.where(abs_diff < delta, 
                                        diff / delta,  # Continuous approximation
                                        np.sign(diff))  # True sign
                    grad_j = - self.A_matrices_train[j].T @ (C.T @ smooth_sign)

                elif self.divergence == 'l2':
                    # For L2 on CDFs: ||F_{P_j} - F_{Q_j}||_2^2 = ||C(P_j - Q_j)||_2^2
                    # \grad_theta ||F_{P_j} - F_{Q_j}||_2^2 = -2 * C^T * C * (F_{P_j} - F_{Q_j})
                    grad_j = -2 * self.A_matrices_train[j].T @ (C.T @ C @ diff)
                    
                grad += grad_j
        
        reg_param_grad = np.concatenate([2 * self.reg_w * theta[:self.n], 2 * self.reg_v * theta[self.n:]])
        reg_mse_grad = 2 * self.reg_mse * self.mse_train_matrix.T @ (self.mse_train_matrix @ theta - self.P_train_mean) / self.m_train
        
        # mask some coordinates of the gradient with zeros if fit_persona_only or fit_dummy_only
        grad = grad / len(self.train_indices) + reg_param_grad + reg_mse_grad
        if self.fit_persona_only:
            grad[self.n:] = 0
        if self.fit_dummy_only:
            grad[:self.n] = 0

        return grad
    
    def compute_naive_baseline_train(self) -> np.ndarray:
        """Compute naive averaging baseline for training questions.
        
        The naive baseline simply counts the frequency of each answer across all digital twins
        for each question and normalizes to a probability distribution.
        
        Returns:
            Array of shape (m_train, K) containing naive baseline distributions.
        """
        Q_naive = np.zeros((len(self.train_indices), self.K))
        
        for j in range(len(self.train_indices)):
            # Count answers for each possible answer
            answer_counts = np.zeros(self.K)
            for i in range(self.n):
                answer_idx = np.where(self.possible_answers == self.Y_hat_train[j, i])[0]
                if len(answer_idx) > 0:
                    answer_counts[answer_idx[0]] += 1
            
            # Convert to probabilities (handle division by zero)
            total = answer_counts.sum()
            if total > 0:
                Q_naive[j] = answer_counts / total
            else:
                # If no answers found, use uniform distribution
                Q_naive[j] = np.ones(self.K) / self.K
        
        return Q_naive

    def compute_naive_baseline_test(self) -> np.ndarray:
        """Compute naive averaging baseline for test questions.
        
        The naive baseline simply counts the frequency of each answer across all digital twins
        for each question and normalizes to a probability distribution.
        
        Returns:
            Array of shape (m_test, K) containing naive baseline distributions.
        """
        Q_naive = np.zeros((len(self.test_indices), self.K))
        
        for j in range(len(self.test_indices)):
            # Count answers for each possible answer
            answer_counts = np.zeros(self.K)
            for i in range(self.n):
                answer_idx = np.where(self.possible_answers == self.Y_hat_test[j, i])[0]
                if len(answer_idx) > 0:
                    answer_counts[answer_idx[0]] += 1
            
            # Convert to probabilities (handle division by zero)
            total = answer_counts.sum()
            if total > 0:
                Q_naive[j] = answer_counts / total
            else:
                # If no answers found, use uniform distribution
                Q_naive[j] = np.ones(self.K) / self.K
        
        return Q_naive
    
    def project_to_simplex(self, theta: np.ndarray) -> np.ndarray:
        """Project theta to the probability simplex.
        
        Ensures non-negativity and normalizes to sum to 1.
        
        Args:
            theta: Parameter vector to project.
            
        Returns:
            Projected vector on the probability simplex.
        """
        # Ensure non-negativity
        theta = np.maximum(theta, 1e-8)
        
        # Project to sum = 1 (always normalize, not just when total > 1)
        total = np.sum(theta)
        if total > 0:
            theta = theta / total
        
        return theta
    
    def optimize(self) -> np.ndarray:
        """Run the optimization algorithm.
        
        Initializes theta uniformly and runs either projected gradient descent or mirror descent
        depending on the specified method. After optimization, optionally prunes small weights
        if weight_tol is set.
        
        Returns:
            Optimized parameter vector theta of shape (n+K,).
        """
        # Initialize theta uniformly
        n_params = self.n + self.K
        if self.fit_persona_only:
            self.theta = np.ones(n_params) / self.n
            self.theta[self.n:] = 0
        elif self.fit_dummy_only:
            self.theta = np.ones(n_params) / self.K
            self.theta[:self.n] = 0
        else:
            self.theta = np.ones(n_params) / n_params
        
        if self.method == 'projected_gradient':
            self._projected_gradient_descent()
        elif self.method == 'mirror_descent':
            self._mirror_descent()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if self.weight_tol is not None:
            n_weights_set_to_0 = np.sum(self.theta < self.weight_tol)
            self.theta = np.where(self.theta > self.weight_tol, self.theta, 0)
            self.theta = self.theta / np.sum(self.theta) # renormalize
            print(f"Set {n_weights_set_to_0} weights less than {self.weight_tol} to 0...")
        
        return self.theta
    
    def _projected_gradient_descent(self) -> None:
        """
        Projected gradient descent implementation.
        """
        print(f"Running projected gradient descent with {self.divergence} divergence...")
        
        current_lr = self.learning_rate
        
        for iteration in tqdm(range(self.max_iter), desc="Projected Gradient Descent"):
            # Compute objective
            obj_val = self.compute_train_objective(self.theta)
            self.objective_history.append(obj_val)
            
            # Compute gradient
            grad = self.compute_gradient(self.theta)
            
            # Clip gradient if too large
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.max_grad_norm:
                grad = grad * (self.max_grad_norm / grad_norm)
            
            # Update step
            theta_new = self.theta - current_lr * grad
            
            # Project to simplex
            theta_new = self.project_to_simplex(theta_new)
            
            # Check convergence
            if obj_val < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            # Adaptive learning rate
            if self.adaptive_lr and iteration > 0:
                if len(self.objective_history) >= 2:
                    if self.objective_history[-1] > self.objective_history[-2]:
                        # Objective increased, reduce learning rate
                        current_lr *= 0.9
                        current_lr = max(current_lr, 1e-6)
                    elif self.objective_history[-1] < self.objective_history[-2] * 0.99:
                        # Objective decreased significantly, increase learning rate slightly
                        current_lr *= 1.01
                        current_lr = min(current_lr, self.learning_rate)
            
            self.theta = theta_new
        
        print(f"Final objective: {self.compute_train_objective(self.theta):.6f}")
    
    def _mirror_descent(self) -> None:
        """Mirror descent implementation using KL divergence as Bregman divergence.
        """
        print(f"Running mirror descent with {self.divergence} divergence...")
        
        # Initialize dual variables (log-space)
        log_theta = np.log(self.theta + 1e-8)
        
        for iteration in tqdm(range(self.max_iter), desc="Mirror Descent"):
            # Compute objective
            obj_val = self.compute_train_objective(self.theta)
            self.objective_history.append(obj_val)
            
            # Compute gradient
            grad = self.compute_gradient(self.theta)
            
            # Clip gradient to prevent overflow in exponential
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.max_grad_norm:
                grad = grad * (self.max_grad_norm / grad_norm)
            
            # Update in dual space
            log_theta_new = log_theta - self.learning_rate * grad
            
            # Clip log_theta to prevent extreme values
            log_theta_new = np.clip(log_theta_new, -10.0, 10.0)
            
            # Transform back to primal space
            theta_new = np.exp(log_theta_new)
            theta_new = self.project_to_simplex(theta_new)
            
            # Check convergence
            if obj_val < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            self.theta = theta_new
            log_theta = np.log(theta_new + 1e-8)
        
        print(f"Final objective: {self.compute_train_objective(self.theta):.6f}")
    
    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract digital twin weights (w) and base distribution weights (v) from theta.
        
        Returns:
            Tuple of (w, v) where:
                - w: Digital twin weights of shape (n,)
                - v: Base distribution weights of shape (K,)
                
        Raises:
            ValueError: If optimize() has not been called yet.
        """
        if self.theta is None:
            raise ValueError("Must run optimize() first")
        
        w = self.theta[:self.n]  # Digital twin weights
        v = self.theta[self.n:]  # Base distribution weights
        
        return w, v
    
    @staticmethod
    def _standard_error(values: List[float]) -> float:
        """Compute the standard error of a list of values."""
        n = len(values)
        if n <= 1:
            return 0.0
        return float(np.std(values, ddof=1) / np.sqrt(n))
    
    def evaluate_train_performance(
        self, 
        verbose: bool = True, 
        div_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate the performance on train data and compare with naive baseline.
        
        Args:
            verbose: Whether to print progress. Default: True.
            div_type: Divergence type to use. Default: None.
            
        Returns:
            Dictionary containing the performance metrics.
        """
        if self.theta is None:
            raise ValueError("Must run optimize() first")
        
        # Compute optimized distributions for training questions
        Q_optimized = self.compute_Q_distributions_train(self.theta)
        
        # Compute naive baseline
        Q_naive = self.compute_naive_baseline_train()
        
        # Evaluate optimized performance
        optimized_divergences = []
        for j in range(len(self.train_indices)):
            div = self.compute_divergence(self.P_train[j], Q_optimized[j], div_type)
            optimized_divergences.append(div)
        
        # Evaluate baseline performance
        baseline_divergences = []
        for j in range(len(self.train_indices)):
            div = self.compute_divergence(self.P_train[j], Q_naive[j], div_type)
            baseline_divergences.append(div)
        
        # Compute statistics (standard error instead of standard deviation)
        opt_avg = np.mean(optimized_divergences)
        opt_se = self._standard_error(optimized_divergences)
        baseline_avg = np.mean(baseline_divergences)
        baseline_se = self._standard_error(baseline_divergences)

        # Improvement
        improvement = baseline_avg - opt_avg
        improvement_pct = (improvement / baseline_avg) * 100 if baseline_avg > 0 else 0

        # Calculate the mean squared error of optimized and baseline distributions
        opt_mse = np.mean((self.P_train_mean - self.mse_train_matrix @ self.theta)**2)
        baseline_mse = np.mean((self.P_train_mean - np.mean(self.mse_train_matrix, axis=1))**2)
        
        if verbose:
            print(f"Train Performance with {self.divergence} divergence:")
            print(f"Optimized:  {opt_avg:.3f} ± {opt_se:.3f} (SE)")
            print(f"Baseline:   {baseline_avg:.3f} ± {baseline_se:.3f} (SE)")
            print(f"Improvement: {improvement:.3f} ({improvement_pct:.3f}%)")
            print(f"Optimized MSE: {opt_mse:.3f}")
            print(f"Baseline MSE: {baseline_mse:.3f}")
            
        # Store results
        self.train_performance = {
            'avg_divergence': opt_avg,
            'se_divergence': opt_se,
            'min_divergence': np.min(optimized_divergences),
            'max_divergence': np.max(optimized_divergences),
            'divergences': optimized_divergences,
            'mse': opt_mse
        }
        
        self.baseline_train_performance = {
            'avg_divergence': baseline_avg,
            'se_divergence': baseline_se,
            'min_divergence': np.min(baseline_divergences),
            'max_divergence': np.max(baseline_divergences),
            'divergences': baseline_divergences,
            'mse': baseline_mse
        }
        
        return {
            'optimized': self.train_performance,
            'baseline': self.baseline_train_performance,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            "Q_optimized": Q_optimized,
            "Q_naive": Q_naive
        }
    
    def evaluate_train_performance_all_div_types(
        self, 
        verbose: bool = False, 
        report: bool = True
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Evaluate the performance on train data for all divergence types.
        
        Args:
            verbose: Whether to print detailed performance for each divergence type. Default: False.
            report: Whether to print and return a summary DataFrame. Default: True.
            
        Returns:
            Tuple of (performance dictionary, summary DataFrame).
             - performance dictionary: Dictionary mapping divergence types to performance dictionaries.
             - summary DataFrame: DataFrame containing the performance metrics for all divergence types. Empty if report=False.
        """
        train_performance_all_div_types = {}
        for div_type in self.admissible_divergences:
            train_performance = self.evaluate_train_performance(verbose, div_type)
            train_performance_all_div_types[div_type] = train_performance
        df = pd.DataFrame()
        if report: # output a dataframe of the performance for all divergence types
            print(f"Train Performance with all divergence types (trained on {self.divergence} divergence):")
            for div_type in self.admissible_divergences:
                opt_avg = train_performance_all_div_types[div_type]['optimized']['avg_divergence']
                opt_se = train_performance_all_div_types[div_type]['optimized']['se_divergence']
                base_avg = train_performance_all_div_types[div_type]['baseline']['avg_divergence']
                base_se = train_performance_all_div_types[div_type]['baseline']['se_divergence']
                df[div_type] = [
                    f"{opt_avg:.3f} ± {opt_se:.3f}",
                    f"{base_avg:.3f} ± {base_se:.3f}"
                ]
            opt_mse = train_performance_all_div_types['kl']['optimized']['mse']
            base_mse = train_performance_all_div_types['kl']['baseline']['mse']
            df['mse'] = [f"{opt_mse:.3f}", f"{base_mse:.3f}"]
            df.index = ['Optimized', 'Baseline']
            print(df)
        return train_performance_all_div_types, df
    
    def evaluate_test_performance(
        self, 
        verbose: bool = True, 
        div_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate the performance on test data and compare with naive baseline.
        
        Args:
            verbose: Whether to print progress. Default: True.
            div_type: Divergence type to use. Default: None.
            
        Returns:
            Dictionary containing the performance metrics.
        """
        if self.theta is None:
            raise ValueError("Must run optimize() first")
        
        # Compute optimized distributions for test questions
        Q_optimized = self.compute_Q_distributions_test(self.theta)
        
        # Compute naive baseline
        Q_naive = self.compute_naive_baseline_test()
        
        # Evaluate optimized performance
        optimized_divergences = []
        for j in range(len(self.test_indices)):
            div = self.compute_divergence(self.P_test[j], Q_optimized[j], div_type)
            optimized_divergences.append(div)
        
        # Evaluate baseline performance
        baseline_divergences = []
        for j in range(len(self.test_indices)):
            div = self.compute_divergence(self.P_test[j], Q_naive[j], div_type)
            baseline_divergences.append(div)
        
        # Compute statistics (standard error instead of standard deviation)
        opt_avg = np.mean(optimized_divergences)
        opt_se = self._standard_error(optimized_divergences)
        baseline_avg = np.mean(baseline_divergences)
        baseline_se = self._standard_error(baseline_divergences)
        
        # Improvement
        improvement = baseline_avg - opt_avg
        improvement_pct = (improvement / baseline_avg) * 100 if baseline_avg > 0 else 0

        # Calculate the mean squared error of optimized and baseline distributions
        opt_mse = np.mean((self.P_test_mean - self.mse_test_matrix @ self.theta)**2)
        baseline_mse = np.mean((self.P_test_mean - np.mean(self.mse_test_matrix, axis=1))**2)
        
        if verbose:
            print(f"Test Performance with {self.divergence} divergence:")
            print(f"Optimized:  {opt_avg:.3f} ± {opt_se:.3f} (SE)")
            print(f"Baseline:   {baseline_avg:.3f} ± {baseline_se:.3f} (SE)")
            print(f"Improvement: {improvement:.3f} ({improvement_pct:.3f}%)")
            print(f"Optimized MSE: {opt_mse:.3f}")
            print(f"Baseline MSE: {baseline_mse:.3f}")
            
        # Store results
        self.test_performance = {
            'avg_divergence': opt_avg,
            'se_divergence': opt_se,
            'min_divergence': np.min(optimized_divergences),
            'max_divergence': np.max(optimized_divergences),
            'divergences': optimized_divergences,
            'mse': opt_mse
        }
        
        self.baseline_test_performance = {
            'avg_divergence': baseline_avg,
            'se_divergence': baseline_se,
            'min_divergence': np.min(baseline_divergences),
            'max_divergence': np.max(baseline_divergences),
            'divergences': baseline_divergences,
            'mse': baseline_mse
        }
        
        return {
            'optimized': self.test_performance,
            'baseline': self.baseline_test_performance,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            "Q_optimized": Q_optimized,
            "Q_naive": Q_naive
        }
    
    def evaluate_test_performance_all_div_types(
        self, 
        verbose: bool = False, 
        report: bool = True
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Evaluate the performance on test data for all divergence types.
        
        Args:
            verbose: Whether to print detailed performance for each divergence type. Default: False.
            report: Whether to print and return a summary DataFrame. Default: True.
            
        Returns:
            Tuple of (performance dictionary, summary DataFrame).
             - performance dictionary: Dictionary mapping divergence types to performance dictionaries.
             - summary DataFrame: DataFrame containing the performance metrics for all divergence types. Empty if report=False.
        """
        test_performance_all_div_types = {}
        for div_type in self.admissible_divergences:
            test_performance = self.evaluate_test_performance(verbose, div_type)
            test_performance_all_div_types[div_type] = test_performance
        df = pd.DataFrame()
        if report: # output a dataframe of the performance for all divergence types
            print(f"Test Performance with all divergence types (trained on {self.divergence} divergence):")
            for div_type in self.admissible_divergences:
                opt_avg = test_performance_all_div_types[div_type]['optimized']['avg_divergence']
                opt_se = test_performance_all_div_types[div_type]['optimized']['se_divergence']
                base_avg = test_performance_all_div_types[div_type]['baseline']['avg_divergence']
                base_se = test_performance_all_div_types[div_type]['baseline']['se_divergence']
                df[div_type] = [
                    f"{opt_avg:.3f} ± {opt_se:.3f}",
                    f"{base_avg:.3f} ± {base_se:.3f}"
                ]
            opt_mse = test_performance_all_div_types['kl']['optimized']['mse']
            base_mse = test_performance_all_div_types['kl']['baseline']['mse']
            df['mse'] = [f"{opt_mse:.3f}", f"{base_mse:.3f}"]
            df.index = ['Optimized', 'Baseline']
            print(df)
        return test_performance_all_div_types, df
    
    def get_train_test_info(self) -> Dict[str, Any]:
        """Get information about the train-test split.
        
        Returns:
            Dictionary containing:
                - 'n_total_questions': Total number of questions
                - 'n_train_questions': Number of training questions
                - 'n_test_questions': Number of test questions
                - 'train_test_ratio': Ratio used for splitting
                - 'train_indices': Array of training question indices
                - 'test_indices': Array of test question indices
        """
        return {
            'n_total_questions': self.m,
            'n_train_questions': len(self.train_indices),
            'n_test_questions': len(self.test_indices),
            'train_test_ratio': self.train_test_ratio,
            'train_indices': self.train_indices,
            'test_indices': self.test_indices
        }
    
    def get_top_personas_and_dummies(
        self, 
        top_num: int = 10
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Get the indices and weights of the top persona and dummy weights.
        
        Returns the top persona weights (digital twin weights) and top dummy weights
        (base distribution weights) sorted by weight value in descending order.
        
        Args:
            top_num: Number of top weights to return for each type. Default: 10.
            
        Returns:
            Tuple of (top_personas, top_dummies) where:
                - top_personas: Dictionary mapping persona index (0 to n-1) to weight value
                - top_dummies: Dictionary mapping dummy index (0 to K-1) to weight value
            
        Raises:
            ValueError: If optimize() has not been called yet.
        """
        if self.theta is None:
            raise ValueError("Must run optimize() first")
        top_personas, top_dummies = {}, {}
        if self.fit_persona_only:
            top_personas = {i: self.theta[:self.n][i] for i in np.argsort(self.theta[:self.n])[::-1][:top_num]}
            print(f"Only fitted persona weights, so only returning the top {top_num} personas...")
        elif self.fit_dummy_only:
            top_dummies = {i: self.theta[self.n:][i] for i in np.argsort(self.theta[self.n:])[::-1][:top_num]}
            print(f"Only fitted dummy weights, so only returning the top {top_num} dummies...")
        else:
            top_personas = {i: self.theta[:self.n][i] for i in np.argsort(self.theta[:self.n])[::-1][:top_num]}
            top_dummies = {i: self.theta[self.n:][i] for i in np.argsort(self.theta[self.n:])[::-1][:top_num]}
            print(f"Fitted both persona and dummy weights, so returning both the top {top_num} personas and top {top_num} dummies...")
        return top_personas, top_dummies
    
    def plot_convergence(self) -> None:
        """Plot the convergence history of the optimization.
        
        Displays the objective value over iterations on a log scale. The figure is automatically
        saved to results_figures_dir/convergence.pdf if dataset_name was provided during initialization.
        """
        if not self.objective_history:
            print("No optimization history available")
            return
        
        print(f"Plotting the convergence history for {self.divergence} divergence, {self.method} method...")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.objective_history)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Objective Value (log scale)', fontsize=14)
        plt.yscale('log')
        plt.tight_layout()
        
        # Save figure if results directory is available
        if self.results_figures_dir is not None:
            fig_path = os.path.join(self.results_figures_dir, 'convergence.pdf')
            plt.savefig(fig_path, format='pdf', bbox_inches='tight')
            print(f"Saved convergence plot to {fig_path}")
        
        plt.show()
    
    def plot_variance_ratio(self) -> None:
        """Plot the variance ratios between the true and synthetic distributions for test questions.
        
        Creates a histogram of variance ratios (synthetic variance / true variance) for all test questions.
        If theta is None, plots only the variance ratios between the true distribution and the 
        naive baseline distribution. If theta is not None, plots both the variance ratios between 
        the true distribution and the trained distribution, and between the trained distribution 
        and the naive baseline distribution.
        
        The figure is automatically saved to results_figures_dir/variance_ratio.pdf
        if dataset_name was provided during initialization.
        """
        Q_naive = self.compute_naive_baseline_test()
        if self.theta is None:
            print("No trained parameters available. Plotting the variance ratios between the true distribution and the naive baseline distribution...")
        else:
            print("Found trained parameters. Plotting the variance ratios between the true distribution and the trained distribution, and that between the trained distribution and the naive baseline distribution...")
            Q_trained = self.compute_Q_distributions_test(self.theta)
        
        naive_ratios, trained_ratios = [], []
        for question_idx in range(len(self.test_indices)):
            P_j = self.P_test[question_idx]
            Q_naive_j = Q_naive[question_idx]
            if self.theta is not None:
                Q_trained_j = Q_trained[question_idx] 
            true_avg_answer = P_j.dot(self.possible_answers)
            naive_avg_answer = Q_naive_j.dot(self.possible_answers)
            true_var = ((np.array(self.possible_answers) - true_avg_answer) ** 2 * P_j).sum()
            naive_var = ((np.array(self.possible_answers) - naive_avg_answer) ** 2 * Q_naive_j).sum()
            if self.theta is not None:
                trained_avg_answer = Q_trained_j.dot(self.possible_answers)
                trained_var = ((np.array(self.possible_answers) - trained_avg_answer) ** 2 * Q_trained_j).sum()
                trained_ratios.append(trained_var / true_var)
            naive_ratios.append(naive_var / true_var)

        plt.figure(figsize=(7, 4))
        plt.hist([r for r in naive_ratios if not np.isnan(r)], bins=30, color='skyblue', edgecolor='k', label='naive / true')
        if self.theta is not None:
            plt.hist([r for r in trained_ratios if not np.isnan(r)], bins=30, color='orange', edgecolor='k', label='trained / true')
        plt.xlabel('Variance ratio', fontsize=14)
        plt.ylabel('Number of questions', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Save figure if results directory is available
        if self.results_figures_dir is not None:
            fig_path = os.path.join(self.results_figures_dir, 'variance_ratio.pdf')
            plt.savefig(fig_path, format='pdf', bbox_inches='tight')
            print(f"Saved variance ratio plot to {fig_path}")
        
        plt.show()

        # print the mean and median of the variance ratios
        print(f"Mean of variance ratios (naive / true): {np.mean(naive_ratios):.4f}")
        print(f"Median of variance ratios (naive / true): {np.median(naive_ratios):.4f}")
        if self.theta is not None:
            print(f"Mean of variance ratios (trained / true): {np.mean(trained_ratios):.4f}")
            print(f"Median of variance ratios (trained / true): {np.median(trained_ratios):.4f}")
        print("\n\n")

    def plot_dist_difference(self, j: int) -> None:
        """Plot the difference between the true and synthetic distributions for the jth test question.
        
        Creates a bar chart comparing the true empirical distribution with synthetic distributions.
        If theta is None, plots only the true distribution and the naive baseline distribution.
        If theta is not None, plots the true distribution, the trained distribution, and the 
        naive baseline distribution.
        
        The figure is automatically saved to results_figures_dir/distribution_difference_question_{j}.pdf
        if dataset_name was provided during initialization.
        
        Args:
            j: Index of the test question to plot (0-indexed within test set).
        """
        print(f"Plotting the difference between the true and synthetic answer distributions for the {j}th test question...")
        Q_naive_j = self.compute_naive_baseline_test()[j]
        if self.theta is None:
            print("No trained parameters available. Comparing the true distribution and the naive baseline distribution...")  
        else:
            print("Found trained parameters. Comparing the true distribution, the trained distribution, and the naive baseline distribution...")
            Q_trained_j = self.compute_Q_distributions_test(self.theta)[j]
        P_j = self.P_test[j]
        
        true_avg_answer = P_j.dot(self.possible_answers)
        naive_avg_answer = Q_naive_j.dot(self.possible_answers)
        if self.theta is not None:
            trained_avg_answer = Q_trained_j.dot(self.possible_answers)
        
        print(f"True average answer: {true_avg_answer:.4f}")
        print(f"Naive average answer: {naive_avg_answer:.4f}")
        if self.theta is not None:
            print(f"Trained average answer: {trained_avg_answer:.4f}")
        print("\n\n")
        
        positions = np.arange(len(self.possible_answers))
        _, ax = plt.subplots(figsize=(9, 4))
        if self.theta is None:
            width = 0.42
            ax.bar(positions - width / 2, P_j, width=width, label='True answers')
            ax.bar(positions + width / 2, Q_naive_j, width=width, label='Naive answers')
        else:
            width = 0.25
            ax.bar(positions - width, P_j, width=width, label='True answers')
            ax.bar(positions, Q_trained_j, width=width, label='Trained answers')
            ax.bar(positions + width, Q_naive_j, width=width, label='Naive answers')
        ax.set_xticks(positions)
        ax.set_xticklabels([str(r) for r in self.possible_answers])
        ax.set_xlabel('Rating', fontsize=14)
        ax.set_ylabel('Probability', fontsize=14)
        ax.legend(fontsize=12)
        plt.tight_layout()
        
        # Save figure if results directory is available
        if self.results_figures_dir is not None:
            fig_path = os.path.join(self.results_figures_dir, f'distribution_difference_question_{j}.pdf')
            plt.savefig(fig_path, format='pdf', bbox_inches='tight')
            print(f"Saved distribution difference plot to {fig_path}")
        
        plt.show()
    
    def run_full_workflow(
        self,
        plot_convergence: bool = True,
        plot_variance_ratio: bool = True,
        plot_dist_difference: bool = True,
        test_question_idx: int = 0,
        get_top_weights: bool = True,
        top_num: int = 10,
        plot_divergence_comparison: bool = True
    ) -> Dict[str, Any]:
        """Run the complete distribution calibration workflow.
        
        This method streamlines the typical workflow on an already-initialized object:
        1. Run optimization
        2. Plot convergence (optional) - saved to results_figures_dir/convergence.pdf
        3. Evaluate test performance for all divergence types - saved to results_tables_dir/test_performance_all_div_types.csv
        4. Plot variance ratio (optional) - saved to results_figures_dir/variance_ratio.pdf
        5. Plot distribution difference for a test question (optional) - saved to results_figures_dir/distribution_difference_question_{j}.pdf
        6. Get top personas and dummies (optional) - prints persona descriptions if personas and pids are provided
        7. Evaluate test performance for the training divergence type
        8. Plot divergence comparison (scatter and histogram, optional) - saved to results_figures_dir/divergence_comparison_*.pdf
        
        All figures and tables are saved automatically if dataset_name was provided during initialization.
        
        Args:
            plot_convergence: Whether to plot convergence history. Default: True.
            plot_variance_ratio: Whether to plot variance ratios. Default: True.
            plot_dist_difference: Whether to plot distribution difference for a test question. Default: True.
            test_question_idx: Index of test question to plot distribution difference for (0-indexed within test set). Default: 0.
            get_top_weights: Whether to get top personas and dummies. Default: True.
            top_num: Number of top weights to return for each type. Default: 10.
            plot_divergence_comparison: Whether to plot scatter and histogram comparing optimized vs baseline divergences. Default: True.
            
        Returns:
            Dictionary containing:
                - theta: Optimized parameter vector of shape (n+K,)
                - test_performance_all: Dictionary mapping divergence types to performance dictionaries
                - test_performance_df: DataFrame summarizing test performance across all divergence types
                - top_personas: Dictionary mapping persona indices to weights
                - top_dummies: Dictionary mapping dummy indices to weights
                - test_performance: Dictionary containing test performance for the training divergence type
        """
        # Optimize
        theta = self.optimize()
        
        # Plot convergence
        if plot_convergence:
            self.plot_convergence()
        
        # Evaluate test performance for all divergence types
        test_performance_all, test_performance_df = self.evaluate_test_performance_all_div_types(
            verbose=False,
            report=True
        )
        
        # Save test performance DataFrame if results directory is available
        if self.results_tables_dir is not None and test_performance_df is not None and not test_performance_df.empty:
            table_path = os.path.join(self.results_tables_dir, 'test_performance_all_div_types.csv')
            test_performance_df.to_csv(table_path)
            print(f"Saved test performance table to {table_path}")
        
        # Plot variance ratio
        if plot_variance_ratio:
            self.plot_variance_ratio()
        
        # Plot distribution difference for a test question
        if plot_dist_difference:
            self.plot_dist_difference(test_question_idx)
        
        # Get top personas and dummies
        top_personas = {}
        top_dummies = {}
        if get_top_weights:
            top_personas, top_dummies = self.get_top_personas_and_dummies(top_num=top_num)
            # Print top personas
            print("Top personas:\n")
            for rank, (ind, weight) in enumerate(top_personas.items()):
                if ind < len(self.pids):
                    pid_ind = self.pids[ind]
                    if pid_ind in self.personas:
                        print(f"{'='*100}")
                        print(f"{'='*50}")
                        print(f"Top {rank+1} persona pid: {pid_ind} with weight {weight:.5f}")
                        print(f"{'='*50}")
                        print(f"{self.personas[pid_ind]}\n")
                        print(f"{'='*100}")
                    else:
                        print(f"Top {rank+1} persona index: {ind} (pid: {pid_ind}) with weight {weight:.5f} (persona description not found)")
                else:
                    print(f"Top {rank+1} persona index: {ind} with weight {weight:.5f} (pid not found)")
            # Print top dummies
            print("\nTop dummies:\n")
            for rank, (ind, weight) in enumerate(top_dummies.items()):
                print(f"Top {rank+1} dummy index: {ind} with weight {weight:.5f}")
        
        # Evaluate test performance for the training divergence type
        test_performance = self.evaluate_test_performance(verbose=False)
        
        # Plot divergence comparison
        if plot_divergence_comparison:
            optimized_divergences = test_performance['optimized']['divergences']
            baseline_divergences = test_performance['baseline']['divergences']
            
            # Scatter plot
            plt.figure(figsize=(8, 6))
            plt.scatter(baseline_divergences, optimized_divergences, alpha=0.6)
            max_val = max(np.max(baseline_divergences), np.max(optimized_divergences))
            plt.plot([0, max_val], [0, max_val], '--', color='red', label='y=x')
            plt.xlabel('Baseline Divergences', fontsize=14)
            plt.ylabel('Optimized Divergences', fontsize=14)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            # Save scatter plot if results directory is available
            if self.results_figures_dir is not None:
                fig_path = os.path.join(self.results_figures_dir, 'divergence_comparison_scatter.pdf')
                plt.savefig(fig_path, format='pdf', bbox_inches='tight')
                print(f"Saved divergence comparison scatter plot to {fig_path}")
            
            plt.show()
            
            # Histogram
            plt.figure(figsize=(8, 6))
            plt.hist(optimized_divergences, bins=20, edgecolor='black', alpha=0.5, label='Optimized Divergences')
            plt.hist(baseline_divergences, bins=20, edgecolor='black', alpha=0.5, label='Baseline Divergences')
            plt.xlabel('Divergences', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            # Save histogram if results directory is available
            if self.results_figures_dir is not None:
                fig_path = os.path.join(self.results_figures_dir, 'divergence_comparison_histogram.pdf')
                plt.savefig(fig_path, format='pdf', bbox_inches='tight')
                print(f"Saved divergence comparison histogram to {fig_path}")
            
            plt.show()
        
        return {
            'theta': theta,
            'test_performance_all': test_performance_all,
            'test_performance_df': test_performance_df,
            'top_personas': top_personas,
            'top_dummies': top_dummies,
            'test_performance': test_performance
        }


class DistributionCalibrationBatch:
    """
    Batch comparison of DistributionCalibration across different divergence measures
    and fitting modes (fit both, fit persona only, fit dummy only).
    """
    
    def __init__(
        self,
        P: np.ndarray,
        Y_hat: np.ndarray,
        dataset_name: str,
        personas: Dict[int, str],
        pids: List[int],
        possible_answers: np.ndarray,
        divergences: Optional[List[str]] = None,
        method: str = 'mirror_descent',
        reg_w: float = 1e-6,
        reg_v: float = 1e-6,
        reg_mse: float = 1e-6,
        weight_tol: Optional[float] = None,
        max_iter: int = 1000,
        tol: float = 1e-4,
        learning_rate: float = 1e-2,
        train_test_ratio: float = 0.8,
        random_state: int = 42,
        adaptive_lr: bool = True,
        max_grad_norm: float = 10.0,
    ) -> None:
        """Initialize the batch comparison.
        
        Args:
            P: Empirical distributions for m questions, each with K possible answers, shape (m, K).
            Y_hat: Digital twin answers for m questions and n twins, shape (m, n).
            dataset_name: Name of the dataset. Used for saving the results. If None, no results will be saved.
            personas: Dictionary mapping persona IDs (pids) to persona descriptions/names.
            pids: List of persona IDs corresponding to each digital twin (length n).
            possible_answers: Possible answers for the questions, shape (K,).
            divergences: List of divergence measures to test. If None, uses all admissible divergences. Default: None.
            method: Optimization method ('mirror_descent' or 'projected_gradient'). Default: 'mirror_descent'.
            reg_w: L2 regularization on digital twin weights. Default: 1e-6.
            reg_v: L2 regularization on base distribution weights. Default: 1e-6.
            reg_mse: Mean squared error regularization. Default: 1e-6.
            weight_tol: Threshold for pruning small weights. Default: None.
            max_iter: Maximum number of iterations. Default: 1000.
            tol: Convergence tolerance. Default: 1e-4.
            learning_rate: Learning rate for optimization. Default: 1e-2.
            train_test_ratio: Ratio of questions for training. Default: 0.8.
            random_state: Random seed for train-test split. Default: 42.
            adaptive_lr: Whether to use adaptive learning rate. Default: True.
            max_grad_norm: Maximum gradient norm for clipping. Default: 10.0.
        """
        self.P = P
        self.Y_hat = Y_hat
        self.dataset_name = dataset_name
        self.personas = personas
        self.pids = pids
        self.possible_answers = possible_answers
        self.method = method
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.reg_mse = reg_mse
        self.weight_tol = weight_tol
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.train_test_ratio = train_test_ratio
        self.random_state = random_state
        self.adaptive_lr = adaptive_lr
        self.max_grad_norm = max_grad_norm
        
        # Set divergences to test
        if divergences is None:
            self.divergences = ['tv', 'chi2', 'kl', 'hellinger', 'ks', 'l1', 'l2']
        else:
            self.divergences = divergences
        
        # Store results directory
        if dataset_name is not None:
            self.results_dir = os.path.join(DISTRIBUTION_CALIBRATION_BATCH_DIR, dataset_name)
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            self.results_figures_dir = os.path.join(self.results_dir, 'figures')
            if not os.path.exists(self.results_figures_dir):
                os.makedirs(self.results_figures_dir)
            self.results_tables_dir = os.path.join(self.results_dir, 'tables')
            if not os.path.exists(self.results_tables_dir):
                os.makedirs(self.results_tables_dir)
        else:
            self.results_dir = None
            self.results_figures_dir = None
            self.results_tables_dir = None
        
        # Store results
        self.results_fit_both = []
        self.results_fit_persona_only = []
        self.results_fit_dummy_only = []
        self.optimizers_fit_both = {}
        self.optimizers_fit_persona_only = {}
        self.optimizers_fit_dummy_only = {}
    
    def run_all_experiments(self) -> Dict[str, pd.DataFrame]:
        """Run all experiments across divergence measures and fitting modes.
        
        Runs optimization experiments for all specified divergence measures across three fitting modes:
        - fit_both: Optimize both persona and dummy weights
        - fit_persona_only: Optimize only persona weights
        - fit_dummy_only: Optimize only dummy weights
        
        All result DataFrames are automatically saved to results_tables_dir/ if dataset_name was
        provided during initialization:
        - results_fit_both.csv
        - results_fit_persona_only.csv
        - results_fit_dummy_only.csv
        - results_combined.csv
        
        Returns:
            Dictionary containing:
                - 'fit_both': DataFrame with results for fit_both mode
                - 'fit_persona_only': DataFrame with results for fit_persona_only mode
                - 'fit_dummy_only': DataFrame with results for fit_dummy_only mode
                - 'combined': DataFrame with combined results showing [fit_both, fit_persona_only, fit_dummy_only] for each entry
        """
        # Run experiments for fit_both
        print(f"{'='*80}")
        print("Fitting both persona and dummy weights...")
        print(f"{'='*80}")
        self._run_experiments_for_mode(fit_persona_only=False, fit_dummy_only=False)
        
        # Run experiments for fit_persona_only
        print(f"{'='*80}")
        print("Fitting persona only...")
        print(f"{'='*80}")
        self._run_experiments_for_mode(fit_persona_only=True, fit_dummy_only=False)

        # Run experiments for fit_dummy_only
        print(f"{'='*80}")
        print("Fitting dummy only...")
        print(f"{'='*80}")
        self._run_experiments_for_mode(fit_persona_only=False, fit_dummy_only=True)
        
        # Create DataFrames
        results_df_fit_both = self._create_results_dataframe(self.results_fit_both)
        results_df_fit_persona_only = self._create_results_dataframe(self.results_fit_persona_only)
        results_df_fit_dummy_only = self._create_results_dataframe(self.results_fit_dummy_only)
        
        # Create combined DataFrame
        results_df_combined = self._create_combined_dataframe(
            results_df_fit_both, results_df_fit_persona_only, results_df_fit_dummy_only
        )

        # Create results dictionary for print_summary
        results_dict = {
            'fit_both': results_df_fit_both,
            'fit_persona_only': results_df_fit_persona_only,
            'fit_dummy_only': results_df_fit_dummy_only,
            'combined': results_df_combined
        }

        # Print summary
        self.print_summary(results_dict)
        print("\n")
        
        # Save DataFrames if results directory is available
        if self.results_tables_dir is not None:
            # Save individual DataFrames
            fit_both_path = os.path.join(self.results_tables_dir, 'results_fit_both.csv')
            results_df_fit_both.to_csv(fit_both_path)
            print(f"Saved fit_both results to {fit_both_path}")
            
            fit_persona_path = os.path.join(self.results_tables_dir, 'results_fit_persona_only.csv')
            results_df_fit_persona_only.to_csv(fit_persona_path)
            print(f"Saved fit_persona_only results to {fit_persona_path}")
            
            fit_dummy_path = os.path.join(self.results_tables_dir, 'results_fit_dummy_only.csv')
            results_df_fit_dummy_only.to_csv(fit_dummy_path)
            print(f"Saved fit_dummy_only results to {fit_dummy_path}")
            
            # Save combined DataFrame
            combined_path = os.path.join(self.results_tables_dir, 'results_combined.csv')
            # For the combined DataFrame, we need to handle the list values in cells
            # Convert lists to strings for CSV saving
            combined_df_str = results_df_combined.copy()
            for idx in combined_df_str.index:
                for col in combined_df_str.columns:
                    if isinstance(combined_df_str.at[idx, col], list):
                        combined_df_str.at[idx, col] = str(combined_df_str.at[idx, col])
            combined_df_str.to_csv(combined_path)
            print(f"Saved combined results to {combined_path}")
        
        return {
            'fit_both': results_df_fit_both,
            'fit_persona_only': results_df_fit_persona_only,
            'fit_dummy_only': results_df_fit_dummy_only,
            'combined': results_df_combined
        }
    
    def _run_experiments_for_mode(
        self,
        fit_persona_only: bool,
        fit_dummy_only: bool
    ) -> None:
        """Run experiments for a specific fitting mode.
        
        Args:
            fit_persona_only: Whether to only fit persona weights.
            fit_dummy_only: Whether to only fit dummy weights.
        """
        results = []
        optimizers = {}
        
        # Use tqdm with leave=False to remove the bar when done
        # and miniters=1, mininterval=0 to update immediately
        for div in tqdm(self.divergences, miniters=1, mininterval=0, leave=False):
            # Suppress print statements and tqdm progress bars (tqdm writes to stderr)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                # Create optimizer
                # dataset_name is None because no results will be saved for individual experiments
                opt = DistributionCalibration(
                    P=self.P,
                    Y_hat=self.Y_hat,
                    dataset_name=None,
                    personas=self.personas,
                    pids=self.pids,
                    possible_answers=self.possible_answers,
                    divergence=div,
                    method=self.method,
                    reg_w=self.reg_w,
                    reg_v=self.reg_v,
                    reg_mse=self.reg_mse,
                    fit_persona_only=fit_persona_only,
                    fit_dummy_only=fit_dummy_only,
                    weight_tol=self.weight_tol,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    learning_rate=self.learning_rate,
                    train_test_ratio=self.train_test_ratio,
                    random_state=self.random_state,
                    adaptive_lr=self.adaptive_lr,
                    max_grad_norm=self.max_grad_norm,
                )
                
                # Optimize
                theta = opt.optimize()
                
                # Evaluate test performance for all divergence types
                test_perf_all_div_types, _ = opt.evaluate_test_performance_all_div_types(
                    verbose=False,
                    report=False
                )
                
                # Store optimizer
                optimizers[div] = opt
                
                # Store results: [avg_divergence, se_divergence for each div_type] + [mse]
                # MSE is the same for all evaluation divergence types, so we use 'kl' as reference
                row = []
                for div_type in self.divergences:
                    row.append({
                        'avg': test_perf_all_div_types[div_type]['optimized']['avg_divergence'],
                        'se': test_perf_all_div_types[div_type]['optimized']['se_divergence']
                    })
                row.append({'avg': test_perf_all_div_types['kl']['optimized']['mse'], 'se': 0.0})  # MSE doesn't have SE
                results.append(row)
        
        # Add baseline results (using the first optimizer's baseline, as baseline is the same for all)
        baseline_opt = optimizers[self.divergences[0]]
        baseline_test_perf, _ = baseline_opt.evaluate_test_performance_all_div_types(
            verbose=False,
            report=False
        )
        baseline_row = []
        for div_type in self.divergences:
            baseline_row.append({
                'avg': baseline_test_perf[div_type]['baseline']['avg_divergence'],
                'se': baseline_test_perf[div_type]['baseline']['se_divergence']
            })
        baseline_row.append({'avg': baseline_test_perf['kl']['baseline']['mse'], 'se': 0.0})  # MSE doesn't have SE
        results.append(baseline_row)
        
        # Store results
        if fit_persona_only:
            self.results_fit_persona_only = results
            self.optimizers_fit_persona_only = optimizers
        elif fit_dummy_only:
            self.results_fit_dummy_only = results
            self.optimizers_fit_dummy_only = optimizers
        else:
            self.results_fit_both = results
            self.optimizers_fit_both = optimizers
    
    def _create_results_dataframe(self, results: List[List[Dict[str, float]]]) -> pd.DataFrame:
        """Create a DataFrame from results list with mean ± standard error format.
        
        Args:
            results: List of result rows, where each row contains dictionaries with 'avg' and 'se' keys.
            
        Returns:
            DataFrame with divergences as columns and (divergences + 'baseline') as index.
            Each entry is formatted as "mean ± se".
        """
        columns = self.divergences + ['mse']
        index = self.divergences + ['baseline']
        df_data = []
        for row_idx, row in enumerate(results):
            if not isinstance(row, list):
                raise TypeError(f"Row {row_idx} is not a list, got {type(row)}: {row}")
            formatted_row = []
            for item_idx, item in enumerate(row):
                if not isinstance(item, dict):
                    raise TypeError(
                        f"Row {row_idx}, item {item_idx} is not a dict, got {type(item)}: {item}. "
                        f"Row structure: {row}"
                    )
                if 'avg' not in item or 'se' not in item:
                    raise KeyError(
                        f"Row {row_idx}, item {item_idx} missing 'avg' or 'se' keys. "
                        f"Item keys: {list(item.keys())}, item: {item}"
                    )
                if item['se'] == 0.0:  # MSE case
                    formatted_row.append(f"{item['avg']:.3f}")
                else:
                    formatted_row.append(f"{item['avg']:.3f} ± {item['se']:.3f}")
            df_data.append(formatted_row)
        return pd.DataFrame(df_data, columns=columns, index=index)
    
    def _create_combined_dataframe(
        self,
        df_fit_both: pd.DataFrame,
        df_fit_persona_only: pd.DataFrame,
        df_fit_dummy_only: pd.DataFrame
    ) -> pd.DataFrame:
        """Create a combined DataFrame showing [fit_both, fit_persona_only, fit_dummy_only] for each entry.
        
        Args:
            df_fit_both: DataFrame for fit_both mode (with mean ± standard error format).
            df_fit_persona_only: DataFrame for fit_persona_only mode (with mean ± standard error format).
            df_fit_dummy_only: DataFrame for fit_dummy_only mode (with mean ± standard error format).
            
        Returns:
            Combined DataFrame where each entry is a list [fit_both, fit_persona_only, fit_dummy_only].
        """
        columns = self.divergences + ['mse']
        index = self.divergences + ['baseline']
        results_df_combined = pd.DataFrame(None, columns=columns, index=index)
        
        for idx in index:
            for col in columns:
                # Extract numeric values from "mean ± se" format or "mean" format for rounding
                fit_both_val = df_fit_both.at[idx, col]
                fit_persona_val = df_fit_persona_only.at[idx, col]
                fit_dummy_val = df_fit_dummy_only.at[idx, col]
                
                # Store as list of formatted strings
                results_df_combined.at[idx, col] = [
                    fit_both_val,
                    fit_persona_val,
                    fit_dummy_val
                ]
        
        return results_df_combined
    
    def print_summary(self, results_dict: Dict[str, pd.DataFrame]) -> None:
        """Print a summary of the results.
        
        Args:
            results_dict: Dictionary returned by run_all_experiments().
        """
        print(f"\n{'='*80}")
        print("Optimization Results Summary")
        print("=" * 80)
        print("Trained with row metrics and tested with column metrics")
        print("Each entry in the table is [fit_both, fit_persona_only, fit_dummy_only] (mean ± standard error)\n")
        print(results_dict['combined'].to_string())

