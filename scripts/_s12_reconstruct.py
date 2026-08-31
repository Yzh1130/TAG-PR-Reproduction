"""Float64, text-only reconstruction for S12's explicit Gram recovery mode.

This module never imports Torch or calls an encoder.  A factor of the archived
gallery Gram matrix fixes an arbitrary coordinate system; the text mapping is
an *unconstrained least-squares* fit, not an orthogonal Procrustes fit.  Finite
rank and float32 archive errors mean equivalence must be measured, not assumed.
"""

import numpy as np


PINV_RCOND = 1e-10
AUTO_EIGENVALUE_RATIO = 1e-8


def _block_size(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("Reconstruction block size must be a positive integer")
    return int(value)


def _matrix(values, name, block_size=256):
    values = np.asanyarray(values)
    if values.ndim != 2 or min(values.shape) < 1 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{name}: expected a nonempty 2D floating matrix; got {values.shape}/{values.dtype}")
    for start in range(0, len(values), _block_size(block_size)):
        if not np.isfinite(values[start:start + block_size]).all():
            raise ValueError(f"{name}: non-finite values near row {start}")
    return values


def _rcond(value):
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("Pseudoinverse rcond must be finite and lie strictly between 0 and 1")
    return value


class ResidualAccumulator:
    """Accumulate exact-element float64 residual diagnostics one row block at a time."""

    def __init__(self):
        self.max_abs_error = 0.0
        self.error_squared = 0.0
        self.reference_squared = 0.0
        self.compared_elements = 0
        self.worst_position = None

    def update(self, actual64, reference64, row_offset=0):
        actual = np.asarray(actual64, dtype=np.float64)
        reference = np.asarray(reference64, dtype=np.float64)
        if actual.shape != reference.shape or actual.ndim != 2 or actual.size == 0:
            raise ValueError(f"Residual operands need equal nonempty 2D shapes: {actual.shape} versus {reference.shape}")
        if not np.isfinite(actual).all() or not np.isfinite(reference).all():
            raise ValueError(f"Residual operands contain non-finite values near row {row_offset}")
        difference = actual - reference
        absolute = np.abs(difference)
        maximum = float(absolute.max())
        if self.worst_position is None or maximum > self.max_abs_error:
            row, column = np.unravel_index(int(absolute.argmax()), actual.shape)
            self.max_abs_error = maximum
            self.worst_position = [int(row_offset + row), int(column)]
        self.error_squared += float(np.einsum("ij,ij->", difference, difference, dtype=np.float64))
        self.reference_squared += float(np.einsum("ij,ij->", reference, reference, dtype=np.float64))
        self.compared_elements += actual.size
        if not np.isfinite(self.error_squared) or not np.isfinite(self.reference_squared):
            raise ValueError("Residual norm overflowed float64")

    def result(self):
        if self.compared_elements == 0:
            raise ValueError("Cannot report a residual before comparing any matrix elements")
        error_norm = float(np.sqrt(self.error_squared))
        reference_norm = float(np.sqrt(self.reference_squared))
        # JSON cannot represent infinity. An undefined relative error is explicit.
        relative = error_norm / reference_norm if reference_norm else (0.0 if error_norm == 0 else None)
        return {
            "max_abs_error": self.max_abs_error,
            "relative_frobenius_residual": relative,
            "relative_frobenius_defined": relative is not None,
            "frobenius_error": error_norm,
            "reference_frobenius_norm": reference_norm,
            "compared_elements": int(self.compared_elements),
            "max_error_position": self.worst_position,
            "computation_dtype": "float64",
        }


def _svd_pinv(values, rcond=PINV_RCOND):
    """NumPy pinv's SVD formula, with one SVD also supplying rank diagnostics.

    Equivalent to ``np.linalg.pinv(values, rcond=rcond)`` for real matrices:
    singular values <= rcond * largest_singular_value are zeroed.  Keeping the
    SVD here avoids repeating the expensive test-text decomposition for rank.
    """
    rcond = _rcond(rcond)
    values = np.asarray(values, dtype=np.float64)
    left, singular, right = np.linalg.svd(values, full_matrices=False)
    cutoff = rcond * float(singular[0])
    kept = singular > cutoff
    inverse = np.zeros_like(singular)
    np.divide(1.0, singular, out=inverse, where=kept)
    pseudoinverse = (right.T * inverse) @ left.T
    if not np.isfinite(pseudoinverse).all():
        raise ValueError("Pseudoinverse produced non-finite values")
    return pseudoinverse, {
        "rcond": rcond,
        "implementation": "SVD formula equivalent to numpy.linalg.pinv(A, rcond=rcond)",
        "singular_value_cutoff": cutoff,
        "largest_singular_value": float(singular[0]),
        "smallest_retained_singular_value": float(singular[kept][-1]) if kept.any() else None,
        "retained_rank": int(kept.sum()),
        "discarded_singular_values": int((~kept).sum()),
        "input_shape": list(values.shape),
    }


def reconstruct_gallery(sim_gg, rank=1536, block_size=256):
    """Return G~ and (a)/(b) diagnostics from the immutable archived Gram matrix.

    Decompose (A+A.T)/2 in float64, clip *all* negative eigenvalues, and select
    largest components. Explicit rank is not silently reduced; zero columns are
    retained if requested rank extends beyond the positive spectrum. Residuals
    compare to the original, possibly asymmetric archive, not its symmetrization.
    """
    block_size = _block_size(block_size)
    original = _matrix(sim_gg, "sim_gg", block_size)
    if original.shape[0] != original.shape[1]:
        raise ValueError(f"sim_gg must be square; got {original.shape}")
    count = len(original)
    automatic = isinstance(rank, str) and rank == "auto"
    if not automatic:
        if isinstance(rank, str):
            try:
                rank = int(rank)
            except ValueError as exc:
                raise ValueError("gg-rank must be 'auto' or a positive integer") from exc
        if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= count:
            raise ValueError(f"gg-rank must lie in [1,{count}] or be 'auto'; got {rank!r}")
        rank = int(rank)

    symmetric = np.empty(original.shape, dtype=np.float64)
    symmetry = ResidualAccumulator()
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        forward = np.asarray(original[start:stop], dtype=np.float64)
        backward = np.asarray(original[:, start:stop].T, dtype=np.float64)
        symmetry.update(forward, backward, row_offset=start)
        symmetric[start:stop] = (forward + backward) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    del symmetric
    largest = float(eigenvalues[-1])
    if largest <= 0:
        raise ValueError(f"sim_gg has no positive eigenvalue (largest={largest}); cannot reconstruct a nonzero gallery")
    negative = eigenvalues < 0
    threshold = largest * AUTO_EIGENVALUE_RATIO
    clipped = np.maximum(eigenvalues, 0.0)
    if automatic:
        rank = int((clipped > threshold).sum())
        if rank == 0:
            raise ValueError("gg-rank auto retained no positive eigenvalues")
    selected = clipped[-rank:][::-1]
    gallery = np.ascontiguousarray(eigenvectors[:, -rank:][:, ::-1] * np.sqrt(selected))
    del eigenvectors
    residual = ResidualAccumulator()
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        residual.update(gallery[start:stop] @ gallery.T, original[start:stop], row_offset=start)
    positive_energy = float(clipped.sum(dtype=np.float64))
    squared_energy = float(clipped @ clipped)
    selected_positive = int((selected > 0).sum())
    structural_count = int((eigenvalues < -threshold).sum())
    warnings = []
    if negative.any():
        warnings.append(f"Clipped {int(negative.sum())} negative eigenvalues to zero; maximum magnitude={float(-eigenvalues[negative].min()):.9g}")
    if structural_count:
        warnings.append(f"{structural_count} negative eigenvalues exceed lambda_max*1e-8={threshold:.9g}; PSD approximation is not merely below the declared relative-noise threshold")
    if selected_positive < rank:
        warnings.append(f"Requested rank {rank} contains {rank - selected_positive} zero columns after negative eigenvalue clipping; rank was not silently reduced")
    dropped_positive = int((clipped > 0).sum()) - selected_positive
    if dropped_positive:
        warnings.append(f"Rank selection discards {dropped_positive} positive eigenvalue components; retained trace-energy ratio={float(selected.sum(dtype=np.float64) / positive_energy):.12g}; archived similarities are approximated, not assumed identical")
    return gallery, {
        "spectrum": {
            "source_shape": list(original.shape),
            "source_dtype": str(original.dtype),
            "computation_dtype": "float64",
            "symmetrization": "0.5 * (sim_gg + sim_gg.T)",
            "symmetry_max_abs_error": symmetry.result()["max_abs_error"],
            "symmetry_relative_frobenius_residual": symmetry.result()["relative_frobenius_residual"],
            "largest_eigenvalue": largest,
            "smallest_eigenvalue": float(eigenvalues[0]),
            "negative_eigenvalues_clipped": int(negative.sum()),
            "negative_eigenvalue_max_abs": float(-eigenvalues[negative].min()) if negative.any() else 0.0,
            "negative_eigenvalues_beyond_relative_threshold": structural_count,
            "positive_eigenvalues": int((clipped > 0).sum()),
            "rank_requested": "auto" if automatic else rank,
            "rank_selected": rank,
            "rank_positive_selected": selected_positive,
            "positive_components_discarded": dropped_positive,
            "auto_eigenvalue_ratio": AUTO_EIGENVALUE_RATIO,
            "auto_eigenvalue_threshold": threshold,
            "retained_energy_ratio": float(selected.sum(dtype=np.float64) / positive_energy),
            "retained_energy_definition": "sum(selected nonnegative eigenvalues) / sum(all nonnegative eigenvalues)",
            "retained_squared_energy_ratio": float((selected @ selected) / squared_energy),
            "warnings": warnings,
        },
        "residual": residual.result(),
    }


def coordinate_texts(sim_qg, gallery, block_size=256, rcond=PINV_RCOND):
    """T~ = sim_qg @ G~ @ pinv(G~.T @ G~), entirely float64."""
    block_size, rcond = _block_size(block_size), _rcond(rcond)
    scores = _matrix(sim_qg, "sim_qg", block_size)
    gallery = np.asarray(_matrix(gallery, "reconstructed gallery", block_size), dtype=np.float64)
    if scores.shape[1] != len(gallery):
        raise ValueError(f"sim_qg gallery columns {scores.shape[1]} != reconstructed gallery rows {len(gallery)}")
    gram = gallery.T @ gallery
    inverse, diagnostics = _svd_pinv(gram, rcond=rcond)
    texts = np.empty((len(scores), gallery.shape[1]), dtype=np.float64)
    residual = ResidualAccumulator()
    for start in range(0, len(scores), block_size):
        stop = min(start + block_size, len(scores))
        reference = np.asarray(scores[start:stop], dtype=np.float64)
        texts[start:stop] = (reference @ gallery) @ inverse
        residual.update(texts[start:stop] @ gallery.T, reference, row_offset=start)
    return texts, {
        "method": "T_tilde = sim_qg @ G_tilde @ pinv(G_tilde.T @ G_tilde, rcond)",
        "rcond": rcond,
        "gram_rank": diagnostics["retained_rank"],
        "pseudoinverse": diagnostics,
        "text_shape": list(texts.shape),
        "residual": residual.result(),
    }


def concatenate_text_features(features):
    """[normalized global/sqrt(2), normalized TSE/sqrt(2)] without changing norms.

    Widths may differ (the repository's TSE width need not equal CLIP's width).
    View-prompt features describe two conditional representations and cannot be
    fitted as one coordinate system by this reconstruction protocol.
    """
    if not isinstance(features, dict):
        raise ValueError("Text features must be a mapping with text_feat and text_tse_feat")
    if any("_view" in str(key) for key in features):
        raise ValueError("Gram reconstruction does not support view-prompt text features: one query has two gallery-view-conditioned representations")
    required = ("text_feat", "text_tse_feat")
    if any(key not in features for key in required):
        raise ValueError(f"Text feature mapping requires {required}; found {sorted(features)}")
    blocks = []
    for key in required:
        values = np.asarray(_matrix(features[key], key), dtype=np.float64)
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, rtol=0, atol=2e-5):
            raise ValueError(f"{key}: not L2 normalized; norm range=({norms.min():.9g}, {norms.max():.9g}); reconstruction does not silently renormalize")
        blocks.append(values / np.sqrt(np.float64(2.0)))
    if len(blocks[0]) != len(blocks[1]):
        raise ValueError(f"Global/TSE text row counts differ: {len(blocks[0])} vs {len(blocks[1])}")
    return np.concatenate(blocks, axis=1)


def fit_text_mapping(raw_texts, coordinate_texts, block_size=256, rcond=PINV_RCOND):
    """Return least-squares W = pinv(T_test_raw, rcond) @ T_tilde and (d)."""
    block_size, rcond = _block_size(block_size), _rcond(rcond)
    raw = np.asarray(_matrix(raw_texts, "raw test texts", block_size), dtype=np.float64)
    target = np.asarray(_matrix(coordinate_texts, "coordinate test texts", block_size), dtype=np.float64)
    if len(raw) != len(target):
        raise ValueError(f"Raw and coordinate test texts need equal row order/counts: {len(raw)} vs {len(target)}")
    inverse, diagnostics = _svd_pinv(raw, rcond=rcond)
    mapping = inverse @ target
    if not np.isfinite(mapping).all():
        raise ValueError("Text-coordinate least-squares mapping produced non-finite values")
    residual = ResidualAccumulator()
    for start in range(0, len(raw), block_size):
        stop = min(start + block_size, len(raw))
        residual.update(raw[start:stop] @ mapping, target[start:stop], row_offset=start)
    warnings = []
    if diagnostics["retained_rank"] < raw.shape[1]:
        warnings.append("Raw test-text features are rank-deficient at the explicit rcond; the minimum-norm fit cannot identify training-text components outside their row span")
    return mapping, {
        "method": "unconstrained least squares W = pinv(T_test_raw, rcond) @ T_tilde; not orthogonal Procrustes",
        "rcond": rcond,
        "pseudoinverse": diagnostics,
        "raw_feature_rank": diagnostics["retained_rank"],
        "raw_feature_width": int(raw.shape[1]),
        "coordinate_width": int(target.shape[1]),
        "mapping_shape": list(mapping.shape),
        "residual": residual.result(),
        "warnings": warnings,
        "calibration_dependency": "W uses archived test-query similarities and freshly extracted test text; the reconstructed training bank is not independent of this test-query calibration",
    }


def iter_mapped_scores(raw_texts, mapping, gallery, block_size=256):
    """Yield (start, stop, float64 bank-gallery scores); caller writes float32."""
    block_size = _block_size(block_size)
    raw = _matrix(raw_texts, "raw text features", block_size)
    mapping = np.asarray(_matrix(mapping, "text coordinate mapping", block_size), dtype=np.float64)
    gallery = np.asarray(_matrix(gallery, "reconstructed gallery", block_size), dtype=np.float64)
    if raw.shape[1] != mapping.shape[0] or mapping.shape[1] != gallery.shape[1]:
        raise ValueError(f"Incompatible mapped-score shapes: raw={raw.shape}, W={mapping.shape}, gallery={gallery.shape}")
    for start in range(0, len(raw), block_size):
        stop = min(start + block_size, len(raw))
        scores = (np.asarray(raw[start:stop], dtype=np.float64) @ mapping) @ gallery.T
        if not np.isfinite(scores).all():
            raise ValueError(f"Mapped similarities are non-finite near row {start}")
        yield start, stop, scores
