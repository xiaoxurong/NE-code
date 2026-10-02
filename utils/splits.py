import numpy as np
from collections import defaultdict

def subject_wise_split(
    subject_ids,
    val_ratio=0.2,
    seed=42
):
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject_ids)
    rng.shuffle(subjects)

    n_val = int(len(subjects) * val_ratio)
    print(f"Total subjects: {len(subjects)}, Train: {len(subjects) - n_val}, Val: {n_val}")
    val_subjects = subjects[:n_val]
    train_subjects = subjects[n_val:]

    train_idx = np.isin(subject_ids, train_subjects)
    val_idx = np.isin(subject_ids, val_subjects)

    return np.where(train_idx)[0], np.where(val_idx)[0]

def subject_wise_stratified_split(
    subject_ids,
    labels,
    val_ratio=0.2,
    seed=42,
    min_val_per_class=1
):
    """
    Subject-wise stratified train/val split.

    Parameters
    ----------
    subject_ids : np.ndarray, shape (N,)
        Subject identifier for each segment
    labels : np.ndarray, shape (N,)
        Segment labels (assumed constant per subject)
    val_ratio : float
        Fraction of subjects per class to use for validation
    seed : int
        Random seed
    min_val_per_class : int
        Ensure at least this many subjects per class in validation

    Returns
    -------
    train_idx : np.ndarray
        Indices for training segments
    val_idx : np.ndarray
        Indices for validation segments
    """

    rng = np.random.default_rng(seed)

    # -------------------------
    # map subject -> label
    # -------------------------
    subject_to_label = {}
    for sid in np.unique(subject_ids):
        subject_labels = labels[subject_ids == sid]
        # sanity check
        if len(np.unique(subject_labels)) != 1:
            raise ValueError(f"Subject {sid} has multiple labels!")
        subject_to_label[sid] = subject_labels[0]

    # -------------------------
    # group subjects by class
    # -------------------------
    class_to_subjects = defaultdict(list)
    for sid, lab in subject_to_label.items():
        class_to_subjects[lab].append(sid)

    val_subjects = []
    train_subjects = []

    # -------------------------
    # stratified split per class
    # -------------------------
    for lab, sids in class_to_subjects.items():
        sids = np.array(sids)
        rng.shuffle(sids)

        n_val = int(len(sids) * val_ratio)
        n_val = max(n_val, min_val_per_class)
        n_val = min(n_val, len(sids) - 1)  # ensure at least 1 train subject

        val_subjects.extend(sids[:n_val])
        train_subjects.extend(sids[n_val:])

        print(
            f"Class {lab}: total={len(sids)}, "
            f"train={len(sids)-n_val}, val={n_val}"
        )

    val_subjects = np.array(val_subjects)
    train_subjects = np.array(train_subjects)

    # -------------------------
    # map back to segment indices
    # -------------------------
    train_idx = np.where(np.isin(subject_ids, train_subjects))[0]
    val_idx = np.where(np.isin(subject_ids, val_subjects))[0]

    print(
        f"Total subjects: {len(subject_to_label)}, "
        f"Train: {len(train_subjects)}, "
        f"Val: {len(val_subjects)}"
    )

    return train_idx, val_idx