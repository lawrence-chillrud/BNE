import numpy as np

def get_parent_node_names(family_tree):
    """Get names of non-leaf nodes of input family tree.

    Args:
        family_tree: (dict of list or None) A dictionary of list of strings to
            specify the family tree between models, if None then assume there's
            no structure (i.e. flat structure).

    Raises:
        (ValueError) If name of any leaf node did not appear in base_pred.
    """
    return np.asarray(list(family_tree.keys()))

def get_leaf_model_names(family_tree):
    """Get names of leaf nodes of input family tree.

    Args:
        family_tree: (dict of list or None) A dictionary of list of strings to
            specify the family tree between models, if None then assume there's
            no structure (i.e. flat structure).

    Raises:
        (ValueError) If name of any leaf node did not appear in base_pred.
    """
    all_node_names = np.concatenate(list(family_tree.values()))
    all_parent_names = get_parent_node_names(family_tree)

    all_leaf_names = [name for name in all_node_names
                      if name not in all_parent_names]

    return all_leaf_names
