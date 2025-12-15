from tools.utils import compute_accuracy, compute_auc
import numpy as np
import pytest

def test_compute_accuracy():
	# perfect match between y_test and y_pred
	assert compute_accuracy(y_test=np.array([0, 1, 1, 0]),
                            y_pred=np.array([0, 1, 1, 0])) == 1.0

	# half accuracy
    assert compute_accuracy(y_test=np.array([0, 1, 1, 0]),
							y_pred=np.array([1, 1, 0, 0])) == 0.5

	# empty arrays
	acc = compute_accuracy(y_test=np.array([]), y_pred=np.array([]))
    assert np.isnan(acc) or acc == 0.0

def test_auc():
    # perfect separation
    assert compute_auc(y_test=np.array([0, 0, 1, 1]),
                       y_prob=np.array([0.1, 0.4, 0.6, 0.9])) == pytest.approx(1.0)
	
    # test if a ValueError is raised when y_test only contains one class
	# cannot compute ROC curve with a single class
    with pytest.raises(ValueError):
        compute_auc(y_test=np.array([1, 1, 1, 1]),
                    y_prob=np.array([0.9, 0.8, 0.95, 0.7]))