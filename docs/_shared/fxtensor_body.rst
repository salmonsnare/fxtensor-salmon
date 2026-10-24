.. Autodoc body shared by the English and Japanese API pages.

.. currentmodule:: fxtensor_salmon

.. autosummary::
   :nosignatures:

   FXTensor
   FXTensor.profile
   FXTensor.labels
   FXTensor.names
   FXTensor.rename
   FXTensor.axis
   FXTensor.get_label_index
   FXTensor.get_index_label
   FXTensor.copy
   FXTensor.composition
   FXTensor.tensor_product
   FXTensor.marginalization
   FXTensor.marginal
   FXTensor.marginalize
   FXTensor.permute
   FXTensor.observe
   FXTensor.discard_prefix
   FXTensor.first_marginalization
   FXTensor.second_marginalization
   FXTensor.partial_composition
   FXTensor.jointification
   FXTensor.conditionalization
   FXTensor.conditional
   FXTensor.is_conditionally_independent
   FXTensor.is_independent
   FXTensor.conditional_product
   FXTensor.is_exchangeable
   FXTensor.symmetrize
   FXTensor.is_kolmogorov_consistent
   FXTensor.iid
   FXTensor.mixture_of_iid
   FXTensor.bayesian_inversion
   FXTensor.support
   FXTensor.almost_surely_equal
   FXTensor.is_absolutely_continuous
   FXTensor.is_markov
   FXTensor.is_deterministic
   FXTensor.from_json
   FXTensor.from_strands
   FXTensor.to_json
   FXTensor.save_to_file
   FXTensor.load_from_file
   FXTensor.to_original_dict
   FXTensor.identity_tensor
   FXTensor.copy_tensor
   FXTensor.delta_tensor
   FXTensor.swap
   FXTensor.exclamation
   FXTensor.unit_tensor

.. autoclass:: FXTensor
   :members:
   :undoc-members:
   :inherited-members:
   :member-order: bysource

The conditional-independence methods are implemented in
:mod:`fxtensor_salmon.independence` and mixed into ``FXTensor``.

.. automodule:: fxtensor_salmon.independence
   :members:
   :show-inheritance:
   :no-index:

The exchangeability methods are implemented in
:mod:`fxtensor_salmon.exchangeability` and mixed into ``FXTensor``.

.. automodule:: fxtensor_salmon.exchangeability
   :members:
   :show-inheritance:
   :no-index:

The named-axis methods are implemented in :mod:`fxtensor_salmon.naming`
and mixed into ``FXTensor``.

.. automodule:: fxtensor_salmon.naming
   :members:
   :show-inheritance:
   :no-index:
