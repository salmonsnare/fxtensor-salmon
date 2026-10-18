Overview
========

An :class:`~fxtensor_salmon.FXTensor` is a morphism between finite sets. The
array axes are split into a **domain** (inputs) and a **codomain** (outputs).
That pair of dimension lists is the **profile**. Each factor may carry a list
of string labels; unlabeled factors store only their lengths.

In a Markov category the objects are the finite sets and the morphisms are
Markov kernels. On finite sets a kernel is a nonnegative array whose
codomain slice sums to 1 for every domain point. A **state** is a kernel out
of the one-point set, written here as an empty domain: a distribution, or a
joint distribution when the codomain has several factors.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Piece
     - In FXTensor
   * - Object
     - One profile factor, for example ``['Sunny', 'Rainy']`` or the length ``2``
   * - Morphism
     - An ``FXTensor`` whose ``data`` has shape ``domain + codomain``
   * - State ``I → X``
     - Empty domain, ``profile`` ``[[], ...]``
   * - Sequential composition
     - :meth:`~fxtensor_salmon.FXTensor.composition`
   * - Monoidal product
     - :meth:`~fxtensor_salmon.FXTensor.tensor_product`
   * - Copy and discard
     - :meth:`~fxtensor_salmon.FXTensor.copy_tensor` and :meth:`~fxtensor_salmon.FXTensor.exclamation`

String diagrams in this documentation are read **top to bottom** (inputs at
the top, outputs at the bottom). A box is a morphism, a triangle is a state,
a dot that branches downward is copy, and a wire that ends in a dot is
discard. The figures follow the drawing conventions of Fritz,
arXiv:1908.07021, and are drawn separately. TikZ sources live in
``docs/diagrams/`` in the repository.

The test module ``tests/test_axioms.py`` checks the finite-tensor forms of
the monoidal and comonoid equations (associativity, interchange, copy laws,
and naturality of discard) on concrete arrays.

References
----------

- `K. Cho and B. Jacobs, Disintegration and Bayesian Inversion via String Diagrams (arXiv:1709.00322) <https://arxiv.org/abs/1709.00322>`_
- `T. Fritz, A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics (arXiv:1908.07021) <https://arxiv.org/abs/1908.07021>`_
- `T. Fritz, T. Gonda, A. Lorenzin, P. Perrone, D. Stein, Absolute continuity, supports and idempotent splitting in categorical probability (arXiv:2308.00651) <https://arxiv.org/abs/2308.00651>`_
- `マルコフ圏 A First Look <https://m-hiyama.hatenablog.com/entry/2020/06/09/154044>`_
- `マルコフ圏におけるテンソル計算の手順とコツ <https://m-hiyama.hatenablog.com/entry/2021/04/05/153325>`_
