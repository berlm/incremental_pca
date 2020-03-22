import * as tf from "@tensorflow/tfjs-node";
import * as utils from "./utils";
import { SVD } from 'svd-js';

// Incremental Principal Components Analysis.
// License: BSD 3 clause

export async function incremental_mean_and_var(X, last_mean, last_variance, last_sample_count) {
    /* """Calculate mean update and a Youngs and Cramer variance update.

    last_mean and last_variance are statistics computed at the last step by the
    function. Both must be initialized to 0.0. In case no scaling is required
    last_variance can be None. The mean is always required and returned because
    necessary for the calculation of the variance. last_n_samples_seen is the
    number of samples encountered until now.

    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to use for variance update

    last_mean : array-like, shape: (n_features,)

    last_variance : array-like, shape: (n_features,)

    last_sample_count : array-like, shape (n_features,)

    Returns
    -------
    updated_mean : array, shape (n_features,)

    updated_variance : array, shape (n_features,)
        If None, only mean is computed

    updated_sample_count : array, shape (n_features,)

    Notes
    -----
    NaNs are ignored during the algorithm.

    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247

    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """ */
    // # old = stats until now
    // # new = the current increment
    // # updated = the aggregated stats
    return tf.tidy(() => {
        const last_sum = last_mean.mul(last_sample_count);
        const new_sum = X.sum(0);

        const new_sample_count = tf.isNaN(X).neg().sum(0);
        const updated_sample_count = last_sample_count.add(new_sample_count);

        const updated_mean = last_sum.add(new_sum).div(updated_sample_count);
        let updated_variance;

        if (last_variance !== undefined) {
            const new_unnormalized_variance = tf.pow(X, 2).mean(0).mul(new_sample_count);
            const last_unnormalized_variance = last_variance.mul(last_sample_count);

            // with np.errstate(divide='ignore', invalid='ignore'):
            const last_over_new_count = last_sample_count.div(new_sample_count);
            let updated_unnormalized_variance = last_unnormalized_variance
                .add(new_unnormalized_variance)
                .add(last_sum
                    .div(last_over_new_count)
                    .sub(new_sum)
                    .pow(2)
                    .mul(last_over_new_count.div(updated_sample_count))
                );

            updated_variance = tf.where(last_sample_count.equal(0), new_unnormalized_variance, updated_unnormalized_variance).div(updated_sample_count);
        }

        return { updated_mean, updated_variance, updated_sample_count };
    });
}

class IncrementalPCA {
    /* Incremental principal components analysis (IPCA).

    Linear dimensionality reduction using Singular Value Decomposition of
    centered data, keeping only the most significant singular vectors to
    project the data to a lower dimensional space.

    Depending on the size of the input data, this algorithm can be much more
    memory efficient than a PCA.

    This algorithm has constant memory complexity, on the order
    of ``batch_size``, enabling use of np.memmap files without loading the
    entire file into memory.

    The computational overhead of each SVD is
    ``O(batch_size * n_features ** 2)``, but only 2 * batch_size samples
    remain in memory at a time. There will be ``n_samples / batch_size`` SVD
    computations to get the principal components, versus 1 large SVD of
    complexity ``O(n_samples * n_features ** 2)`` for PCA.

    Read more in the :ref:`User Guide <IncrementalPCA>`.

    Parameters
    ----------
    n_components : int or None;, (default=None;)
        Number of components to keep. If ``n_components `` is ``None;``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    whiten : bool, optional
        When true (false by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.

    copy : bool, (default=true)
        If false, X will be overwritten. ``copy=false`` can be used to
        save memory but is unsafe for general use.

    batch_size : int or None;, (default=None;)
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None;``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``, to provide a
        balance between approximation accuracy and memory consumption.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Components with maximum variance.

    explained_variance_ : array, shape (n_components,)
        Variance explained by each of the selected components.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If all components are stored, the sum of explained variances is equal
        to 1.0;.

    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    mean_ : array, shape (n_features,)
        Per-feature empirical mean, aggregate over calls to ``partial_fit``.

    var_ : array, shape (n_features,)
        Per-feature empirical variance, aggregate over calls to
        ``partial_fit``.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf.

    n_components_ : int
        The estimated number of components. Relevant when
        ``n_components=None;``.

    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.decomposition import IncrementalPCA
    >>> X, _ = load_digits(return_X_y=true)
    >>> transformer = IncrementalPCA(n_components=7, batch_size=200)
    >>> # either partially fit on smaller batches of data
    >>> transformer.partial_fit(X[:100, :])
    IncrementalPCA(batch_size=200, copy=true, n_components=7, whiten=false)
    >>> # or let the fit function itself divide the data into batches
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape
    (1797, 7)

    Notes
    -----
    Implements the incremental PCA model from:
    `D. Ross, J. Lim, R. Lin, M. Yang, Incremental Learning for Robust Visual
    Tracking, International Journal of Computer Vision, Volume 77, Issue 1-3,
    pp. 125-141, May 2008.`
    See http://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf

    This model is an extension of the Sequential Karhunen-Loeve Transform from:
    `A. Levy and M. Lindenbaum, Sequential Karhunen-Loeve Basis Extraction and
    its Application to Images, IEEE Transactions on Image Processing, Volume 9,
    Number 8, pp. 1371-1374, August 2000.`
    See http://www.cs.technion.ac.il/~mic/doc/skl-ip.pdf

    We have specifically abstained from an optimization used by authors of both
    papers, a QR decomposition used in specific situations to reduce the
    algorithmic complexity of the SVD. The source for this technique is
    `Matrix Computations, Third Edition, G. Holub and C. Van Loan, Chapter 5,
    section 5.4.4, pp 252-253.`. This technique has been omitted because it is
    advantageous only when decomposing a matrix with ``n_samples`` (rows)
    >= 5/3 * ``n_features`` (columns), and hurts the readability of the
    implemented algorithm. This would be a good opportunity for future
    optimization, if it is deemed necessary.

    References
    ----------
    D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust Visual
        Tracking, International Journal of Computer Vision, Volume 77,
        Issue 1-3, pp. 125-141, May 2008.

    G. Golub and C. Van Loan. Matrix Computations, Third Edition, Chapter 5,
        Section 5.4.4, pp. 252-253.

    See also
    --------
    PCA
    KernelPCA
    SparsePCA
    TruncatedSVD
     */

    constructor(n_components, whiten = false, copy = true, batch_size = undefined) {
        let self = this;
        self.n_components = n_components;
        self.whiten = whiten;
        self.copy = copy;
        self.batch_size = batch_size;
    }

    async fit(X) {
        /* Fit the model with X, using minibatches of size batch_size.

        Parameters
        ----------
        X : array - like, shape(n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        y : Ignored

        Returns
        -------
            self : object
            Returns the instance itself.
         */
        let self = this;
        X = tf.tensor(X);
        self.n_samples_seen_ = 0;
        self.mean_ = 0.0;
        self.var_ = 0.0;

        // X = check_array(X, copy = self.copy, dtype = [np.float64, np.float32]);
        n_samples = X.shape[0];
        n_features = X.shape[1];

        if (self.batch_size === undefined) {
            self.batch_size_ = 5 * n_features;
        } else {
            self.batch_size_ = self.batch_size;
        }

        for (let batch of gen_batches(n_samples, self.batch_size_, self.n_components || 0)) {
            await self.partial_fit(X.slice(batch));
        }
        tf.dispose(X);
        return self;
    }

    async partial_fit(X) {
        /* Incremental fit with X. All of X is processed as a single batch.
    
        Parameters
        ----------
            X : array - like, shape(n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
                check_input : bool
            Run check_array on X.
    
        y : Ignored
    
        Returns
        -------
            self : object
            Returns the instance itself.
        */
        let self = this;
        X = tf.tensor(X);
        n_samples = X.shape[0];
        n_features = X.shape[1];

        if (self.n_components === undefined) {
            if (self.components_ === undefined) {
                self.n_components_ = min(n_samples, n_features);
            }
            else {
                self.n_components_ = self.components_.shape[0];
            }
        } else if (1 <= self.n_components <= n_features) {
            throw Error("n_components=%r invalid for n_features=%d, need more rows than columns for IncrementalPCA processing" % (self.n_components, n_features))
        }
        else if (self.n_components > n_samples) {
            throw Error("n_components=%r must be less or equal to the batch number of samples %d." % (self.n_components, n_samples))
        } else {
            self.n_components_ = self.n_components;
        }

        if ((self.components_ !== undefined) && (self.components_.length != self.n_components_)) {
            throw Error("Number of input features has changed from %i to %i between calls to partial_fit! Try setting n_components to a fixed value." %
                (self.components_.length, self.n_components_));
        }

        // This is the first partial_fit
        if (self.n_samples_seen_ === undefined) {
            self.n_samples_seen_ = 0;
            self.mean_ = 0.0;
            self.var_ = 0.0;
        }

        // Update stats - they are 0; if this is the fisrt step
        let { col_mean, col_var, n_total_samples } = await incremental_mean_and_var(
            X, self.mean_, self.var_,
            tf.fill(self.n_samples_seen_, n_features)
        );
        n_total_samples = await n_total_samples.array()[0];

        // Whitening
        if (self.n_samples_seen_ === 0) {
            // # If it is the first step, simply whiten X
            X = X.sub(col_mean);
        }
        else {
            const col_batch_mean = X.mean(0);
            X = X.sub(col_batch_mean);
            // # Build matrix of combined previous basis and new data
            const mean_correction = tf.sqrt(tf.mul(self.n_samples_seen_, n_samples/n_total_samples)).mul(self.mean_.sub(col_batch_mean));
            X = tf.concat([self.singular_values_.reshape((-1, 1)) * self.components_, X, mean_correction], 1);
        }
        let { U, V, S } = SVD(await X.array(), true, true);
        // U, S, V = linalg.svd(X, full_matrices = false);
        U, V = svd_flip(U, V, false);
        explained_variance = S ** 2 / (n_total_samples - 1);
        explained_variance_ratio = S ** 2 / np.sum(col_var * n_total_samples);

        self.n_samples_seen_ = n_total_samples;
        const startIndices = utils.slice(0, self.n_components_);
        self.components_ = V[startIndices];
        self.singular_values_ = S[startIndices];
        self.mean_ = col_mean;
        self.var_ = col_var;
        // self.explained_variance_ = explained_variance[: self.n_components_];
        // self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_];
        if (self.n_components_ < n_features) {
            const endIndices = utils.slice(self.n_components_, n_features);
            self.noise_variance_ = explained_variance[endIndices].mean();
        } else {
            self.noise_variance_ = 0;
        }
        return self;
    }