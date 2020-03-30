// """Tests for Incremental PCA."""
const tf = require("@tensorflow/tfjs-node");
const IPCA = require("../src/incremental_pca");
const PCA = require('pca-js');
const chai = require("chai");

describe('Incremental PCA', function () {
    it('should perform fit and transform and distinguish different vectors', async function () {
        const pca = new IPCA.IncrementalPCA(2, true, 2);
        const X = [[1, 1, 1, 1, 1], [1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [2, 2, 5, 5, 5], [1, 1, 3, 3, 3], [2, 2, 5, 5, 5]];
        await pca.fit(X);
        const X_transformed = await pca.transform(X);
        chai.expect([X_transformed.length, X_transformed[0].length]).to.deep.equal([X.length, 2]);
        chai.expect(X_transformed[2]).to.deep.equal(X_transformed[4]);
        chai.expect(X_transformed[3]).to.deep.equal(X_transformed[5]);
        chai.expect(X_transformed[0]).to.be.deep.not.equal(X_transformed[1]);
    });
});

describe('Fast PCA', function () {
    it('should perform fit and transform and distinguish different vectors', async function () {
        const pca = new IPCA.FastPCA(2, true, 2);
        const X = [[1, 1, 1, 1, 1], [1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [2, 2, 5, 5, 5], [1, 1, 3, 3, 3], [2, 2, 5, 5, 5]];
        await pca.fit(X);
        const X_transformed = await pca.transform(X);
        chai.expect([X_transformed.length, X_transformed[0].length]).to.deep.equal([X.length, 2]);
        chai.expect(X_transformed[2]).to.deep.equal(X_transformed[4]);
        chai.expect(X_transformed[3]).to.deep.equal(X_transformed[5]);
        chai.expect(X_transformed[0]).to.be.deep.not.equal(X_transformed[1]);
    });
});

function l1Norm(a, b) {
    return tf.tidy(() =>
        tf.tensor(a).sub(tf.tensor(b)).abs().sum().asScalar().arraySync()
    );
}

describe('Incremental PCA on iris dataset', function () {
    it('should be similar to PCA on Iris dataset', async function () {
        let iris = require('js-datasets-iris');
        const X = iris.data.map(x => x.slice(0, 4));
        const batch_size = Math.floor(X.length / 3);

        const vectors = PCA.getEigenVectors(X);
        const explained_pca = vectors.map((v, ix) =>
            PCA.computePercentageExplained(vectors, v)
        );

        const ipca = new IPCA.IncrementalPCA(vectors.length, true, batch_size);
        await ipca.fit(X);
        const explained_variance_ipca = ipca.explained_variance_ratio();

        chai.expect(l1Norm(explained_variance_ipca, explained_pca)).to.be.lessThan(0.05);

        const serialized = JSON.parse(JSON.stringify(ipca.serialize()));
        const ipcaNew = IPCA.IncrementalPCA.deserialize(serialized);
        const X_transformed = ipca.transform(X);
        const X_transformed_new = ipcaNew.transform(X);
        const l1_diff_tr = l1Norm(X_transformed, X_transformed_new);
        chai.expect(l1_diff_tr).to.be.lessThan(0.01);
    });
});