
function serializeObjPublicProps(obj) {
    const json = {};
    for (let prop in obj) {
        let value, conversion;
        if (obj[prop].arraySync) {
            value = obj[prop].arraySync();
            conversion = tensorConversion;
        } else {
            value = obj[prop];
        }
        json[prop] = { value, conversion };
    }
    return json;
}

function deserializeFromPublicProps(cls, json) {
    const self = new cls(json.n_components);
    for (let prop in json) {
        let { value, conversion } = json[prop];
        if (conversion) value = conversions[conversion](value);
        self[prop] = value;
    }
    return self;
}

function slice(start, end) {
    return Array.apply(null, { length: end - start }).map(Number.call, function (n) { return n + start; });
}

function* gen_batches(n, batch_size, min_batch_size = 0) {
    /* Generator to create slices containing batch_size elements, from 0 to n.

    The last slice contains more than batch_size elements when batch_size
    does not divide n.

    Parameters
    ----------
    n : int
    batch_size : int
        Number of element in each batch
    min_batch_size : int, default=0
        Minimum batch size to produce.

    Yields
    ------
    slice of batch_size elements

    Examples
    --------
    >>> list(gen_batches(7, 3))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(6, 3))
    [slice(0, 3, None), slice(3, 6, None)]
    >>> list(gen_batches(2, 3))
    [slice(0, 2, None)]
    >>> list(gen_batches(7, 3, min_batch_size=0))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(7, 3, min_batch_size=2))
    [slice(0, 3, None), slice(3, 7, None)]
     */
    let start = 0;
    for (let end = start + batch_size; end + min_batch_size <= n; end += batch_size) {
        yield slice(start, end);
        start = end;
    }
    if (n - start >= min_batch_size) {
        yield slice(start, n);
    }
}

module.exports = {
    slice,
    gen_batches,
    serializeObjPublicProps,
    deserializeFromPublicProps,
};