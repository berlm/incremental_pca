


export function slice(start, end) {
    return Array.apply(null, { length: end - start }).map(Number.call, function (n) { return n + start; });
}

export function* gen_batches(n, batch_size, min_batch_size = 0) {
    /* Generator to create slices containing batch_size elements, from 0 to n.

    The last slice may contain less than batch_size elements, when batch_size
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
    let start = 0, end = 0;
    for (let i = 0; i < n / batch_size; i++) {
        let end = start + batch_size;
        if (end + min_batch_size > n) {
            continue;
        }
        yield slice(start, end);
    }

    start = end;
    if (start < n) {
        yield slice(start, n);
    }
}