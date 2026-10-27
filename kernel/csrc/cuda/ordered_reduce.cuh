// The physical workers evaluate independent nodes of the *existing* NumPy-like
// reduction tree. In particular, neither warp order nor block width defines a
// floating-point parenthesis. All block participants must call this helper.
struct OrderedArrayValues {
    const double* values;
    __device__ double operator()(i64 index) const { return values[index]; }
};

__device__ bool ordered_tree_node(i64 count, int depth, int index,
                                  i64& offset, i64& size) {
    offset = 0; size = count;
    for (int level = 0; level < depth; ++level) {
        if (size <= 128) return false;
        i64 split = size / 2;
        split -= split % 8;
        if ((index >> (depth - level - 1)) & 1) {
            offset += split; size -= split;
        } else size = split;
    }
    return true;
}

// partials contains 128 shared doubles and total one shared double. The input
// may be global or shared, but must not alias partials. Calls are collective.
// Seven tree levels are necessary for the 8191-value boundary (a six-level
// eight-aligned split can leave a 135-value child).
__device__ double ordered_block_sum(const double* values, i64 count,
                                     double* partials, double* total) {
    if (!threadIdx.x) *total = -0.0;
    __syncthreads();
    if (count <= 128) {
        if (!threadIdx.x) *total = pairwise_sum(OrderedArrayValues{values}, 0, count);
        __syncthreads();
        return *total;
    }
    for (i64 first = 0; first < count; first += 8192) {
        const i64 remaining = count - first;
        const i64 chunk = remaining < 8192 ? remaining : 8192;
        for (int slot = threadIdx.x; slot < 128; slot += blockDim.x) {
            i64 offset = 0, size = chunk;
            int depth = 0;
            while (size > 128) {
                i64 split = size / 2;
                split -= split % 8;
                if ((slot >> (6 - depth)) & 1) {
                    offset += split; size -= split;
                } else size = split;
                ++depth;
            }
            const int width = 1 << (7 - depth);
            if (!(slot % width))
                partials[slot] = pairwise_block_sum<0>(
                    OrderedArrayValues{values + first}, offset, size);
        }
        __syncthreads();
        int deepest = 0;
        for (i64 maximum = chunk; maximum > 128; ++deepest) {
            i64 split = maximum / 2;
            split -= split % 8;
            maximum -= split;  // The right child is never smaller.
        }
        for (int depth = deepest - 1; depth >= 0; --depth) {
            const int nodes = 1 << depth;
            for (int node = threadIdx.x; node < nodes; node += blockDim.x) {
                i64 offset, size;
                if (ordered_tree_node(chunk, depth, node, offset, size) && size > 128) {
                    const int left = node << (7 - depth);
                    const int right = left + (1 << (6 - depth));
                    partials[left] = partials[left] + partials[right];
                }
            }
            __syncthreads();
        }
        // The external 8192-value buffers have always been folded serially.
        if (!threadIdx.x) *total += partials[0];
        __syncthreads();
    }
    return *total;
}
