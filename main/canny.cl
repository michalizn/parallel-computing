size_t idx(size_t x, size_t y, int width, int height, int xoff, int yoff) {
    size_t resx = x;
    if ((xoff > 0 && x < width - xoff) || (xoff < 0 && x >= (-xoff)))
        resx += xoff;
    size_t resy = y;
    if ((yoff > 0 && y < height - yoff) || (yoff < 0 && y >= (-yoff)))
        resy += yoff;
    return resy * width + resx;
}

__kernel void sobel3x3(__global const int16 *in,
                        __global int16 *output_x,
                        __global int16 *output_y,
                        __private const int width,
                        __private const int height) {

    size_t gid_x = get_global_id(0);  // Assuming 1D global work size
    size_t gid_y = get_global_id(1);

    // Ensure that the global work items are within the image dimensions
    if (gid_x < width && gid_y < height) {
        size_t gid = gid_y * width + gid_x;

        printf("in_val: %d|gid: %zu|x: %zu|y: %zu|pos_input: %d|index: %zu\n", in[gid], gid, gid_x, gid_y, in[idx(gid_x, gid_y, width, height, -1, -1)], idx(gid_x, gid_y, width, height, -1, -1));


        output_x[gid] = (-1) * in[idx(gid_x, gid_y, width, height, -1, -1)] +
                        1 * in[idx(gid_x, gid_y, width, height, 1, -1)] +
                        (-2) * in[idx(gid_x, gid_y, width, height, -1, 0)] +
                        2 * in[idx(gid_x, gid_y, width, height, 1, 0)] +
                        (-1) * in[idx(gid_x, gid_y, width, height, -1, 1)] +
                        1 * in[idx(gid_x, gid_y, width, height, 1, 1)];

        output_y[gid] = (-1) * in[idx(gid_x, gid_y, width, height, -1, -1)] +
                        1 * in[idx(gid_x, gid_y, width, height, -1, 1)] +
                        (-2) * in[idx(gid_x, gid_y, width, height, 0, -1)] +
                        2 * in[idx(gid_x, gid_y, width, height, 0, 1)] +
                        (-1) * in[idx(gid_x, gid_y, width, height, 1, -1)] +
                        1 * in[idx(gid_x, gid_y, width, height, 1, 1)];
    }
}