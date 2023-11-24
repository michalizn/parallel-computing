size_t idx(size_t x, size_t y, int width, int height, int xoff, int yoff) {
    size_t resx = x;
    if ((xoff > 0 && x < width - xoff) || (xoff < 0 && x >= (-xoff))) {
        resx += xoff;
    }
    size_t resy = y;
    if ((yoff > 0 && y < height - yoff) || (yoff < 0 && y >= (-yoff))) {
        resy += yoff;
    }
    return resy * width + resx;
}

__kernel void sobel3x3(__global const uchar *in,
                        __global short *output_x,
                        __global short *output_y,
                        __private const uint width,
                        __private const uint height) {

    size_t gid_x = get_global_id(0);
    size_t gid_y = get_global_id(1);

    if (gid_x < width && gid_y < height) {
        size_t gid = gid_y * width + gid_x;

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

__kernel void phaseAndMagnitude(__global const short *in_x,
                                      __global const short *in_y,
                                      __private const uint width,
                                      __private const uint height,
                                      __global uchar *phase_out,
                                      __global ushort *magnitude_out) {
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);

    if (gid_x < width && gid_y < height) {
        int gid = gid_y * width + gid_x;

        // Output in range -PI:PI
        float angle = atan2((float)in_y[gid], (float)in_x[gid]);

        // Shift range -1:1
        angle /= M_PI;

        // Shift range -127.5:127.5
        angle *= 127.5;

        // Shift range 0.5:255.5
        angle += (127.5 + 0.5);

        // Downcasting truncates angle to range 0:255
        phase_out[gid] = convert_uchar_sat(angle);

        magnitude_out[gid] = abs(in_x[gid]) + abs(in_y[gid]);
    }
}

__kernel void nonMaxSuppression(
    __global const ushort *magnitude,
    __global const uchar *phase,
    __private const uint width,
    __private const uint height,
    const short threshold_lower,
    const ushort threshold_upper,
    __global uchar *out) {

    size_t gid_x = get_global_id(0);
    size_t gid_y = get_global_id(1);
    size_t gid = gid_y * width + gid_x;

    uchar sobel_angle = phase[gid];

    if (sobel_angle > 127) {
        sobel_angle -= 128;
    }

    int sobel_orientation = 0;

    if (sobel_angle < 16 || sobel_angle >= (7 * 16)) {
        sobel_orientation = 2;
    } else if (sobel_angle >= 16 && sobel_angle < 16 * 3) {
        sobel_orientation = 1;
    } else if (sobel_angle >= 16 * 3 && sobel_angle < 16 * 5) {
        sobel_orientation = 0;
    } else if (sobel_angle > 16 * 5 && sobel_angle <= 16 * 7) {
        sobel_orientation = 3;
    }

    ushort sobel_magnitude = magnitude[gid];

    /* Non-maximum suppression
     * Pick out the two neighbours that are perpendicular to the
     * current edge pixel */
    ushort neighbour_max = 0;
    ushort neighbour_max2 = 0;

    switch (sobel_orientation) {
        case 0:
            neighbour_max =
                magnitude[idx(gid_x, gid_y, width, height, 0, -1)];
            neighbour_max2 =
                magnitude[idx(gid_x, gid_y, width, height, 0, 1)];
            break;
        case 1:
            neighbour_max =
                magnitude[idx(gid_x, gid_y, width, height, -1, -1)];
            neighbour_max2 =
                magnitude[idx(gid_x, gid_y, width, height, 1, 1)];
            break;
        case 2:
            neighbour_max =
                magnitude[idx(gid_x, gid_y, width, height, -1, 0)];
            neighbour_max2 =
                magnitude[idx(gid_x, gid_y, width, height, 1, 0)];
            break;
        case 3:
        default:
            neighbour_max =
                magnitude[idx(gid_x, gid_y, width, height, 1, -1)];
            neighbour_max2 =
                magnitude[idx(gid_x, gid_y, width, height, -1, 1)];
            break;
    }

    // Suppress the pixel here
    if ((sobel_magnitude < neighbour_max) ||
        (sobel_magnitude < neighbour_max2)) {
        sobel_magnitude = 0;
    }

    /* Double thresholding */
    // Marks YES pixels with 255, NO pixels with 0 and MAYBE pixels
    // with 127
    uchar t = 127;
    if (sobel_magnitude > threshold_upper) t = 255;
    if (sobel_magnitude <= threshold_lower) t = 0;
    out[gid] = t;
}

// #define WIDTH 100
// #define HEIGHT 100
// #define MAX_STACK_SIZE (WIDTH * HEIGHT)

// typedef struct {
//     size_t x;
//     size_t y;
// } coord_t;

// constant coord_t neighbour_offsets[8] = {
//     {-1, -1}, {0, -1},  {+1, -1}, {-1, 0},
//     {+1, 0},  {-1, +1}, {0, +1},  {+1, +1},
// };

// __kernel void edgeTracing(__global uchar *image,
//                           __private const uint width,
//                           __private const uint height) {
    
//     __local coord_t tracing_stack[MAX_STACK_SIZE];
//     __local coord_t *tracing_stack_pointer = tracing_stack;

//     size_t gid_x = get_global_id(0);
//     size_t gid_y = get_global_id(1);
//     size_t gid = gid_y * width + gid_x;

//     if (image[gid] == 255) {
//         coord_t yes_pixel = {gid_x, gid_y};
//         *tracing_stack_pointer = yes_pixel;
//         tracing_stack_pointer++;
//     }

//     while (tracing_stack_pointer != tracing_stack) {
//         tracing_stack_pointer--;
//         coord_t known_edge = *tracing_stack_pointer;
//         for (int k = 0; k < 8; k++) {
//             coord_t dir_offs = neighbour_offsets[k];
//             coord_t neighbour = {
//                 known_edge.x + dir_offs.x, known_edge.y + dir_offs.y};

//             if (neighbour.x < 0) neighbour.x = 0;
//             if (neighbour.x >= width) neighbour.x = width - 1;
//             if (neighbour.y < 0) neighbour.y = 0;
//             if (neighbour.y >= height) neighbour.y = height - 1;

//             // Only MAYBE neighbors are potential edges
//             size_t neighbour_gid = neighbour.y * width + neighbour.x;
//             if (image[neighbour_gid] == 127) {
//                 // Convert MAYBE to YES
//                 image[neighbour_gid] = 255;

//                 // Add the newly added pixel to stack, so changes will
//                 // propagate
//                 *tracing_stack_pointer = neighbour;
//                 tracing_stack_pointer++;
//             }
//         }
//     }

//     if (image[gid] == 127) {
//         image[gid] = 0;
//     }
// }