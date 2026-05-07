#pragma once

#include <foundation/python_types.h>

namespace features {

py::tuple dspsift(foundation::pyarray_f image, float peak_threshold,
                float edge_threshold, int target_num_features,
                bool feature_root, bool domain_size_pooling, bool estimate_affine_shape);

}
