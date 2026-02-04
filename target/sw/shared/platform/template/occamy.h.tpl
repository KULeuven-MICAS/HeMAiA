// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#define N_CHIPLETS ${nr_chiplets}
#define N_CLUSTERS ${nr_clusters}
#define N_SNITCHES ${nr_cores}

#define N_CHIPLETS_WIDTH               ${clog2_nr_chiplets}
#define N_CLUSTERS_PER_CHIPLET         ${nr_clusters_per_chiplet}
#define N_CLUSTERS_PER_CHIPLET_WIDTH   ${clog2_nr_clusters_per_chiplet}
#define N_CORES_PER_CLUSTER            ${nr_cores_per_cluster}
#define N_CORES_PER_CLUSTER_WIDTH      ${clog2_nr_cores_per_cluster}