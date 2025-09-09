// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Cyril Koenig <cykoenig@iis.ee.ethz.ch>
#pragma once

#include <stdint.h>
#include <uart.h> // for printf
#include <string.h>
/////////////////////
///// LOGGING  //////
/////////////////////

extern int libhero_log_level;

enum log_level {
  LOG_ERROR = 0,
  LOG_WARN = 1,
  LOG_INFO = 2,
  LOG_DEBUG = 3,
  LOG_TRACE = 4,
  LOG_MIN = LOG_ERROR,
  LOG_MAX = LOG_TRACE,
};

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define pr_error(fmt, ...)                                            \
  ({                                                                  \
    if (LOG_ERROR <= libhero_log_level)                               \
      printf("[ERROR %s:%s()] " fmt, __FILENAME__, __func__, ##__VA_ARGS__); \
  })
#define pr_warn(fmt, ...)                                             \
  ({                                                                  \
    if (LOG_WARN <= libhero_log_level)                                \
      printf("[WARN  %s:%s()] " fmt, __FILENAME__, __func__, ##__VA_ARGS__); \
  })
#define pr_info(fmt, ...)                                             \
  ({                                                                  \
    if (LOG_INFO <= libhero_log_level)                                \
      printf("[INFO  %s:%s()] " fmt, __FILENAME__, __func__, ##__VA_ARGS__); \
  })
#define pr_debug(fmt, ...)                                            \
  ({                                                                  \
    if (LOG_DEBUG <= libhero_log_level)                               \
      printf("[DEBUG %s:%s()] " fmt, __FILENAME__, __func__, ##__VA_ARGS__); \
  })
#define pr_trace(fmt, ...)                                            \
  ({                                                                  \
    if (LOG_TRACE <= libhero_log_level)                               \
      printf("[TRACE %s:%s()] " fmt, __FILENAME__, __func__, ##__VA_ARGS__); \
  })
