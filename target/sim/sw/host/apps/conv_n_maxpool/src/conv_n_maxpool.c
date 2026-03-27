// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * conv_last_layer.c
 *
 * Naive, parametrizable pipeline: 2-D convolution followed by MaxPool.
 * No activation, no batch-norm, no bias — pure MAC loop + sliding max.
 *
 * Layouts:
 *   Input / conv output / pool output : NHWC – [batch, height, width, channels]
 *   Weights                           : [out_c, in_c, kH, kW]
 *
 * Shared parameters (stride, pad) are used by both conv and maxpool.
 * MaxPool adds one extra parameter: pool_size (square window, e.g. 2).
 *
 * Intended for CVA6 on VSim: UART output via host.c, cycle count via mcycle().
 * Input data and golden outputs are provided by data.h.
 */

#include <stdint.h>
#include <stdio.h>
#include "host.c"
#include "data.h"

/* -------------------------------------------------------------------------
 * Parameters
 * ---------------------------------------------------------------------- */

typedef struct {
    int batch;      /* number of input samples (N)                  */
    int in_h;       /* input height                                  */
    int in_w;       /* input width                                   */
    int in_c;       /* input channels                                */
    int out_c;      /* output channels (number of filters)           */
    int kH;         /* conv kernel height                            */
    int kW;         /* conv kernel width                             */
    int stride;     /* stride shared by conv and maxpool (H and W)  */
    int pad;        /* zero-padding shared by conv and maxpool       */
    int pool_size;  /* maxpool window size (square, e.g. 2)          */
} ConvParams;

/* Output spatial size for one dimension */
static int out_size(int in, int k, int s, int p)
{
    return (in + 2 * p - k) / s + 1;
}

/* -------------------------------------------------------------------------
 * Convolution
 *
 * input   : [N, IH, IW, IC]   – int8
 * weights : [OC, IC, kH, kW]  – int8
 * output  : [N, OH, OW, OC]   – int32, pre-allocated by caller
 * ---------------------------------------------------------------------- */

void conv2d(const int8_t *input, const int8_t *weights, int32_t *output,
            const ConvParams *p)
{
    const int OH = out_size(p->in_h, p->kH, p->stride, p->pad);
    const int OW = out_size(p->in_w, p->kW, p->stride, p->pad);

    for (int n  = 0; n  < p->batch; n++)
    for (int oc = 0; oc < p->out_c; oc++)
    for (int oh = 0; oh < OH;       oh++)
    for (int ow = 0; ow < OW;       ow++) {

        int32_t acc = 0;

        for (int ic = 0; ic < p->in_c; ic++)
        for (int kh = 0; kh < p->kH;   kh++)
        for (int kw = 0; kw < p->kW;   kw++) {

            int ih = oh * p->stride - p->pad + kh;
            int iw = ow * p->stride - p->pad + kw;

            if (ih < 0 || ih >= p->in_h) continue;  /* zero-pad */
            if (iw < 0 || iw >= p->in_w) continue;

            int i_idx = ((n * p->in_h + ih) * p->in_w + iw) * p->in_c + ic;
            int w_idx = ((oc * p->in_c + ic) * p->kH + kh) * p->kW + kw;

            acc += (int32_t)input[i_idx] * (int32_t)weights[w_idx];
        }

        output[((n * OH + oh) * OW + ow) * p->out_c + oc] = acc;
    }
}

/* -------------------------------------------------------------------------
 * MaxPool
 *
 * Slides a (pool_size × pool_size) window over the spatial dims of the
 * conv output, taking the maximum value. Stride and padding are reused
 * from ConvParams.
 *
 * input  : [N, IH, IW, C]  – int32 (conv output)
 * output : [N, OH, OW, C]  – int32, pre-allocated by caller
 *
 * IH / IW here are the conv output spatial dims, passed explicitly.
 * ---------------------------------------------------------------------- */

void maxpool2d(const int32_t *input, int32_t *output,
               int in_h, int in_w,
               const ConvParams *p)
{
    const int C   = p->out_c;
    const int PS  = p->pool_size;
    const int OH  = out_size(in_h, PS, p->stride, p->pad);
    const int OW  = out_size(in_w, PS, p->stride, p->pad);

    for (int n  = 0; n  < p->batch; n++)
    for (int c  = 0; c  < C;        c++)
    for (int oh = 0; oh < OH;       oh++)
    for (int ow = 0; ow < OW;       ow++) {

        int32_t max_val = INT32_MIN;

        for (int ph = 0; ph < PS; ph++)
        for (int pw = 0; pw < PS; pw++) {

            int ih = oh * p->stride - p->pad + ph;
            int iw = ow * p->stride - p->pad + pw;

            /* out-of-bounds positions treated as -inf (skip) */
            if (ih < 0 || ih >= in_h) continue;
            if (iw < 0 || iw >= in_w) continue;

            int32_t val = input[((n * in_h + ih) * in_w + iw) * C + c];
            if (val > max_val) max_val = val;
        }

        output[((n * OH + oh) * OW + ow) * C + c] = max_val;
    }
}

/* -------------------------------------------------------------------------
 * Main – ResNet8 last conv block (CIFAR-10 variant)
 *   Conv  : 8×8×64 -> 8×8×64   (3×3 kernel, stride 1, same padding)
 *   Pool  : 8×8×64 -> 4×4×64   (2×2 window, stride 2, no padding)
 *
 * Expects data.h to define:
 *   int8_t  conv_input[]    – flattened NHWC input tensor
 *   int8_t  conv_weights[]  – flattened [OC,IC,kH,kW] weight tensor
 *   int32_t conv_output[]   – golden reference for conv output
 *   int32_t pool_output[]   – golden reference for maxpool output
 *   Macros: BATCH, IN_H, IN_W, IN_C, OUT_C, KH, KW, STRIDE, PAD, POOL_SIZE
 * ---------------------------------------------------------------------- */

int main(void)
{
    init_uart(32, 1);
    asm volatile("fence" : : : "memory");

    ConvParams p = {
        .batch     = BATCH,
        .in_h      = IN_H,  .in_w = IN_W,  .in_c = IN_C,
        .out_c     = OUT_C,
        .kH        = KH,    .kW   = KW,
        .stride    = STRIDE,
        .pad       = PAD,
        .pool_size = POOL_SIZE
    };

    /* Conv output spatial dims */
    const int conv_OH = out_size(p.in_h, p.kH, p.stride, p.pad);
    const int conv_OW = out_size(p.in_w, p.kW, p.stride, p.pad);
    const int n_conv  = p.batch * conv_OH * conv_OW * p.out_c;

    /* Pool output spatial dims (pool reuses stride and pad) */
    const int pool_OH = out_size(conv_OH, p.pool_size, p.stride, p.pad);
    const int pool_OW = out_size(conv_OW, p.pool_size, p.stride, p.pad);
    const int n_pool  = p.batch * pool_OH * pool_OW * p.out_c;

    int32_t conv_result[n_conv];
    int32_t pool_result[n_pool];

    /* --- timed region covers both operations --- */
    uint64_t start_cycles = mcycle();
    conv2d(conv_input, conv_weights, conv_result, &p);
    maxpool2d(conv_result, pool_result, conv_OH, conv_OW, &p);
    uint64_t end_cycles = mcycle();

    /* Compare conv output against golden */
    uint16_t conv_err = 0;
    for (int i = 0; i < n_conv; i++)
        if (conv_result[i] != conv_output[i]) conv_err++;

    /* Compare pool output against golden */
    uint16_t pool_err = 0;
    for (int i = 0; i < n_pool; i++)
        if (pool_result[i] != pool_output[i]) pool_err++;

    /* Report via UART */
    char uart_tx_buffer[512];
    if (conv_err || pool_err) {
        sprintf(uart_tx_buffer, "Fail: %d conv errors, %d pool errors \r\n",
                conv_err, pool_err);
        print_uart(uart_tx_buffer);
    } else {
        print_uart("Pass! \r\n");
        sprintf(uart_tx_buffer, "EXET: %llu cycles \r\n",
                end_cycles - start_cycles);
        print_uart(uart_tx_buffer);
    }

    return 0;
}