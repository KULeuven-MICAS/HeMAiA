// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * conv_last_layer.c
 *
 * Naive, parametrizable 2-D convolution for the last layer of a CNN
 * (e.g. ResNet8). No activation, no batch-norm, no bias — pure MAC loop.
 *
 * Weights layout : [out_c, in_c, kH, kW]
 * Tensor layout  : NHWC – [batch, height, width, channels]
 *
 * Intended for CVA6 on VSim: UART output via host.c, cycle count via mcycle().
 * Input data and golden output are provided by data.h.
 */

#include <stdint.h>
#include <stdio.h>
#include "host.c"
#include "data.h"

/* -------------------------------------------------------------------------
 * Parameters
 * ---------------------------------------------------------------------- */

typedef struct {
    int batch;      /* number of input samples (N)         */
    int in_h;       /* input height                        */
    int in_w;       /* input width                         */
    int in_c;       /* input channels                      */
    int out_c;      /* output channels (number of filters) */
    int kH;         /* kernel height                       */
    int kW;         /* kernel width                        */
    int stride;     /* stride (same for H and W)           */
    int pad;        /* zero-padding (same for H and W)     */
} ConvParams;

/* Output spatial size for one dimension */
static int out_size(int in, int k, int s, int p)
{
    return (in + 2 * p - k) / s + 1;
}

/* -------------------------------------------------------------------------
 * Convolution
 *
 * input   : [N, IH, IW, IC]   – int8, from data.h
 * weights : [OC, IC, kH, kW]  – int8, from data.h
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

            /* NHWC input index */
            int i_idx = ((n * p->in_h + ih) * p->in_w + iw) * p->in_c + ic;

            /* [OC, IC, kH, kW] weight index */
            int w_idx = ((oc * p->in_c + ic) * p->kH + kh) * p->kW + kw;

            acc += (int32_t)input[i_idx] * (int32_t)weights[w_idx];
        }

        /* NHWC output index */
        output[((n * OH + oh) * OW + ow) * p->out_c + oc] = acc;
    }
}

/* -------------------------------------------------------------------------
 * Main – ResNet8 last conv block (CIFAR-10 variant)
 *   Input : 8 x 8 x 64,  kernel 3x3, 64 filters, stride 1, same padding
 *
 * Expects data.h to define:
 *   int8_t  conv_input[]    – flattened NHWC input tensor
 *   int8_t  conv_weights[]  – flattened [OC,IC,kH,kW] weight tensor
 *   int32_t conv_output[]   – golden reference output
 *   Macros: BATCH, IN_H, IN_W, IN_C, OUT_C, KH, KW, STRIDE, PAD
 * ---------------------------------------------------------------------- */

int main(void)
{
    init_uart(32, 1);
    asm volatile("fence" : : : "memory");

    ConvParams p = {
        .batch  = BATCH,
        .in_h   = IN_H,  .in_w = IN_W,  .in_c = IN_C,
        .out_c  = OUT_C,
        .kH     = KH,    .kW   = KW,
        .stride = STRIDE,
        .pad    = PAD
    };

    const int OH = out_size(p.in_h, p.kH, p.stride, p.pad);
    const int OW = out_size(p.in_w, p.kW, p.stride, p.pad);
    const int n_output = p.batch * OH * OW * p.out_c;

    int32_t result[n_output];

    uint64_t start_cycles = mcycle();
    conv2d(conv_input, conv_weights, result, &p);
    uint64_t end_cycles = mcycle();

    /* Compare against golden output */
    uint16_t err = 0;
    for (int i = 0; i < n_output; i++) {
        if (result[i] != conv_output[i])
            err++;
    }

    /* Report via UART */
    char uart_tx_buffer[512];
    if (err) {
        sprintf(uart_tx_buffer, "Fail: %d errors \r\n", err);
        print_uart(uart_tx_buffer);
    } else {
        print_uart("Pass! \r\n");
        sprintf(uart_tx_buffer, "EXET: %llu cycles \r\n", end_cycles - start_cycles);
        print_uart(uart_tx_buffer);
    }

    return 0;
}