#include <stdint.h>

int Batch = 1;

int M = 1;

int K = 1;

int N = 1;

int32_t Aslstride0 = 1;

int32_t Aslstride1 = 8.0;

int32_t Atlbound0 = 1;

int32_t Atlstride0 = 64.0;

int32_t Atlbound1 = 1;

int32_t Atlstride1 = 0;

int32_t Atlbound2 = 1;

int32_t Atlstride2 = 64.0;

int32_t Atlbound3 = 1;

int32_t Atlstride3 = 0;

int32_t Atlbound4 = 1;

int32_t Atlstride4 = 0;

int32_t Atlbound5 = 1;

int32_t Atlstride5 = 0;

int32_t Bslstride0 = 1;

int32_t Bslstride1 = 8.0;

int32_t Btlbound0 = 1;

int32_t Btlstride0 = 64.0;

int32_t Btlbound1 = 1;

int32_t Btlstride1 = 64.0;

int32_t Btlbound2 = 1;

int32_t Btlstride2 = 0;

int32_t Cslstride0 = 8.0;

int32_t Cslstride1 = 64.0;

int32_t Ctlbound0 = 1;

int32_t Ctlstride0 = 256.0;

int32_t Ctlbound1 = 1;

int32_t Ctlstride1 = 256.0;

int32_t Ctlbound2 = 1;

int32_t Ctlstride2 = 0;

int32_t D32slstride0 = 8.0;

int32_t D32slstride1 = 64.0;

int32_t D32tlbound0 = 1;

int32_t D32tlstride0 = 256.0;

int32_t D32tlbound1 = 1;

int32_t D32tlstride1 = 256.0;

int32_t D32tlbound2 = 1;

int32_t D32tlstride2 = 0;

int32_t D8slstride0 = 1;

int32_t D8slstride1 = 8.0;

int32_t D8tlbound0 = 1;

int32_t D8tlstride0 = 64.0;

int32_t D8tlbound1 = 1;

int32_t D8tlstride1 = 64.0;

int32_t D8tlbound2 = 1;

int32_t D8tlstride2 = 0;

int32_t delta_local_a = 0;

int32_t delta_local_b = 64.0;

int32_t delta_local_c = 128.0;

int32_t delta_local_d32 = 384.0;

int32_t delta_local_d8 = 384.0;

int8_t subtraction_a = -26;

int8_t subtraction_b = 51;

int8_t A[64]  = {
	-36,
	-114,
	-22,
	-57,
	60,
	-108,
	-26,
	-7,
	82,
	86,
	-54,
	74,
	-41,
	-12,
	-29,
	-25,
	23,
	2,
	21,
	-76,
	-127,
	-41,
	107,
	29,
	-91,
	1,
	63,
	59,
	-108,
	32,
	75,
	-71,
	-107,
	124,
	107,
	-40,
	-80,
	90,
	-70,
	126,
	41,
	91,
	59,
	79,
	-114,
	61,
	61,
	46,
	61,
	-78,
	-21,
	-74,
	115,
	-65,
	120,
	2,
	100,
	-78,
	6,
	-108,
	-56,
	38,
	-111,
	3,
};

int8_t B[64]  = {
	-40,
	-69,
	-115,
	113,
	121,
	-120,
	-39,
	-76,
	1,
	-45,
	-37,
	-18,
	59,
	70,
	43,
	124,
	-121,
	46,
	-94,
	77,
	-48,
	35,
	-79,
	-25,
	3,
	-127,
	125,
	5,
	-75,
	-23,
	-125,
	-75,
	92,
	62,
	17,
	89,
	-85,
	33,
	73,
	61,
	99,
	-115,
	-34,
	-81,
	-114,
	71,
	77,
	86,
	123,
	120,
	61,
	-89,
	84,
	79,
	108,
	-47,
	-18,
	-76,
	-105,
	25,
	88,
	123,
	59,
	-5,
};

int32_t channel_en_C = 0;

int32_t broadcast_C = 0;

int32_t C[64]  = {
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
};

int32_t transposed_A = 0;

int32_t transposed_B = 0;

int32_t D32[64]  = {
	26513,
	11252,
	-7872,
	10704,
	-12722,
	2715,
	-3732,
	8262,
	-15721,
	-20345,
	-10901,
	-30536,
	12144,
	-21516,
	852,
	-19535,
	-42181,
	-3966,
	-27914,
	-18060,
	16301,
	22057,
	11368,
	-19802,
	-25862,
	-16686,
	-2170,
	-5076,
	9716,
	-10646,
	-4742,
	-10568,
	-72535,
	-7868,
	-8809,
	-25514,
	-917,
	-22839,
	-8130,
	-36355,
	-65748,
	-23798,
	-31166,
	-42316,
	18170,
	-16249,
	-3518,
	-35800,
	-5640,
	4777,
	-51120,
	-36446,
	-13981,
	-550,
	18585,
	3078,
	-24699,
	5307,
	-17392,
	19698,
	1738,
	27819,
	10399,
	-3756,
};

int32_t bypassSIMD = 1;

int8_t input_zp_i = 108;

int8_t output_zp_i = -88;

int8_t max_int_i = 127;

int8_t min_int_i = -128;

int8_t double_round_i = 0;

int32_t shared_bitpacked_shift0 = 2887196;

int32_t shared_bitpacked_shift1 = 386401816;

int32_t shared_multiplier0 = -451480448;

int32_t shared_multiplier1 = -1294006293;

int32_t shared_multiplier2 = -886961529;

int32_t shared_multiplier3 = -2123766313;

int32_t shared_multiplier4 = -2087011266;

int32_t shared_multiplier5 = 1354896522;

int32_t shared_multiplier6 = -1293462030;

int32_t shared_multiplier7 = 888445520;

int8_t D8[64]  = {
	128,
	128,
	168,
	167,
	127,
	128,
	127,
	127,
	127,
	128,
	168,
	168,
	128,
	128,
	127,
	128,
	127,
	127,
	169,
	168,
	128,
	127,
	128,
	128,
	127,
	128,
	168,
	168,
	128,
	128,
	127,
	128,
	127,
	127,
	168,
	168,
	127,
	128,
	127,
	128,
	127,
	128,
	169,
	168,
	128,
	128,
	128,
	128,
	127,
	128,
	170,
	168,
	127,
	128,
	128,
	127,
	127,
	128,
	168,
	167,
	128,
	127,
	128,
	128,
};

int32_t set_addr_remap_index_A = 0;

int32_t set_addr_remap_index_B = 0;

int32_t set_addr_remap_index_C = 0;

int32_t set_addr_remap_index_D32 = 0;

int32_t set_addr_remap_index_D8 = 0;
