#include <stdint.h>

int32_t delta_config_data = 1408;
int32_t delta_config_1 = 1408;

int CONFIG_SIZE = 16 * 3 + 16 * 1 + 16 * 3 * 2;

uint32_t CONFIG[160] = {
	// k
	16843009,
	16843009,
	16843009,
	16843009,
	// 16843009, // 0x01010101
	// 16843010, // 0x01010102
	// 16843011, // 0x01010103
	// 16843012, // 0x01010104
	33686018,
	33686018,
	33686018,
	33686018,
	// 33686018, // 0x02020202
	// 33686019, // 0x02020203
	// 33686020, // 0x02020204
	// 33686021, // 0x02020205
	67372036,
	67372036,
	67372036,
	67372036,
	134744072,
	134744072,
	134744072,
	134744072,
	// b
	33686018,
	33686018,
	33686018,
	33686018,
	50529027,
	50529027,
	50529027,
	50529027,
	84215045,
	84215045,
	84215045,
	84215045,
	151587081,
	151587081,
	151587081,
	151587081,
	// p
	// [-82, -57, -51, -46, -15, -8, 3, 8, 47, 51, 52, 58, 76, 95, 106, 126]
	3536701358, 134478065, 976499503, 2120900428,
	// [-125, -123, -108, -88, -73, -23, -21, -20, 11, 29, 40, 44, 45, 64, 113, 125]
	2828305795, 3974883767, 740826379, 2104573997,
	// [-127, -125, -109, -99, -83, -56, -24, -5, 28, 34, 41, 106, 107, 111, 122, 123]
	2643690369, 4226336941, 1781080604, 2071621483,
	// [-109, -91, -85, -73, -54, -45, -15, 5, 20, 40, 44, 49, 51, 59, 90, 118] 
	3081479571, 99734474, 824977428, 1985624883,
	// const
	4, 0, 0, 0,
	3, 3, 0, 0,
	2, 2, 2, 0,
	1, 1, 1, 1,

	// PROGRAM
	         1048576,           0,           0,           0,     5244608,           0,           0,           0,  
		2689599168,           0,           0,           0,   542118466,           0,           0,           0,  
		         0,           0,           0,           0,   679489544,           0,           0,           0,  
		  12623873,           0,           0,           0,   125898755,           0,           0,           0,  

		   1048576,           0,           0,           0,     1048576,           0,           0,           0,  
		2689598976,           0,           0,           0,   542116419,           0,           0,           0,  
		         0,           0,           0,           0,           0,           0,           0,           0,  
		 138416128,           0,           0,           0,    25235459,           0,           0,           0,  

		         0,           0,           0,           0,           0,           0,           0,           0,  
		         0,           0,           0,           0,           0,           0,           0,           0,  
		         0,           0,           0,           0,           0,           0,           0,           0,  
		         0,           0,           0,           0,           0,           0,           0,           0
};

int delta_comp_data = 2688;
int delta_store_data = 5376;

int COMP_DATA_SIZE = 44;
int COMP_DATA[44] = {
	16843009, // 0x01010101
	16843010, // 0x01010102
	16843011, // 0x01010103
	16843012, // 0x01010104
	
	17039620, // 0x01040104
	16974084, // 0x01030104
	16908548, // 0x01020104
	16843012, // 0x01010104

	67371268, // 0x04040104
	67305732, // 0x04030104
	67240196, // 0x04020104
	67174660, // 0x04010104

	33686018, // 0x02020202
	33686019, // 0x02020203
	33686020, // 0x02020204
	33686021, // 0x02020205

	84017666, // 0x05020202
	67240451, // 0x04020203
	50463236, // 0x03020204
	33686021, // 0x02020205

	67108612, // 0x03FFFF04
	49212933, // 0x02EEEE05

	16843009, // 0x01010101
	16843010, // 0x01010102
	16843011, // 0x01010103
	16843012, // 0x01010104
	
	17039620, // 0x01040104
	16974084, // 0x01030104
	16908548, // 0x01020104
	16843012, // 0x01010104

	67371268, // 0x04040104
	67305732, // 0x04030104
	67240196, // 0x04020104
	67174660, // 0x04010104

	33686018, // 0x02020202
	33686019, // 0x02020203
	33686020, // 0x02020204
	33686021, // 0x02020205

	84017666, // 0x05020202
	67240451, // 0x04020203
	50463236, // 0x03020204
	33686021, // 0x02020205

	67108612, // 0x03FFFF04
	49212933, // 0x02EEEE05
};

int COMP_DATA_REF[48] = {
	-3,
	-3,
	1,
	2, 
	6,
	3939,
	3933,
	-3686,
	-3692,
	-3595,
	-3586,
	1535,
	1288,
	-4024,
	-4011,
	153,
	166,
	2318,
	2337,
	2565,
	2584,
	-261,
	-246,
	3499,
	3514,
	-2434,
	-2415,
	-103,
	-84,
	-4081,
	-4066,
	-3967,
	-3952,
	973,
	986,
	1084,
	1097,
	1198,
	1215,
	1346,
	1363,
	2645,
	2658,
	1215,
	499,
	124,
	31,
	7
};
