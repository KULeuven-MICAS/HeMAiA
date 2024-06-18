# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
set errs 0
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000000 -len 32 -type write -data 01010413_00813423_ff010113_ffdff06f_10500073_000280e7_01f29293_0010029b_2c8000ef_30531073_01c30313_00000317_fb010113_6f020117_64018193_00000197_fcdff06f_000280e7_0002a283_fec28293_01000297_10500073_30429073_00828293_000802b7_30046073_30529073_02028293_00000297_f1402573_0202ce63_301022f3
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000000 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 0101041300813423ff010113ffdff06f10500073000280e701f292930010029b2c8000ef3053107301c3031300000317fb0101136f0201176401819300000197fcdff06f000280e70002a283fec2829301000297105000733042907300828293000802b730046073305290730202829300000297f14025730202ce63301022f3
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000080 -len 32 -type write -data 02051663_0004c503_00050493_02010413_00113c23_00913423_00813823_fe010113_00008067_01010113_0007c503_f3478793_01002797_00813403_fe078ce3_0017f793_00074783_f6070713_01002717_01010413_00813423_ff010113_00008067_01010113_f6a78423_01002797_00813403_fe078ce3_0207f793_00074783_f9470713_01002717
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000080 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 020516630004c503000504930201041300113c230091342300813823fe01011300008067010101130007c503f34787930100279700813403fe078ce30017f79300074783f6070713010027170101041300813423ff0101130000806701010113f6a784230100279700813403fe078ce30207f79300074783f947071301002717
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000100 -len 32 -type write -data 00813423_ff010113_fd5ff06f_0ff7f793_00d787b3_00170713_0006c683_00e506b3_00008067_01010113_00078513_00813403_00b6ea63_0007069b_00000793_00000713_01010413_00813423_ff010113_fc9ff06f_00148493_f4dff0ef_00008067_02010113_00813483_01013403_01813083_fe078ce3_0407f793_00074783_f1470713_01002717
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000100 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 00813423ff010113fd5ff06f0ff7f79300d787b3001707130006c68300e506b30000806701010113000785130081340300b6ea630007069b00000793000007130101041300813423ff010113fc9ff06f00148493f4dff0ef0000806702010113008134830101340301813083fe078ce30407f79300074783f147071301002717
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000180 -len 32 -type write -data 44010413_41613023_41513423_41413823_41313c23_42813823_42113c23_bc010113_fedff06f_b0002773_00008067_01010113_00813403_00f76863_00a787b3_b00027f3_01010413_00813423_ff010113_fd9ff06f_00d70023_00178793_00f50733_00074683_00f58733_00008067_01010113_00813403_00c76863_0007871b_00000793_01010413
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000180 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 4401041341613023415134234141382341313c234281382342113c23bc010113fedff06fb000277300008067010101130081340300f7686300a787b3b00027f30101041300813423ff010113fd9ff06f00d700230017879300f507330007468300f5873300008067010101130081340300c768630007871b0000079301010413
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000200 -len 32 -type write -data 01800513_0ff7f793_fff9079b_00008067_44010113_40013b03_40813a83_41013a03_41813983_42013903_42813483_43013403_43813083_e99ff0ef_36850513_00000517_e39ff0ef_00600513_05551063_00050913_e7dff0ef_00100a13_00400a93_00100993_e59ff0ef_01500513_fa1ff0ef_08050513_02faf537_00050b13_43213023_42913423
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000200 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 018005130ff7f793fff9079b000080674401011340013b0340813a8341013a034181398342013903428134834301340343813083e99ff0ef3685051300000517e39ff0ef006005130555106300050913e7dff0ef00100a1300400a9300100993e59ff0ef01500513fa1ff0ef0805051302faf53700050b134321302342913423
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000280 -len 32 -type write -data f31ff06f_e81ff0ef_01650533_02055513_02051513_0019899b_bc040593_00048613_0295053b_fff9851b_da1ff0ef_00600513_04a91663_e6dff0ef_bc040513_00048593_00050913_0004849b_df5ff0ef_0497c863_0009079b_00000913_08000493_01491463_40000493_08951063_0ff57513_fff54513_e1dff0ef_00050493_e25ff0ef_08fa6e63
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000280 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp f31ff06fe81ff0ef0165053302055513020515130019899bbc040593000486130295053bfff9851bda1ff0ef0060051304a91663e6dff0efbc04051300048593000509130004849bdf5ff0ef0497c8630009079b00000913080004930149146340000493089510630ff57513fff54513e1dff0ef00050493e25ff0ef08fa6e63
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000300 -len 32 -type write -data 01002717_c8f70623_01002717_00300793_caf70223_01002717_f8000793_ca078423_01002797_08010413_03a13023_03913423_06913423_06113c23_03813823_03713c23_05613023_05513423_05413823_05313c23_07213023_06813823_f8010113_f0dff06f_d59ff0ef_01500513_f99ff06f_00190913_00a78023_012787b3_bc040793_da9ff0ef
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000300 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 01002717c8f706230100271700300793caf7022301002717f8000793ca078423010027970801041303a13023039134230691342306113c230381382303713c2305613023055134230541382305313c230721302306813823f8010113f0dff06fd59ff0ef01500513f99ff06f0019091300a78023012787b3bc040793da9ff0ef
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000380 -len 32 -type write -data 000b8513_ce9ff0ef_000b0513_cf1ff0ef_000a8513_cf9ff0ef_000a0513_d01ff0ef_00098513_00300c93_00200493_01f91c13_258b8b93_00000b97_230b0b13_00000b17_230a8a93_00000a97_218a0a13_00000a17_21898993_00000997_c6f70823_01002717_02200793_00100913_c6f70c23_01002717_fc700793_c8f70423_01002717_c8070423
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000380 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 000b8513ce9ff0ef000b0513cf1ff0ef000a8513cf9ff0ef000a0513d01ff0ef0009851300300c930020049301f91c13258b8b9300000b97230b0b1300000b17230a8a9300000a97218a0a1300000a172189899300000997c6f70823010027170220079300100913c6f70c2301002717fc700793c8f7042301002717c8070423
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000400 -len 32 -type write -data 01f91513_00008067_08010113_02013d03_02813c83_03013c03_03813b83_04013b03_04813a83_05013a03_05813983_06013903_06813483_07013403_07813083_ca1ff0ef_2f850513_00000517_fb9514e3_06950263_fa9ff06f_00100073_f1402573_cc1ff0ef_28050513_00000517_fc0514e3_02a94063_07250863_fcf5051b_ca5ff0ef_ce1ff0ef
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000400 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 01f91513000080670801011302013d0302813c8303013c0303813b8304013b0304813a8305013a030581398306013903068134830701340307813083ca1ff0ef2f85051300000517fb9514e306950263fa9ff06f00100073f1402573cc1ff0ef2805051300000517fc0514e302a9406307250863fcf5051bca5ff0efce1ff0ef
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000480 -len 32 -type write -data 0007b783_26c78793_00000797_f8f43823_00000d13_0004849b_0007b783_27c78793_00000797_c09ff0ef_22050513_00000517_06071863_0007c703_00a00693_00000493_000c8023_00048793_09a51063_00ac8023_bfdff0ef_00d00d13_000c8493_c41ff0ef_f8840c93_22c50513_00000517_f45ff06f_c55ff0ef_12450513_00000517_d61ff0ef
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000480 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 0007b78326c7879300000797f8f4382300000d130004849b0007b78327c7879300000797c09ff0ef2205051300000517060718630007c70300a0069300000493000c80230004879309a5106300ac8023bfdff0ef00d00d13000c8493c41ff0eff8840c9322c5051300000517f45ff06fc55ff0ef1245051300000517d61ff0ef
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000500 -len 32 -type write -data 00f707b3_004cd793_001d0d13_fa040713_0007cc83_01ac07b3_b11ff0ef_00a00513_b19ff0ef_00d00513_00079a63_00f7f793_f7dff06f_009704b3_fd048493_00178793_02d484b3_f75ff06f_001c8c93_ea5ff06f_b7dff0ef_bb9ff0ef_1f850513_00000517_fe078ce3_0407f793_00074783_b0870713_01002717_0497e463_000d079b_f8f43c23
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000500 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 00f707b3004cd793001d0d13fa0407130007cc8301ac07b3b11ff0ef00a00513b19ff0ef00d0051300079a6300f7f793f7dff06f009704b3fd0484930017879302d484b3f75ff06f001c8c93ea5ff06fb7dff0efbb9ff0ef1f85051300000517fe078ce30407f79300074783b0870713010027170497e463000d079bf8f43c23
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000580 -len 32 -type write -data 756e2065_68742072_65746e45_20090a0d_00000000_00000a0d_00006d6f_72746f6f_4220796d_6163634f_206f7420_656d6f63_6c655720_09090a0d_00000000_4a325b1b_000a0d0a_0d202e64_65687369_6e696620_64616f4c_20090a0d_f61ff06f_ad5ff0ef_02000513_addff0ef_ff0cc503_01978cb3_fa040793_aedff0ef_00fcfc93_ff07c503
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000580 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 756e20656874207265746e4520090a0d0000000000000a0d00006d6f72746f6f4220796d6163634f206f7420656d6f636c65572009090a0d000000004a325b1b000a0d0a0d202e64656873696e69662064616f4c20090a0df61ff06fad5ff0ef02000513addff0efff0cc50301978cb3fa040793aedff0ef00fcfc93ff07c503
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000600 -len 32 -type write -data 206f7420_65756e69_746e6f43_202e3420_090a0d30_30303030_30303878_30206d6f_72662079_726f6d65_6d20746e_69725020_2e332009_0a0d5452_4155206d_6f726620_64616f4c_202e3220_090a0d47_41544a20_6d6f7266_2064616f_4c202e31_20090a0d_00000000_00000020_3a65646f_6d206568_74207463_656c6573_206f7420_7265626d
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000600 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 206f742065756e69746e6f43202e3420090a0d30303030303030387830206d6f72662079726f6d656d20746e697250202e3320090a0d54524155206d6f72662064616f4c202e3220090a0d4741544a206d6f72662064616f4c202e3120090a0d00000000000000203a65646f6d20656874207463656c6573206f74207265626d
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000680 -len 32 -type write -data 66207972_6f6d656d_20656854_20090a0d_00000000_0000203a_65747962_206e6920_79726f6d_656d2065_68742066_6f20657a_69732065_68742072_65746e45_20090a0d_00000000_00000000_0a0d0a0d_202e2e2e_72656767_75626564_206f7420_7265766f_646e6148_20090a0d_00000000_30303030_30303038_7830206d_6f726620_746f6f42
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000680 -len 32 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 662079726f6d656d2065685420090a0d000000000000203a65747962206e692079726f6d656d2065687420666f20657a697320656874207265746e4520090a0d00000000000000000a0d0a0d202e2e2e7265676775626564206f74207265766f646e614820090a0d0000000030303030303030387830206d6f726620746f6f42
if {$exp ne $resp} { puts Error; incr errs }
create_hw_axi_txn -cache 0 -force txn [get_hw_axis hw_axi_1] -address 00000700 -len 26 -type write -data 46454443_42413938_37363534_33323130_00000000_0000000a_0d0a0d20_2e2e2e30_30303030_30303878_30207461_20676e69_746f6f42_20090a0d_0000202e_64656873_696e6966_20746e69_72502009_0a0d0a0d_00000000_00003a73_69203030_30303030_30387830_206d6f72
run_hw_axi txn
create_hw_axi_txn -cache 0 -force wb [get_hw_axis hw_axi_1] -address 00000700 -len 26 -type read
run_hw_axi wb
set resp [get_property DATA [get_hw_axi_txns wb]]
set exp 46454443424139383736353433323130000000000000000a0d0a0d202e2e2e3030303030303038783020746120676e69746f6f4220090a0d0000202e64656873696e696620746e69725020090a0d0a0d0000000000003a73692030303030303030387830206d6f72
if {$exp ne $resp} { puts Error; incr errs }
puts "Errors: $errs"