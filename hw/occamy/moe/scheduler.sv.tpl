<%page args="max_trans_atop_filter_per"/>\
  ///////////////////
  // MoE Scheduler //
  ///////////////////

  <% out_moe_scheduler = soc_narrow_xbar.__dict__["out_moe_scheduler"] \
    .atomic_adapter(context, filter=True, max_trans=max_trans_atop_filter_per,
                    name="out_moe_scheduler_noatop",
                    inst_name="i_out_moe_scheduler_atop_filter") \
  %>\

  `REG_BUS_TYPEDEF_ALL(moe_scheduler_reg_a${out_moe_scheduler.aw}_d${out_moe_scheduler.dw}, logic [${out_moe_scheduler.aw-1}:0], logic [${out_moe_scheduler.dw-1}:0], logic [${out_moe_scheduler.dw//8-1}:0])

  moe_scheduler_reg_a${out_moe_scheduler.aw}_d${out_moe_scheduler.dw}_req_t moe_scheduler_reg_req;
  moe_scheduler_reg_a${out_moe_scheduler.aw}_d${out_moe_scheduler.dw}_rsp_t moe_scheduler_reg_rsp;

  axi_to_reg #(
    .ADDR_WIDTH ( ${out_moe_scheduler.aw} ),
    .DATA_WIDTH ( ${out_moe_scheduler.dw} ),
    .ID_WIDTH   ( ${out_moe_scheduler.iw} ),
    .USER_WIDTH ( ${out_moe_scheduler.uw} ),
    .axi_req_t  ( ${out_moe_scheduler.req_type()} ),
    .axi_rsp_t  ( ${out_moe_scheduler.rsp_type()} ),
    .reg_req_t  ( moe_scheduler_reg_a${out_moe_scheduler.aw}_d${out_moe_scheduler.dw}_req_t ),
    .reg_rsp_t  ( moe_scheduler_reg_a${out_moe_scheduler.aw}_d${out_moe_scheduler.dw}_rsp_t )
  ) i_axi_to_reg_moe_scheduler (
    .clk_i,
    .rst_ni,
    .testmode_i ( 1'b0 ),
    .axi_req_i  ( ${out_moe_scheduler.req_name()} ),
    .axi_rsp_o  ( ${out_moe_scheduler.rsp_name()} ),
    .reg_req_o  ( moe_scheduler_reg_req ),
    .reg_rsp_i  ( moe_scheduler_reg_rsp )
  );

  moe_scheduler_reg_wrapper #(
    .reg_req_t ( moe_scheduler_reg_a${out_moe_scheduler.aw}_d${out_moe_scheduler.dw}_req_t ),
    .reg_rsp_t ( moe_scheduler_reg_a${out_moe_scheduler.aw}_d${out_moe_scheduler.dw}_rsp_t )
  ) i_moe_scheduler_reg_wrapper (
    .clk_i,
    .rst_ni,
    .reg_req_i ( moe_scheduler_reg_req ),
    .reg_rsp_o ( moe_scheduler_reg_rsp )
  );
