create_waiver -of_objects [get_drc_violations -name hemaia_system_wrapper_drc_routed.rpx {ADEF-911#1}] -description disable_adef911
set_property SEVERITY {Warning} [get_drc_checks ADEF-911]
