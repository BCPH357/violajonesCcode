# run_rung0.tcl — Vitis HLS 2023.2 C-synthesis, rung0
#
# Change: only hygiene — malloc/free build helpers wrapped in
#         #ifndef __SYNTHESIS__; evaluator and top unchanged.
#
# Run from repo root:
#   vitis_hls -f hls/run_rung0.tcl
#
open_project -reset vj_rung0_proj
set_top vj_cascade_top
add_files src/vj_fixed.cpp  -cflags "-Isrc -Ihls"
add_files hls/vj_cascade_top.cpp -cflags "-Isrc -Ihls"
open_solution -reset solution1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csynth_design
exit
