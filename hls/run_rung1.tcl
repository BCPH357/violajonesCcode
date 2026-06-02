# run_rung1.tcl — Vitis HLS 2023.2 C-synthesis, rung1
#
# Change (on top of rung0 hygiene):
#   - New entry point vj_evaluate_window_fixed_flat() in vj_fixed.cpp:
#     same algorithm, but three lookup tables received as independent
#     const-array parameters (no vj_cascade_fixed_t* struct indirection).
#   - vj_cascade_top() passes g_stages/g_weak_classifiers/g_features_fixed
#     directly; &g_cascade is never touched by the synthesis call graph.
#
# Run from repo root:
#   vitis_hls -f hls/run_rung1.tcl
#
open_project -reset vj_rung1_proj
set_top vj_cascade_top
add_files src/vj_fixed.cpp  -cflags "-Isrc -Ihls"
add_files hls/vj_cascade_top.cpp -cflags "-Isrc -Ihls"
open_solution -reset solution1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csynth_design
exit
