tech=7
mm=0
nn=15
xx=6
yy=6
ss=3
bb=64
rr=500
ff=1
xcut=1
ycut=2
package_type=FO
IO_type=UCIe
_NoP_bw=16
DDR_type=HBM
_DRAM_bw=131072
_NoC_bw=32
_mac_dim=1024
_ul3=2048
total_tops=73728

start_time=$(date +%s)

echo "$tech $mm $nn $xx $yy $ss $bb $rr $ff $xcut $ycut $package_type $IO_type $_NoP_bw $DDR_type $_DRAM_bw $_NoC_bw $_mac_dim $_ul3 $total_tops" | ./build/stschedule


end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "=========================================="
echo "Total runtime: ${elapsed} seconds"
echo "=========================================="
