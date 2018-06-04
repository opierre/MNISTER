#export TF_CPP_MIN_LOG_LEVEL=2

dmesg |grep -i pstate
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq 

sudo sh -c "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
sudo sh -c "echo performance > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor"
sudo sh -c "echo performance > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor"
sudo sh -c "echo performance > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor"
sudo sh -c "echo performance > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"
sudo sh -c "echo performance > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor"
sudo sh -c "echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor"
sudo sh -c "echo performance > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor"
#cat /proc/cpuinfo
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
sudo sh -c "echo 1 >/sys/devices/system/cpu/intel_pstate/no_turbo "
