#!/bin/bash
good_clusters_SSD=("gruss","grue","grat") #only one of these nodes is used
# exotic_SSD=("grouillle")

walltime="24:00:00"

home_dir="/group_calculus/users/jayilo"



name="speech_wsj0"
fname="enudiffuse/runs/separate_wsjqut_speech_modelling_default_6M.sh"


fname="${home_dir}/${fname}"
chmod +x "${fname}"



# full_command="oarsub -vv -l /nodes=1/gpu=2,walltime=${walltime} -p grat -q production -n ${name} ${fname}" #--stdout=\"/home/bnortier/Documents/logs/gr.%jobid%.stdout\" --stderr=\"/home/bnortier/Documents/logs/gr.%jobid%.stderr\"

full_command="oarsub -vv -l /nodes=1/gpu=2,walltime=${walltime} -p \"cluster in (${good_clusters_SSD[@]})\" -q production -n ${name} ${fname}" #--stdout=\"/home/bnortier/Documents/logs/gr.%jobid%.stdout\" --stderr=\"/home/bnortier/Documents/logs/gr.%jobid%.stderr\"

# full_command="oarsub -vv -l /nodes=1,walltime=${walltime} -p \"cluster in (${exotic_SSD[@]})\" -t exotic -n ${name} ${fname}" #--stdout=\"/home/bnortier/Documents/logs/gr.%jobid%.stdout\" --stderr=\"/home/bnortier/Documents/logs/gr.%jobid%.stderr\"

echo $full_command

read -p "Continue and launch this command? (y/n)" 

case $REPLY in 
	[Yy]* ) 
	eval $full_command
	echo "Command launched"
	;;
	"" )
	eval $full_command
	echo "Command launched"
	;;

	* ) 
	echo "Aborted"
	exit
	;;
esac