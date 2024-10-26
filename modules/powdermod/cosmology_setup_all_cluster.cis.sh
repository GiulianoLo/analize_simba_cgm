#!/bin/bash 

#Powderday cluster setup convenience script for SLURM queue mananger
#on COSMA at the Durham.  This sets up the model
#files for a cosmological simulation where we want to model many
#galaxies at once.

#Notes of interest:

#1. This does *not* set up the parameters_master.py file: it is
#assumed that you will *very carefully* set this up yourself.

#2. This requires bash versions >= 3.0.  To check, type at the shell
#prompt: 

#> echo $BASH_VERSION

n_nodes=$1
model_dir=$2
hydro_dir=$3
model_run_name=$4
COSMOFLAG=$5
model_dir_remote=$6
hydro_dir_remote=$7
xpos=$8
ypos=$9
zpos=${10}
galaxy=${11}
snap=${12}
tcmb=${13}
index=${14}
job_flag=${15}
N=${16}
halo=${17}
radius=${18}
subset_type=${19}


# echo "processing model file for galaxy,snapshot:  $galaxy,$snap"


#clear the pyc files
rm -f *.pyc

#set up the model_**.py file
#echo "setting up the output directory in case it doesnt already exist"
# echo "snap is: $snap"
# echo "model dir is: $model_dir"
# Create model, out, and err directories if they do not exist
mkdir -p "$model_dir/out" "$model_dir/err" || {
    echo "Error creating directories" >&2
    exit 1
}


# echo "model output dir is: $model_dir_remote"
if [ ! -d "$model_dir_remote" ]; then
    mkdir $model_dir_remote
fi

# copy over parameters_master file
# cp parameters_master.py $model_dir/.

filem="$model_dir/snap${snap}_${galaxy}.py"
# echo "writing to $filem"
rm -f $filem

if [ $N -gt 1000 ]; then
    Nout=1000
else
    Nout=${N}
fi

echo "#Snapshot Parameters" >> $filem
echo "#<Parameter File Auto-Generated by setup_all_cluster.sh>" >> $filem
echo "snapshot_num = $snap" >> $filem
echo "galaxy_num = $galaxy" >>$filem
echo -e "\n" >> $filem


echo "galaxy_num_str = '{:06.0f}'.format(galaxy_num)">>$filem
echo "snapnum_str = '{:03.0f}'.format(snapshot_num)">>$filem
echo -e "\n" >>$filem

if [ $COSMOFLAG -eq 1 ]
then
    echo "hydro_dir = '$hydro_dir_remote/'">>$filem
    if [ "$subset_type" == "plist" ]; then
        echo "snapshot_name = f'subset_snap{snapnum_str}_gal'+galaxy_num_str+'.h5'" >> $filem
    elif [ "$subset_type" == "region" ]; then
        echo "snapshot_name = f'region_snap{snapnum_str}_r${radius}_gal'+galaxy_num_str+'.h5'" >> $filem
    else
        echo "Error: Invalid snapshot_type. Must be 'plist' or 'region'." >&2
        exit 1
    fi
else
    echo "hydro_dir = '$hydro_dir_remote/'">>$filem
    if [ "$subset_type" == "plist" ]; then
        echo "snapshot_name = f'subset_snap{snapnum_str}_gal'+galaxy_num_str+'.h5'" >> $filem
    elif [ "$subset_type" == "region" ]; then
        echo "snapshot_name = f'region_snap{snapnum_str}_r${radius}_gal'+galaxy_num_str+'.h5'" >> $filem
    else
        echo "Error: Invalid snapshot_type. Must be 'plist' or 'region'." >&2
        exit 1
    fi
fi

echo -e "\n" >>$filem

echo "#where the files should go" >>$filem
echo "PD_output_dir = '${model_dir_remote}/' ">>$filem
echo "Auto_TF_file = 'snap'+snapnum_str+'.logical' ">>$filem
echo "Auto_dustdens_file = 'snap'+snapnum_str+'.dustdens' ">>$filem

echo -e "\n\n" >>$filem
echo "#===============================================" >>$filem
echo "#FILE I/O" >>$filem
echo "#===============================================" >>$filem
echo "inputfile = PD_output_dir+'snap'+snapnum_str+'.galaxy'+galaxy_num_str+'.rtin'" >>$filem
echo "outputfile = PD_output_dir+'snap'+snapnum_str+'.galaxy'+galaxy_num_str+'.rtout'" >>$filem

echo -e "\n\n" >>$filem
echo "#===============================================" >>$filem
echo "#GRID POSITIONS" >>$filem
echo "#===============================================" >>$filem
echo "x_cent = ${xpos}" >>$filem
echo "y_cent = ${ypos}" >>$filem
echo "z_cent = ${zpos}" >>$filem

echo -e "\n\n" >>$filem
echo "#===============================================" >>$filem
echo "#CMB INFORMATION" >>$filem
echo "#===============================================" >>$filem
echo "TCMB = ${tcmb}" >>$filem

if [ "$job_flag" -eq 1 ]; then
    echo "writing slurm submission master script file"
    qsubfile="$model_dir/master.snap${snap}.job"
    rm -f $qsubfile
    echo $qsubfile
    
    echo "#! /bin/bash" >>$qsubfile
    echo "#SBATCH --job-name=${model_run_name}.snap${snap}" >>$qsubfile
    echo "#SBATCH --output=out/pd.master.snap${snap}.%a.%N.%j.o" >>$qsubfile
    echo "#SBATCH --error=err/pd.master.snap${snap}.%a.%N.%j.e" >>$qsubfile
    echo "#SBATCH --mail-type=ALL" >>$qsubfile
    # echo "#SBATCH --mail-user=c.lovell@herts.ac.uk" >>$qsubfile

    if [ $N -gt 1000 ]; then
        echo "#SBATCH -t 0-08:00" >>$qsubfile
    else
        echo "#SBATCH -t 1-00:00" >>$qsubfile
    fi

    echo "#SBATCH --ntasks=8">>$qsubfile
    echo "#SBATCH --cpus-per-task=8">>$qsubfile
    echo "#SBATCH --mem-per-cpu=3800">>$qsubfile
    echo "#SBATCH --array=0-${Nout}">>$qsubfile
    echo -e "\n">>$qsubfile
    echo -e "\n" >>$qsubfile

    echo "source $HOME/miniconda3/etc/profile.d/conda.sh">>$qsubfile
    echo "conda activate pd39">>$qsubfile
    echo "cd $HOME/simbanator/output/hdf5/powderday_sed_out/snap_${snap}/">>$qsubfile
    echo -e "\n">>$qsubfile
    
    
    echo "PD_FRONT_END=\"/mnt/home/glorenzon/powderday/pd_front_end.py\"">>$qsubfile
    echo -e "\n">>$qsubfile

    echo "id=\$(head -n \$((\$SLURM_ARRAY_TASK_ID+1)) ids.txt | tail -n 1)">>$qsubfile
    
    echo "python \$PD_FRONT_END . parameters_master snap${snap}_\$id > gal_\$id/snap${snap}_\$id.LOG">>$qsubfile
    echo -e "\n">>$qsubfile
    
    echo "echo \"Job done, info follows...\"">>$qsubfile
    echo "sacct -j \$SLURM_JOBID --format=JobID,JobName,Partition,Elapsed,ExitCode,MaxRSS,CPUTime,SystemCPU,ReqMem">>$qsubfile
    echo "exit">>$qsubfile
fi

#done