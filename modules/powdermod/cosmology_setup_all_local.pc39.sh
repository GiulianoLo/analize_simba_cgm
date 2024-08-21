#!/bin/bash

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pd_env

# Local setup convenience script for a cosmological simulation.
# This script sets up the model files for a cosmological simulation
# where we want to model many galaxies at once.

# Notes of interest:
# 1. This does *not* set up the parameters_master.py file: it is
# assumed that you will *very carefully* set this up yourself.

# 2. This requires bash versions >= 3.0. To check, type at the shell
# prompt:
# > echo $BASH_VERSION

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

# echo "processing model file for galaxy,snapshot:  $galaxy,$snap"

# Clear the pyc files
rm -f *.pyc

# Set up the model_**.py file
if [ ! -d "$model_dir" ]; then
    mkdir -p $model_dir
    out_folder="$model_dir/out"
    err_folder="$model_dir/err"
    mkdir -p $out_folder
    mkdir -p $err_folder
fi

# echo "model output dir is: $model_dir_remote"
if [ ! -d "$model_dir_remote" ]; then
    mkdir -p $model_dir_remote
fi

# Copy over parameters_master file (if needed)
# cp parameters_master.py $model_dir/.

filem="$model_dir/snap${snap}_${galaxy}.py"
# echo "writing to $filem"
rm -f $filem

if [ $N -gt 1000 ]; then
    Nout=1000
else
    Nout=${N}
fi

echo "# Snapshot Parameters" >> $filem
echo "# <Parameter File Auto-Generated by setup_all_local.sh>" >> $filem
echo "snapshot_num = $snap" >> $filem
echo "galaxy_num = $galaxy" >> $filem
echo -e "\n" >> $filem

echo "galaxy_num_str = '{:06.0f}'.format(galaxy_num)" >> $filem
echo "snapnum_str = '{:03.0f}'.format(snapshot_num)" >> $filem
echo -e "\n" >> $filem

if [ $COSMOFLAG -eq 1 ]; then
    echo "hydro_dir = '$hydro_dir_remote/'" >> $filem
    echo "snapshot_name = 'subset_'+galaxy_num_str+'.h5'" >> $filem
else
    echo "hydro_dir = '$hydro_dir_remote/'" >> $filem
    echo "snapshot_name = 'subset_'+galaxy_num_str+'.h5'" >> $filem
fi

echo -e "\n" >> $filem

echo "# Where the files should go" >> $filem
echo "PD_output_dir = '${model_dir_remote}/'" >> $filem
echo "Auto_TF_file = 'snap'+snapnum_str+'.logical'" >> $filem
echo "Auto_dustdens_file = 'snap'+snapnum_str+'.dustdens'" >> $filem

echo -e "\n\n" >> $filem
echo "# ===============================================" >> $filem
echo "# FILE I/O" >> $filem
echo "# ===============================================" >> $filem
echo "inputfile = PD_output_dir+'snap'+snapnum_str+'.galaxy'+galaxy_num_str+'.rtin'" >> $filem
echo "outputfile = PD_output_dir+'snap'+snapnum_str+'.galaxy'+galaxy_num_str+'.rtout'" >> $filem

echo -e "\n\n" >> $filem
echo "# ===============================================" >> $filem
echo "# GRID POSITIONS" >> $filem
echo "# ===============================================" >> $filem
echo "x_cent = ${xpos}" >> $filem
echo "y_cent = ${ypos}" >> $filem
echo "z_cent = ${zpos}" >> $filem

echo -e "\n\n" >> $filem
echo "# ===============================================" >> $filem
echo "# CMB INFORMATION" >> $filem
echo "# ===============================================" >> $filem
echo "TCMB = ${tcmb}" >> $filem

# Run the model locally
if [ "$job_flag" -eq 1 ]; then
    echo "Generating local run script"

    runfile="$model_dir/run_local.sh"
    rm -f $runfile
    echo $runfile

    echo "#!/bin/bash" >> $runfile
    echo "echo 'Starting simulation...'" >> $runfile
    echo "python /home/lorenzong/powderday/pd_front_end.py $model_dir parameters_master "snap${snap}_${galaxy}" > $model_dir/simulation.log" >> $runfile
    echo "echo 'Simulation complete'" >> $runfile

    chmod +x $runfile
    echo "Executing local run script..."
    $runfile
fi

#done