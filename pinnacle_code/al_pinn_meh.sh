#!/bin/bash

# Get the current directory of the script
current_dir=$(dirname -- "$0")

# Generate a timestamped log file name
timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
log_file="${current_dir}/script_${timestamp}.log"

# Redirect output and error to the timestamped log file
exec > >(tee -a "$log_file") 2>&1

# Log the current directory
echo "Current directory: $current_dir"

# COMMAND
# al_pinn.sh {tests to run} {methods to run} {repeats} {other args for python script}

# EXAMPLE COMMAND
# al_pinn.sh "0" "0 3 6" "0 1 2 3 4" "--pdebench_dir /home/a/apivich/pdebench"

# "--hidden_layers 8 --eqn conv-1d --use_pdebench --data_seed 40 --const 1.0 --train_steps 10000 --num_points 200 --mem_pts_total_budget 1000"
# "--hidden_layers 8 --eqn conv-1d --use_pdebench --data_seed 40 --const 1.0 --train_steps 200000 --num_points 200 --mem_pts_total_budget 1000"

pdes=(
    "--hidden_layers 4 --eqn poisson-2d  --data_seed 20 --const 1 --train_steps 20000 --num_points 100 --mem_pts_total_budget 300 --rand_res_prop 0.5 --scaling 10."
    "--hidden_layers 4 --eqn poisson-2d  --data_seed 20 --const 1 --train_steps 30000 --num_points 100 --mem_pts_total_budget 300 --rand_res_prop 0.5 --scaling 1."

    "--hidden_layers 4 --eqn burgers-1d --use_pdebench --data_seed 20 --const 0.02 --train_steps 100000 --num_points 100 --mem_pts_total_budget 300"
    "--hidden_layers 4 --eqn burgers-1d --use_pdebench --data_seed 20 --const 0.02 --train_steps 30000 --num_points 100 --mem_pts_total_budget 300"
    "--hidden_layers 4 --eqn burgers-1d --use_pdebench --data_seed 20 --const 0.02 --train_steps 200000 --num_points 100 --mem_pts_total_budget 300 --scaling 0.1"
    "--hidden_layers 4 --eqn burgers-1d --use_pdebench --data_seed 20 --const 0.02 --train_steps 30000 --num_points 100 --mem_pts_total_budget 300 --scaling 0.1"
)

algs=(
    "--method random --al_every 1000"
    "--method random --optim multiadam --al_every 1000"
    "--method random --autoscale_loss_w_bcs --al_every 1000"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --al_every 1000"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --optim multiadam --al_every 1000"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --autoscale_loss_w_bcs --al_every 1000"
    "--method random --lra_loss_w_bcs --al_every 1000"

    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --al_every 1000"
    "--method sampling --eig_weight_method alignment v --random_points_for_weights --eig_sampling pseudo --eig_memory --auto_al --autoscale_loss_w_bcs --al_every 1000"
    
    "--method random --sample_each_round"
    "--method random --optim multiadam --sample_each_round"
    "--method random --autoscale_loss_w_bcs --sample_each_round"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --optim multiadam"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --autoscale_loss_w_bcs"

    "--method sampling --optim multiadam --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --autoscale_loss_w_bcs"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --autoscale_loss_w_bcs --autoscale_first"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --autoscale_loss_w_bcs --random_points_for_weights"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --autoscale_loss_w_bcs --random_points_for_weights --autoscale_first"
)

losses=(
    "--loss_w_bcs 1.0"
)

for j in $3; do

    for k in $1; do

        pde="${pdes[$k]}"
        echo "PDE params: $pde"

        for loss in "${losses[@]}"; do

            for m in $2; do

                alg="${algs[$m]}"

                pdeargs="$pde $alg $loss $4"

                python -u ${current_dir}/al_pinn.py $pdeargs
                
            done

        done

    done

done