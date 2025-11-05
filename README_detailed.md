## Experiments 
The following scripts construct and test a Markovian queueing network with a parent queue that probabilistically routes jobs to two child queues. The configurations for all experiments are managed in the `configs` folder. All scripts accept two arguments: the configuration file and the experiment number. The user can also specify a '-v' option for verbose print statements.

1. Update the configuration file to generate data using the Markovian queueing simulator: `markovian_simulator.yaml`. The settings are explained below. Each experiment is labeled `experiment_<experiment_number>`. 

    1.1. `experiment_design`: full-factorial, random-sampling or random-sampling-stable

    1.2 `arrival_rates`: Arrival rates, for full-factorial specify all the possible arrival rates as a list. In case we specify min and max, the first item in the list corresponds to min and the second max. 

    1.3 `parent_service_rates`: Service rates for the parent.  

    1.4 `child1_service_rates`: Service rates for child 1 queue.

    1.5 `child2_service_rates`: Service rates for child 2 queue. 

    1.6 `routing_probability`: Routing probability to child 1 queue (List possible values or specify min and max)

    1.7 `replications`: Number of replications for each experimental design point

    1.8 `configurations`: Number of possible design points (for random-sampling and random-sampling-stable only)

    1.9 `simulation_end`: Simulation end time

    1.10 `varying_iql`: Flag to specify whether initial queue length varies 

    1.11 `max_iql`: If varying_iql = True, specify the maximum initial queue length 

    1.12 `output_folder`: Specify the location where the time series event data is stored. 

    Running script on cluster: `cluster/scripts/gen_qnetwork_simdata.py`. 

    Running script locally: `src/mm1_qnetwork_simulator.py`.

2. Extract the time-sampled data sets from the continuous time simulator data. The configuration file is `time_discretization.yaml`. The parameters of the config are explained below. Every experiment is denoted by the key `experiment_<experiment_number>`. 

    2.1 `time_series_experiment`: The experiment number of the data generation procedure as explained in Step 1. 

    2.2 `time_discretization_folder`: The location of the discretized time data. 

    2.3 `sampling_interval`: The sampling interval used for time discretization.

    Running script on cluster: `cluster/scripts/gen_discrete_simdata.py`. 
    
    Running script locally: `subsampling_discrete_2tbn.py` and `subsampling_discrete_timeseries.py`. 

3. Construct the DBN with a specific structure as defined in `construct_dbn_mm1_qnetwork.py`. The structure of the DBN is manually specified in this file. The configuration file for the rest of parameters is `markovian_qnetwork_dbns.yaml`. 

    3.1 `time_discretization_experiment`: The experiment number of the time discretization procedure as referenced in Step 2. 

    3.2 `dbn_output_folder`: Location of the output files for the constructed DBN

    3.3 `dbn_unique_id`: Unique identifier for the DBN (structures are manually specified in the script)

    3.4 `maximum_qlength`: Value for the maximum qlength for all queue length variables. 

    Running script locally: `src/construct_dbn_mm1_qnetwork.py`. 
    
    Running script on cluster: `cluster/scripts/construct_dbn.py` 

4. Specify the details of each query in the configuration file `inference_queries.yaml`. Each query is referenced by `experiment_<experiment_number>`. The details are listed below.

    4.1 `start_parameters`: The parameters that the network is initialized to.  
        
        4.1.1 `Lambdap`: Parent arrival rate

        4.1.2 `Mup`: Parent service rate 

        4.1.3 `Mufc`: Child 1 service rate 

        4.1.4 `Musc`: Child 2 service rate

        4.1.5 `Rp`: Routing probability to child 1 

        4.1.6 `Lp`: Initial state of the parent system

        4.1.7 `Lfc`: Initial state of the child 1 system

        4.1.8 `Lsc`: Initial state of the child 2 system 

    4.2 `inference_algorithm`: The algorithm that will be used for inference. Can be one of `LoopyBeliefPropagation`, `LazyPropagation`, `VariableElimination` ... (add the others here)

    4.3 `interventions`: The list of interventions (or evidence) that will be provided to the model. Each intervention has the following keys. 

        4.3.1 `intervention_start`: Start time of the intervention

        4.3.2 `intervention_variable`: Name of the variable to be intervened on. Can be one of `Lambdap, Mup, Mufc, Musc, Rp, Lp, Lfc, Lsc`. 

        4.3.3 `intervention_value`: The value to set the variable to. 

        4.3.4 `intervention_type`: The type of possible interventions. Can be one of `conditional, parameter_intervention, additive, subtractive, interventional`

    4.4 `query_variable`: The variable that is queried

    4.5 `query_time`: The time at which the query variables probability distribution is requested. 

    4.6 `dbn_experiment_number`: The experiment number referencing the construction of the DBN as in Step 3. 

    4.7 `results_folder`: Location where the posterior distribution is stored.

    4.8 `gt_replications`: The number of replications to compute the ground truth

    4.9 `gt_results_folder`: Location of the ground truth probability distribution 

    4.10 `figures_folder`: Location where the figures comparing DBN and ground truth is stored. 

    Local scripts: 

    - Inference on DBN: `src/inference_dbn_mm1_qnetwork.py` 
    
    - Ground truth: `src/mm1_qnetwork_simulator_interventions.py`

    - Comparing DBN results with ground truth: `utils/compare_mm1_qnetwork_dbn_gt.py`. 

    Cluster scripts: 

    - Inference on DBN: `cluster/scripts/inference_dbn.sh`

    - Ground truth: `cluster/scripts/gen_qnetwork_gt.sh`

    - Comparing DBN results with ground truth: `cluster/scripts/compare_dbn_gt.sh`

