# MLPF FCC
Machine learning based pipeline for particle flow at FCC. 

> **DELPHI (branch `delphi-mask3d`).** This branch adds training and evaluation on the DELPHI detector at LEP: HitPF and
> Mask3D clustering, a shared energy-correction/PID stage with a noise class, and BSC (MareNostrum 5) job scripts.
> Start at [`scripts/bsc/README.md`](scripts/bsc/README.md).

Latest documentation/ presentations can be found here:
- [HitPF Note](https://repository.cern/records/n9wc2-09n03)
- [Talk at FCC workshop](https://indico.cern.ch/event/1588696/contributions/6857314/attachments/3209610/5716524/MachineLearning_Physics_Workshop_29_01.pptx-1.pdf)

Additional information and setup instructions can be found in the wiki!


## ML pipeline:
- One event is formed by hits (which can be tracks or calo hits). An input is an event in the form of a graph, and the output is a single particle (in coming versions of the dataset there will be more). 
- Models: The goal of the current model is to first custer and then regress the particle's information (PID and energy). Currently the best approach is the [object condensation](https://arxiv.org/abs/2002.03605), since it allows to regress a variable number of particles. 
- The pipeline includes the following four steps: training the clustering, training energy correction and PID, evaluation, plotting 
- Training: To train a model check the wiki/Training section 


## Model logging 
Runs for this project can be found in the following work space: https://wandb.ai/imdea_dolo/mlpf?workspace=user-imdea_dolo

## Environment 
You can use the docker container ```docker://dologarcia/gatr:v9``` and use singularity to run the container as:
 ```export APPTAINER_CACHEDIR=/home/ADDUSER/cache/```
 ```singularity  shell   -B /eos -B /afs  --nv docker://dologarcia/gatr:v9```

You might need to set up a conda env to run notebooks. To set up the env create a conda env following the instructions from [Weaver](https://github.com/hqucms/weaver-core/tree/main) and also install the packages in the requirements.sh script above 
Alternatively, you can try to use a pre-built environment from [this link](https://cernbox.cern.ch/s/Rwz2S35BUePbwG4) - the .tar.gz file was built using conda-pack on fcc-gpu-04v2.cern.ch.




