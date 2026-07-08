
DATA_DIR="/eos/home-b/bdudar/mlpf/data/ILD/processed/zuds_from_dolores/parquet/"
CFG_DATA="config_files/config_hits_track_v4.yaml"
CFG_NET="src/models/wrapper/example_mode_gatr_noise.py"
MODEL_PREFIX="/eos/user/m/mgarciam/datasets_mlpf/models_trained_ILD/020426/"
mkdir -p "${MODEL_PREFIX}"

LOG_FILE="${MODEL_PREFIX}/train_$(date +%Y%m%d_%H%M%S).log"

python -m src.train_lightning1 \
	    --data-train "${DATA_DIR}" \
	        --data-config "${CFG_DATA}" \
		    --network-config "${CFG_NET}" \
		        --model-prefix "${MODEL_PREFIX}/" \
			    --num-workers 4 \
			        --gpus 0,1,2,3 \
				    --batch-size 20 \
				        --start-lr 1e-3 \
					    --num-epochs 10 \
					        --optimizer ranger \
						    --fetch-step 4 \
						        --condensation \
							    --log-wandb \
							        --wandb-displayname Zuds_ILD \
								    --wandb-projectname mlpf_debug \
								        --wandb-entity ml4hep \
									    --frac_cluster_loss 0 \
									        --qmin 3 \
										    --use-average-cc-pos 0.98 \
										        --tracks \
											    --train-val-split 0.98 \
											        --fetch-by-files --ILD \
												    --train-batches 11000 \
    2>&1 | tee "${LOG_FILE}"