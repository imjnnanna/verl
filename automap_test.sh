 #!/bin/bash                                                                                                                                                             
  set -x          

  ARGS=(
      algorithm.adv_estimator=gae
      data.train_files=$HOME/data/gsm8k/train.parquet                                                                                                                     
      data.val_files=$HOME/data/gsm8k/test.parquet
      data.train_batch_size=64                                                                                                                                            
      data.max_prompt_length=512                                                                                                                                          
      data.max_response_length=512
      data.filter_overlong_prompts=True                                                                                                                                   
      data.truncation=error
      actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
      actor_rollout_ref.actor.optim.lr=1e-6                                                                                                                               
      actor_rollout_ref.model.use_remove_padding=True
      actor_rollout_ref.actor.ppo_mini_batch_size=32                                                                                                                      
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
      actor_rollout_ref.actor.fsdp_config.param_offload=False                                                                                                             
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
      actor_rollout_ref.actor.use_kl_loss=False                                                                                                                           
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
      actor_rollout_ref.rollout.tensor_model_parallel_size=1                                                                                                              
      actor_rollout_ref.rollout.name=vllm
      actor_rollout_ref.rollout.gpu_memory_utilization=0.4                                                                                                                
      critic.optim.lr=1e-5
      critic.model.use_remove_padding=True                                                                                                                                
      critic.model.path=Qwen/Qwen2.5-0.5B-Instruct                                                                                                                        
      critic.ppo_micro_batch_size_per_gpu=2
      critic.fsdp.param_offload=False                                                                                                                                     
      critic.fsdp.optimizer_offload=False                                                                                                                                 
      algorithm.use_kl_in_reward=False
      trainer.critic_warmup=0                                                                                                                                             
      trainer.logger=[console]                                                                                                                                            
      trainer.project_name=auto_mapping_smoke
      trainer.experiment_name=single_node_8gpu                                                                                                                            
      trainer.n_gpus_per_node=8
      trainer.nnodes=1                                                                                                                                                    
      trainer.total_epochs=1
      trainer.save_freq=-1                                                                                                                                                
      trainer.test_freq=-1
      +trainer.auto_mapping.enable=true                                                                                                                                   
      +trainer.auto_mapping.per_gpu_budget_gb=80
      +trainer.auto_mapping.bandwidth.intra_host=600                                                                                                                      
      +trainer.auto_mapping.bandwidth.intra_block=25                                                                                                                      
      +trainer.auto_mapping.bandwidth.inter_block=12
  )                                                                                                                                                                       
                  
  python3 -m verl.trainer.main_ppo "${ARGS[@]}"
