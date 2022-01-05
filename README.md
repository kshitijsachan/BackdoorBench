# bdzoo2
- attack : all attack should be put here separately
- defense : all defense should be put here separately 
- config : all config file in yaml (all attack and defense config should all be put here separately)
- data : (only) data file 
- experiment : analysis script and the final main entry will be put here 
- models : models that do not in the torchvision
- record : all experiment generated files and logs
- utils : frequent-use functions and other tools
  - aggregate_block : frequent-use blocks in script
  - bd_img_transform : basic perturbation on img
  - bd_label_transform : basic transform on label
  - bd_groupwise_transform : for special case, such that data poison must be carried out groupwise, eg. HiddenTriggerBackdoorAttacks
  - bd_trainer : the training process can replicate for attack (for re-use, eg. noise training)
  - dataset : script for loading the dataset
  - dataset_preprocess : script for preprocess transforms on dataset 
  - backdoor_generate_pindex.py : some function for generation of poison index 
  - bd_dataset.py : the wrapper of backdoored datasets 
  - trainer_cls.py : some basic functions for classification case
- resource : pre-trained model (eg. auto-encoder for attack), or other large file (other than data)
